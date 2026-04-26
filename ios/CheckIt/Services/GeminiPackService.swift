import Foundation

/// Two parallel `gemini-2.5-pro` calls per user turn:
/// - Chat call: free-text reply governed by `geminiChatSystemPrompt`.
/// - Pack-update call: `responseSchema` JSON returning a fresh `RegionPack`.
///
/// Both calls use the same conversation history (post-Whisper user turns + prior
/// assistant replies). The pack-update call is skipped when the trimmed
/// transcript is empty. On any quota / unrecoverable network error, both calls
/// are abandoned and `cannedQuotaReply` is surfaced as the chat reply.
final class GeminiPackService: GeminiServiceProtocol, @unchecked Sendable {

    private let packStore: any PackStoreProtocol
    private let transcriptStore: any TranscriptStoreProtocol
    private let urlSession: URLSession

    init(
        packStore: any PackStoreProtocol,
        transcriptStore: any TranscriptStoreProtocol,
        urlSession: URLSession = .shared
    ) {
        self.packStore = packStore
        self.transcriptStore = transcriptStore
        self.urlSession = urlSession
    }

    func dispatch(transcript rawTranscript: String) async -> GeminiTurnResult {
        let transcript = Self.applyDenylist(to: rawTranscript)
        let history = await transcriptStore.snapshot()

        async let chat = chatCall(userTurn: transcript, history: history)
        async let pack = packCall(userTurn: transcript, history: history)
        let chatOutcome = await chat
        let packOutcome = await pack

        // Quota / network failure on the chat call swaps in the canned reply.
        if case .quota = chatOutcome.kind {
            return GeminiTurnResult(
                chatReply: PromptConfig.cannedQuotaReply,
                updatedPack: nil,
                chatLatencyMs: chatOutcome.latencyMs,
                packLatencyMs: nil,
                canned: true
            )
        }

        let updatedPack: RegionPack?
        if case .pack(let p) = packOutcome.kind {
            updatedPack = p
            if let p { try? await packStore.write(pack: p) }
        } else {
            updatedPack = nil
        }

        let chatReply: String
        switch chatOutcome.kind {
        case .text(let reply): chatReply = reply
        default: chatReply = PromptConfig.cannedQuotaReply
        }

        return GeminiTurnResult(
            chatReply: chatReply,
            updatedPack: updatedPack,
            chatLatencyMs: chatOutcome.latencyMs,
            packLatencyMs: packOutcome.latencyMs,
            canned: false
        )
    }

    // MARK: Internals

    private struct CallOutcome {
        enum Kind {
            case text(String)
            case pack(RegionPack?)
            case quota
        }
        let kind: Kind
        let latencyMs: Double
    }

    private func chatCall(userTurn: String, history: [ModelReply]) async -> CallOutcome {
        let start = Date()
        do {
            let body = chatRequestBody(userTurn: userTurn, history: history)
            let response: GeminiChatResponse = try await postJSON(
                model: "gemini-2.5-pro",
                body: body,
                expecting: GeminiChatResponse.self
            )
            let latency = Date().timeIntervalSince(start) * 1000
            let text = response.firstText() ?? ""
            return CallOutcome(kind: .text(text), latencyMs: latency)
        } catch GeminiError.quota {
            return CallOutcome(kind: .quota, latencyMs: Date().timeIntervalSince(start) * 1000)
        } catch {
            print("[GeminiPackService] chatCall error: \(error)")
            return CallOutcome(kind: .quota, latencyMs: Date().timeIntervalSince(start) * 1000)
        }
    }

    private func packCall(userTurn: String, history: [ModelReply]) async -> CallOutcome {
        let start = Date()
        let trimmed = userTurn.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || !history.isEmpty else {
            return CallOutcome(kind: .pack(nil), latencyMs: 0)
        }
        var attempt = 0
        var lastError: Error?
        while attempt < 2 {
            attempt += 1
            do {
                let body = packRequestBody(userTurn: userTurn, history: history, retryHint: lastError?.localizedDescription)
                let response: GeminiChatResponse = try await postJSON(
                    model: "gemini-2.5-pro",
                    body: body,
                    expecting: GeminiChatResponse.self
                )
                guard let raw = response.firstText(),
                      let data = raw.data(using: .utf8) else {
                    lastError = GeminiError.parse("missing JSON payload")
                    continue
                }
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let pack = try decoder.decode(RegionPack.self, from: data)
                return CallOutcome(kind: .pack(pack), latencyMs: Date().timeIntervalSince(start) * 1000)
            } catch GeminiError.quota {
                return CallOutcome(kind: .quota, latencyMs: Date().timeIntervalSince(start) * 1000)
            } catch {
                lastError = error
                continue
            }
        }
        return CallOutcome(kind: .pack(nil), latencyMs: Date().timeIntervalSince(start) * 1000)
    }

    // MARK: Request bodies

    private func chatRequestBody(userTurn: String, history: [ModelReply]) -> [String: Any] {
        // Gemini requires strictly alternating user/model turns starting with user.
        // TranscriptStore only persists assistant replies (no user turns), so we
        // cannot reconstruct valid multi-turn contents. We inject prior replies into
        // the system instruction instead, which is safe and keeps full context.
        let systemText = history.isEmpty
            ? PromptConfig.geminiChatSystemPrompt
            : PromptConfig.geminiChatSystemPrompt + Self.historyFootnote(history)

        return [
            "system_instruction": ["parts": [["text": systemText]]],
            "contents": [["role": "user", "parts": [["text": userTurn]]]],
            "generationConfig": [
                "temperature": 0.4,
                "maxOutputTokens": 600
            ]
        ]
    }

    private func packRequestBody(userTurn: String, history: [ModelReply], retryHint: String?) -> [String: Any] {
        let systemText = history.isEmpty
            ? PromptConfig.geminiPackSystemPrompt
            : PromptConfig.geminiPackSystemPrompt + Self.historyFootnote(history)

        var userText = userTurn
        if let retryHint, !retryHint.isEmpty {
            userText += "\n\n[previous JSON failed validation: \(retryHint)]"
        }
        return [
            "system_instruction": ["parts": [["text": systemText]]],
            "contents": [["role": "user", "parts": [["text": userText.isEmpty ? "Update the pack." : userText]]]],
            "generationConfig": [
                "temperature": 0.2,
                "responseMimeType": "application/json",
                "responseSchema": PromptConfig.geminiPackResponseSchema
            ]
        ]
    }

    /// Appends prior assistant replies as a read-only conversation log in the
    /// system instruction, capped to the most recent 6 to stay within token budget.
    private static func historyFootnote(_ history: [ModelReply]) -> String {
        let recent = history.suffix(6)
        let lines = recent.map { "- \($0.text)" }.joined(separator: "\n")
        return "\n\nPrior assistant replies (for context only):\n\(lines)"
    }

    // MARK: HTTP

    private func postJSON<R: Decodable>(
        model: String,
        body: [String: Any],
        expecting: R.Type
    ) async throws -> R {
        let url = URL(string: "https://generativelanguage.googleapis.com/v1beta/models/\(model):generateContent?key=\(SecretsLoader.geminiAPIKey)")!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body, options: [])
        let (data, response) = try await urlSession.data(for: req)
        if let http = response as? HTTPURLResponse {
            if http.statusCode == 429 || http.statusCode == 403 {
                throw GeminiError.quota
            }
            guard (200..<300).contains(http.statusCode) else {
                throw GeminiError.http(http.statusCode)
            }
        }
        return try JSONDecoder().decode(R.self, from: data)
    }

    // MARK: Whisper hallucination filter

    static func applyDenylist(to transcript: String) -> String {
        let trimmed = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "" }
        let lowered = trimmed.lowercased()
        let punctuationStripped = lowered.unicodeScalars.filter { !CharacterSet.punctuationCharacters.contains($0) }
        let cleaned = String(String.UnicodeScalarView(punctuationStripped))
            .trimmingCharacters(in: .whitespaces)
        for entry in PromptConfig.whisperHallucinationDenylist {
            let entryClean = entry.lowercased().trimmingCharacters(in: .whitespaces)
            if cleaned == entryClean { return "" }
        }
        return trimmed
    }
}

private enum GeminiError: Error {
    case quota
    case http(Int)
    case parse(String)
}

/// Minimal subset of Gemini's REST response we care about.
private struct GeminiChatResponse: Decodable {
    let candidates: [Candidate]?

    struct Candidate: Decodable {
        let content: Content?
    }
    struct Content: Decodable {
        let parts: [Part]?
    }
    struct Part: Decodable {
        let text: String?
    }

    func firstText() -> String? {
        candidates?.first?.content?.parts?.compactMap { $0.text }.joined()
    }
}

// MARK: Stub for previews

final class StubGeminiService: GeminiServiceProtocol, @unchecked Sendable {
    func dispatch(transcript: String) async -> GeminiTurnResult {
        GeminiTurnResult(
            chatReply: "Sounds like you're testing previews.\n\nThis is a stub Gemini reply.\n\nWhere are you headed?",
            updatedPack: nil,
            chatLatencyMs: 0,
            packLatencyMs: 0,
            canned: false
        )
    }
}
