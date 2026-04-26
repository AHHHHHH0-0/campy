import Foundation

/// Final outcome of one user turn against Gemini. Carries the chat reply that gets
/// appended to the transcript and (optionally) a refreshed region pack.
struct GeminiTurnResult: Equatable, Sendable {
    let chatReply: String
    let updatedPack: RegionPack?
    let chatLatencyMs: Double
    let packLatencyMs: Double?
    let canned: Bool
}

protocol GeminiServiceProtocol: AnyObject, Sendable {
    /// Fires the parallel chat + pack-update calls against `gemini-2.5-pro`.
    /// `transcript` is the post-Whisper, post-denylist user turn. The full
    /// conversation history is sourced internally from `TranscriptStore`.
    func dispatch(transcript: String) async -> GeminiTurnResult
}
