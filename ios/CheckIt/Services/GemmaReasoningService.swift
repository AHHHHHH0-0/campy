import Foundation
import os

#if canImport(ZeticMLange)
@preconcurrency import ZeticMLange
#endif

/// Wraps the Zetic-hosted Gemma LLM (`changgeun/gemma-4-E2B-it`).
/// Two prompt paths (in-pack narrator, not-found narrator) plus an in-memory
/// final-paragraph cache keyed by scientific name / YOLO class.
final class GemmaReasoningService: GemmaReasoningServiceProtocol, @unchecked Sendable {

    private let modelLoader: ModelLoader
    private let cacheLock = OSAllocatedUnfairLock<[String: GemmaNarrative]>(initialState: [:])
    private let inferenceQueue = DispatchQueue(label: "ai.checkit.gemma.inference", qos: .userInitiated)

    init(modelLoader: ModelLoader) {
        self.modelLoader = modelLoader
    }

    func narrate(plant entry: RegionPack.Entry, prepBlurb: String?) -> AsyncStream<String> {
        let prompt = PromptConfig.gemmaTapPromptTemplate
            .replacingOccurrences(of: "{scientific_name}", with: entry.scientificName)
            .replacingOccurrences(of: "{common_name}", with: entry.commonName)
            .replacingOccurrences(of: "{category}", with: entry.category.rawValue)
            .replacingOccurrences(of: "{rationale}", with: entry.rationale)
            .replacingOccurrences(of: "{prep_notes}", with: entry.prepNotes ?? "—")
            .replacingOccurrences(of: "{prep_blurb}", with: prepBlurb ?? "—")
        return streamTokens(prompt: prompt, cacheKey: entry.scientificName.lowercased())
    }

    func narrateNotFound(scientificName: String) -> AsyncStream<String> {
        let prompt = PromptConfig.gemmaNotFoundPromptTemplate
            .replacingOccurrences(of: "{scientific_name}", with: scientificName)
        return streamTokens(prompt: prompt, cacheKey: "notfound:\(scientificName.lowercased())")
    }

    func narrateObject(yoloClass: String) -> AsyncStream<String> {
        let prompt = PromptConfig.gemmaObjectPromptTemplate
            .replacingOccurrences(of: "{yolo_class}", with: yoloClass)
        return streamTokens(prompt: prompt, cacheKey: "notfood:\(yoloClass.lowercased())")
    }

    func cachedNarrative(forKey key: String) -> GemmaNarrative? {
        cacheLock.withLock { $0[key.lowercased()] }
    }

    func cache(narrative: GemmaNarrative) {
        cacheLock.withLock { $0[narrative.key.lowercased()] = narrative }
    }

    func clearCache() {
        cacheLock.withLock { $0.removeAll() }
    }

    // MARK: Internals

    private func streamTokens(prompt: String, cacheKey: String) -> AsyncStream<String> {
        AsyncStream { continuation in
            // Cache hit: emit the stored paragraph as a single chunk.
            if let cached = cachedNarrative(forKey: cacheKey) {
                continuation.yield(cached.paragraph)
                continuation.finish()
                return
            }
            // Fetch the model on the MainActor *before* hopping to the inference
            // queue. MainActor.assumeIsolated would crash on a background thread.
            Task { @MainActor [weak self] in
                guard let self else { continuation.finish(); return }
#if canImport(ZeticMLange)
                let model = self.modelLoader.gemma
                self.inferenceQueue.async { [weak self] in
                    guard let self else { continuation.finish(); return }
                    self.runOnce(model: model, prompt: prompt, cacheKey: cacheKey, continuation: continuation)
                }
#else
                continuation.finish()
#endif
            }
        }
    }

#if canImport(ZeticMLange)
    private func runOnce(model: ZeticMLangeLLMModel?, prompt: String, cacheKey: String, continuation: AsyncStream<String>.Continuation) {
        guard let model else {
            continuation.finish()
            return
        }
        do {
            _ = try model.run(prompt)
        } catch {
            print("[GemmaReasoningService] run error: \(error)")
            continuation.finish()
            return
        }
        var paragraph = ""
        while true {
            let result = model.waitForNextToken()
            if result.generatedTokens == 0 { break }
            paragraph.append(result.token)
            continuation.yield(result.token)
        }
        cache(narrative: GemmaNarrative(key: cacheKey, paragraph: paragraph))
        try? model.cleanUp()
        continuation.finish()
    }
#endif
}

