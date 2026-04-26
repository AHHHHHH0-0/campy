import Foundation

protocol GemmaReasoningServiceProtocol: AnyObject, Sendable {
    /// Token-streamed paragraph for the in-pack narrator path. Emits tokens as
    /// they arrive; the caller appends them to the detail sheet text in real time.
    func narrate(plant entry: RegionPack.Entry, prepBlurb: String?) -> AsyncStream<String>

    /// Token-streamed paragraph for the not-found narrator path.
    func narrateNotFound(scientificName: String) -> AsyncStream<String>

    /// Cached final paragraph keyed by scientific name (or YOLO class for non-plant).
    func cachedNarrative(forKey key: String) -> GemmaNarrative?

    /// Updates the cache and returns the stored narrative.
    func cache(narrative: GemmaNarrative)

    /// Clears the in-memory cache (called by the triple-tap clear flow).
    func clearCache()
}
