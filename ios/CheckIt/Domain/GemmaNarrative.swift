import Foundation

/// Final paragraph emitted by `GemmaReasoningService` for one detail-sheet open.
/// Cached in memory keyed by scientific name (or YOLO class for non-plant), not
/// persisted to disk in v1.
struct GemmaNarrative: Codable, Equatable, Sendable {
    let key: String
    let paragraph: String
    let generatedAt: Date

    init(key: String, paragraph: String, generatedAt: Date = .init()) {
        self.key = key
        self.paragraph = paragraph
        self.generatedAt = generatedAt
    }
}
