import Foundation

/// One entry in the persisted assistant transcript. User turns are never modeled —
/// only the assistant's replies are visible. Persisted to
/// `Application Support/transcript.json` via atomic writes.
struct ModelReply: Codable, Equatable, Sendable, Identifiable {
    let id: UUID
    let createdAt: Date
    let text: String

    init(id: UUID = UUID(), createdAt: Date = .init(), text: String) {
        self.id = id
        self.createdAt = createdAt
        self.text = text
    }
}
