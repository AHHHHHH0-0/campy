import Foundation

protocol TranscriptStoreProtocol: AnyObject, Sendable {
    /// Decode `Application Support/transcript.json` if present. Returns `[]`
    /// when the file is missing or malformed.
    func load() async -> [ModelReply]

    /// Snapshot of the current in-memory transcript.
    func snapshot() async -> [ModelReply]

    /// Append a reply, atomically rewrite the entire array to disk.
    func append(_ reply: ModelReply) async

    /// Delete the file and clear in-memory state.
    func clear() async
}
