import Foundation

protocol WhisperTranscriberServiceProtocol: AnyObject, Sendable {
    /// Single non-throwing entrypoint. All internal failures (resample, mel,
    /// encoder, decoder, detokenize) are caught and surfaced as an empty string —
    /// the caller routes that through the chat-call system prompt's empty-input path.
    func transcribe(audio: CapturedAudio) async -> String
}
