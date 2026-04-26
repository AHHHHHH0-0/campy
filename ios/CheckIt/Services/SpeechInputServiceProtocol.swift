import AVFoundation
import Foundation

/// Captured push-to-talk audio. Whisper expects mono float32 PCM @ 16 kHz; this
/// service returns the raw device-native buffer plus the source rate so the
/// transcriber can resample once at dispatch.
struct CapturedAudio: Equatable, Sendable {
    let pcm: [Float]
    let sampleRate: Double
    let channelCount: UInt32
}

protocol SpeechInputServiceProtocol: AnyObject, Sendable {
    /// Smoothed RMS level (0…1) tapped off the input bus while a hold is active.
    /// `AssistantView` binds the audio visualizer to this stream.
    var levelStream: AsyncStream<Double> { get }

    /// Emits exactly once per hold lifecycle when capture ends abnormally
    /// (audio interruption, scene-phase background, drag-off cancel from caller).
    var cancelStream: AsyncStream<Void> { get }

    @MainActor
    func start() async throws

    @MainActor
    func stop() async -> CapturedAudio?

    @MainActor
    func cancel()
}
