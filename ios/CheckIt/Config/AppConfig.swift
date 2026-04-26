import Foundation

/// Runtime tunables. Single source of truth for non-prompt, non-color constants.
/// Edit values here — never inline magic numbers in services or views.
enum AppConfig {
    // Detection / classification
    static let detectionConfidenceThreshold: Float = 0.75
    static let plantClassificationAcceptanceThreshold: Float = 0.30
    /// Hard cap on the number of boxes shown per frame (highest-confidence kept).
    static let maxDetectionsPerFrame: Int = 3
    static let frameStrideHz: Double = 30.0
    static let inferenceTimeoutMs: Int = 1500

    // Welcome / loading
    static let welcomeDurationSeconds: Double = 3.0

    // Gesture
    static let holdMinimumDuration: Double = 0.5
    /// Sized to match `whisperWindowSeconds` exactly so there is never any chunking.
    static let holdHardCapSeconds: Double = 30.0
    static let tripleTapWindowSeconds: Double = 0.5

    // Box tracker
    static let boxIoUMatchThreshold: Float = 0.4
    static let boxMaxMissedFrames: Int = 5

    // Transcript layout
    static let transcriptDividerPaddingTop: CGFloat = 16
    static let transcriptDividerPaddingBottom: CGFloat = 8

    // Whisper preprocessing
    static let whisperSampleRate: Double = 16_000
    static let whisperWindowSeconds: Int = 30
    static let whisperMelBins: Int = 80
    static let whisperFFTSize: Int = 400
    static let whisperHopSize: Int = 160
    static let whisperMaxDecodeTokens: Int = 224

    // Performance targets (used by debug HUD)
    static let targetFPS: Double = 30.0
    static let targetDetectionToLabelLatencyMs: Int = 400
}
