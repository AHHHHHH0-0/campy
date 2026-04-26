import Foundation
import QuartzCore
import Observation

/// Local-only debug HUD telemetry. No external sinks. Mutated from inference workers
/// via `@MainActor` hop on update.
@MainActor
@Observable
final class InferenceTelemetry {

    var fps: Double = 0
    var lastInferenceLatencyMs: Double = 0
    var lastDetectionClass: String = "—"
    var lastScientificName: String = "—"
    var lastLookupStatus: String = "—"
    var gemmaCacheSize: Int = 0
    var lastChatLatencyMs: Double = 0
    var lastPackLatencyMs: Double = 0
    var lastWhisperLatencyMs: Double = 0
    var lastWhisperFailureReason: String? = nil

    private var lastFrameTimestamp: CFTimeInterval = 0
    private var smoothedFPS: Double = 0

    func recordFrame() {
        let now = CACurrentMediaTime()
        if lastFrameTimestamp > 0 {
            let dt = now - lastFrameTimestamp
            if dt > 0 {
                let instant = 1.0 / dt
                let alpha = 0.15
                smoothedFPS = smoothedFPS * (1 - alpha) + instant * alpha
                fps = smoothedFPS
            }
        }
        lastFrameTimestamp = now
    }
}
