import Foundation
import CoreVideo

/// Per-frame orchestrator for the offline identify pipeline.
/// - Runs YOLO11.
/// - Tracks boxes across frames for stable IDs.
/// - Emits neutral detection states for live overlay labels/colors.
actor InferenceWorker {

    struct FrameOutcome: Sendable {
        let detections: [TrackedDetection]
        let states: [UUID: DetectionState]
        let frameSize: CGSize
    }

    private let detection: any ObjectDetectionServiceProtocol
    private let tracker: BoxTracker
    private let telemetry: InferenceTelemetry

    init(
        detection: any ObjectDetectionServiceProtocol,
        tracker: BoxTracker,
        telemetry: InferenceTelemetry
    ) {
        self.detection = detection
        self.tracker = tracker
        self.telemetry = telemetry
    }

    func process(frame: CameraFrame) async -> FrameOutcome? {
        let started = Date()
        guard let raws = await detection.detect(in: frame) else { return nil }
        let tracked = await tracker.update(with: raws)
        let frameSize = CGSize(
            width: CVPixelBufferGetWidth(frame.pixelBuffer),
            height: CVPixelBufferGetHeight(frame.pixelBuffer)
        )

        var states: [UUID: DetectionState] = [:]
        for det in tracked {
            // Live path is YOLO-only: every visible detection gets a neutral state.
            states[det.id] = .notFood(yoloClass: det.yoloClass)
        }

        let elapsedMs = Date().timeIntervalSince(started) * 1000
        Task { @MainActor in
            telemetry.recordFrame()
            telemetry.lastInferenceLatencyMs = elapsedMs
        }
        return FrameOutcome(detections: tracked, states: states, frameSize: frameSize)
    }

    func reset() async {
        await tracker.reset()
    }
}
