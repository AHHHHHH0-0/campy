import Foundation

/// One stable detection emitted by `BoxTracker`. The IoU matcher assigns a
/// stable `id` across frames so labels and tap targets do not strobe at 30 fps.
struct TrackedDetection: Identifiable, Equatable, Sendable {
    let id: UUID
    let yoloClass: String
    var confidence: Float
    var bbox: CGRect
    var lastSeenFrame: Int

    init(id: UUID = UUID(), yoloClass: String, confidence: Float, bbox: CGRect, lastSeenFrame: Int) {
        self.id = id
        self.yoloClass = yoloClass
        self.confidence = confidence
        self.bbox = bbox
        self.lastSeenFrame = lastSeenFrame
    }
}

/// Raw per-frame YOLO output before tracking — fed into `BoxTracker.update(...)`.
struct RawDetection: Equatable, Sendable {
    let yoloClass: String
    let confidence: Float
    let bbox: CGRect
}
