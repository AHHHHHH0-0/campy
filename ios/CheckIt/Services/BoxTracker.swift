import Foundation

/// IoU-based per-detection tracker. Keys raw detections to stable UUIDs across
/// frames so labels and tap targets do not strobe at 30 fps. Boxes survive up to
/// `boxMaxMissedFrames` consecutive misses before being dropped.
actor BoxTracker {

    private var tracked: [TrackedDetection] = []
    private var currentFrameIndex: Int = 0

    func update(with raws: [RawDetection]) -> [TrackedDetection] {
        currentFrameIndex += 1
        var unmatched = Array(raws.indices)

        // Match each existing track to the best IoU candidate of the same class.
        for trackIdx in tracked.indices {
            let track = tracked[trackIdx]
            var best: (Int, Float)? = nil
            for i in unmatched {
                let raw = raws[i]
                guard raw.yoloClass == track.yoloClass else { continue }
                let score = Self.iou(track.bbox, raw.bbox)
                if score >= AppConfig.boxIoUMatchThreshold,
                   best == nil || score > best!.1 {
                    best = (i, score)
                }
            }
            if let (matchedIdx, _) = best {
                let raw = raws[matchedIdx]
                tracked[trackIdx].bbox = raw.bbox
                tracked[trackIdx].confidence = raw.confidence
                tracked[trackIdx].lastSeenFrame = currentFrameIndex
                unmatched.removeAll { $0 == matchedIdx }
            }
        }

        // New detections become fresh tracks.
        for i in unmatched {
            let raw = raws[i]
            tracked.append(TrackedDetection(
                yoloClass: raw.yoloClass,
                confidence: raw.confidence,
                bbox: raw.bbox,
                lastSeenFrame: currentFrameIndex
            ))
        }

        // Drop stale tracks.
        tracked.removeAll { (currentFrameIndex - $0.lastSeenFrame) > AppConfig.boxMaxMissedFrames }
        return tracked
    }

    func reset() {
        tracked.removeAll()
        currentFrameIndex = 0
    }

    private static func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        if inter.isNull { return 0 }
        let interArea = Float(inter.width * inter.height)
        let unionArea = Float(a.width * a.height + b.width * b.height) - interArea
        if unionArea <= 0 { return 0 }
        return interArea / unionArea
    }
}
