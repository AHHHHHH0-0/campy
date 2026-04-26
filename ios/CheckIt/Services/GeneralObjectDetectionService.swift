import CoreVideo
import Foundation
import os

#if canImport(ZeticMLange)
import ZeticMLange
#endif

/// Wraps the Zetic-hosted YOLO11 model (`Steve/YOLO11_comparison` v5).
/// Drops frames if a previous inference is still in flight (no queueing).
actor GeneralObjectDetectionService: ObjectDetectionServiceProtocol {

    private let tensorFactory: any TensorFactoryProtocol
    private let modelLoader: ModelLoader
    private var inFlight: Bool = false

    /// COCO class names exposed by the YOLO11 model. Pulled lazily because we
    /// don't want to ship the whole map on disk if the runtime later supplies it.
    private let cocoClasses: [String] = COCO80Classes

    init(tensorFactory: any TensorFactoryProtocol, modelLoader: ModelLoader) {
        self.tensorFactory = tensorFactory
        self.modelLoader = modelLoader
    }

    func detect(in frame: CameraFrame) async -> [RawDetection]? {
        guard !inFlight else { return nil }
        inFlight = true
        defer { inFlight = false }

#if canImport(ZeticMLange)
        guard let model = await modelLoader.yolo else { return nil }

        let bytes = tensorFactory.makeNormalizedNCHWBytes(
            from: frame.pixelBuffer,
            crop: nil,
            spec: ModelInputSpec.yolo
        )
        let inputTensor = Tensor(
            data: bytes,
            dataType: BuiltinDataType.float32,
            shape: [1, 3, ModelInputSpec.yolo.height, ModelInputSpec.yolo.width]
        )
        do {
            let outputs = try model.run(inputs: [inputTensor])
            let frameSize = CGSize(
                width: CVPixelBufferGetWidth(frame.pixelBuffer),
                height: CVPixelBufferGetHeight(frame.pixelBuffer)
            )
            return Self.parseYOLOOutputs(
                outputs,
                cocoClasses: cocoClasses,
                modelInputSize: CGSize(width: ModelInputSpec.yolo.width, height: ModelInputSpec.yolo.height),
                frameSize: frameSize
            )
        } catch {
            return nil
        }
#else
        return []
#endif
    }

#if canImport(ZeticMLange)
    /// YOLO11 emits a flat tensor of shape `[1, num_classes + 4, num_anchors]`
    /// (4 box coords + per-class confidence). We linearly scan the predictions,
    /// run a confidence threshold, and undo the letterbox transform.
    private static func parseYOLOOutputs(
        _ outputs: [Tensor],
        cocoClasses: [String],
        modelInputSize: CGSize,
        frameSize: CGSize
    ) -> [RawDetection] {
        guard let raw = outputs.first else { return [] }
        let shape = raw.shape
        guard shape.count == 3 else { return [] }
        let attrCount = shape[1]
        let anchorCount = shape[2]
        let classCount = attrCount - 4
        guard classCount == cocoClasses.count else { return [] }

        let floats: [Float] = raw.data.withUnsafeBytes { buf in
            let typed = buf.bindMemory(to: Float.self)
            return Array(typed)
        }
        guard floats.count == attrCount * anchorCount else { return [] }

        // Index helpers — flat layout is [attr][anchor].
        func at(_ attr: Int, _ anchor: Int) -> Float {
            floats[attr * anchorCount + anchor]
        }

        // Letterbox parameters used in `CheckItTensorFactory.makeNormalizedNCHWBytes`.
        let scale = min(modelInputSize.width / frameSize.width, modelInputSize.height / frameSize.height)
        let scaledW = frameSize.width * scale
        let scaledH = frameSize.height * scale
        let dx = (modelInputSize.width - scaledW) / 2
        let dy = (modelInputSize.height - scaledH) / 2

        var detections: [RawDetection] = []
        detections.reserveCapacity(anchorCount / 8)

        for anchor in 0..<anchorCount {
            // Pick best class.
            var bestClass = -1
            var bestScore: Float = 0
            for c in 0..<classCount {
                let s = at(4 + c, anchor)
                if s > bestScore {
                    bestScore = s
                    bestClass = c
                }
            }
            if bestScore < AppConfig.detectionConfidenceThreshold || bestClass < 0 { continue }

            let cx = CGFloat(at(0, anchor))
            let cy = CGFloat(at(1, anchor))
            let w = CGFloat(at(2, anchor))
            let h = CGFloat(at(3, anchor))

            // Undo letterbox: subtract padding, divide by scale.
            let imageX = (cx - dx) / scale
            let imageY = (cy - dy) / scale
            let imageW = w / scale
            let imageH = h / scale

            let bbox = CGRect(
                x: imageX - imageW / 2,
                y: imageY - imageH / 2,
                width: imageW,
                height: imageH
            ).intersection(CGRect(origin: .zero, size: frameSize))
            if bbox.isNull || bbox.width < 1 || bbox.height < 1 { continue }

            detections.append(RawDetection(
                yoloClass: cocoClasses[bestClass],
                confidence: bestScore,
                bbox: bbox
            ))
        }

        return Self.applyNMS(detections, iouThreshold: AppConfig.boxIoUMatchThreshold)
    }

    private static func applyNMS(_ dets: [RawDetection], iouThreshold: Float) -> [RawDetection] {
        let sorted = dets.sorted { $0.confidence > $1.confidence }
        var keep: [RawDetection] = []
        for candidate in sorted {
            if keep.count >= AppConfig.maxDetectionsPerFrame { break }
            var rejected = false
            for already in keep {
                if already.yoloClass == candidate.yoloClass &&
                    iou(already.bbox, candidate.bbox) > iouThreshold {
                    rejected = true
                    break
                }
            }
            if !rejected { keep.append(candidate) }
        }
        return keep
    }

    private static func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        if inter.isNull { return 0 }
        let interArea = Float(inter.width * inter.height)
        let unionArea = Float(a.width * a.height + b.width * b.height) - interArea
        if unionArea <= 0 { return 0 }
        return interArea / unionArea
    }
#endif
}

/// Standard COCO 80 class list emitted by `Steve/YOLO11_comparison`.
let COCO80Classes: [String] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

// MARK: Stub used for previews / when Zetic is unavailable

actor StubObjectDetectionService: ObjectDetectionServiceProtocol {
    func detect(in frame: CameraFrame) async -> [RawDetection]? { [] }
}
