import CoreImage
import CoreVideo
import Foundation
import Vision

actor VisionPlantGateService: VisionPlantGateServiceProtocol {
    private let ciContext = CIContext()
    private var didLogSupportedIdentifiers = false
    private let plantKeywords = [
        "plant", "flower", "leaf", "tree", "herb", "vegetable",
        "fruit", "flora", "foliage", "shrub", "grass", "moss", "fern"
    ]
    private let plantThreshold: Float = 0.35
    private let nonPlantThreshold: Float = 0.12

    func evaluateTap(detection: TrackedDetection, frame: CameraFrame) async -> VisionPlantGateOutcome? {
        do {
            if !didLogSupportedIdentifiers {
                let supported = try Self.supportedIdentifiers()
                print("[VisionPlantGate] supported_identifiers_count=\(supported.count)")
                print("[VisionPlantGate] supported_identifiers_sample=\(supported.prefix(120).joined(separator: ", "))")
                didLogSupportedIdentifiers = true
            }

            guard let crop = makeCropImage(from: frame.pixelBuffer, bbox: detection.bbox) else {
                print("[VisionPlantGate] crop_failed yolo_class=\(detection.yoloClass)")
                return nil
            }

            let request = VNClassifyImageRequest()
            let handler = VNImageRequestHandler(cgImage: crop, options: [:])
            try handler.perform([request])

            let observations = (request.results as? [VNClassificationObservation]) ?? []
            let top = observations.prefix(8).map {
                VisionPlantCandidate(identifier: $0.identifier.lowercased(), confidence: $0.confidence)
            }
            let report = VisionPlantAuditReport(yoloClass: detection.yoloClass, candidates: top)
            let rendered = top
                .map { "\($0.identifier)=\(String(format: "%.3f", $0.confidence))" }
                .joined(separator: ", ")
            let bestPlantScore = top
                .filter { isPlantKeywordMatch($0.identifier) }
                .map(\.confidence)
                .max() ?? 0
            let decision: VisionPlantDecision
            if bestPlantScore >= plantThreshold {
                decision = .plantLike
            } else if bestPlantScore <= nonPlantThreshold {
                decision = .nonPlant
            } else {
                decision = .unsure
            }
            print("[VisionPlantGate] tap_audit yolo_class=\(detection.yoloClass) top=\(rendered)")
            print("[VisionPlantGate] decision=\(decision.rawValue) best_plant_score=\(String(format: "%.3f", bestPlantScore))")
            return VisionPlantGateOutcome(decision: decision, report: report)
        } catch {
            print("[VisionPlantGate] audit_error yolo_class=\(detection.yoloClass) error=\(error.localizedDescription)")
            return nil
        }
    }

    private func isPlantKeywordMatch(_ identifier: String) -> Bool {
        for keyword in plantKeywords where identifier.contains(keyword) {
            return true
        }
        return false
    }

    private static func supportedIdentifiers() throws -> [String] {
        let request = VNClassifyImageRequest()
        return try request.supportedIdentifiers().map { $0.lowercased() }.sorted()
    }

    private func makeCropImage(from pixelBuffer: CVPixelBuffer, bbox: CGRect) -> CGImage? {
        let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        guard width > 1, height > 1 else { return nil }

        let clamped = CGRect(
            x: max(0, min(bbox.minX, width - 1)),
            y: max(0, min(bbox.minY, height - 1)),
            width: max(1, min(bbox.width, width)),
            height: max(1, min(bbox.height, height))
        )
        guard clamped.width >= 1, clamped.height >= 1 else { return nil }

        let ciY = height - clamped.maxY
        let ciRect = CGRect(x: clamped.minX, y: ciY, width: clamped.width, height: clamped.height).integral
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer).cropped(to: ciRect)
        return ciContext.createCGImage(ciImage, from: ciImage.extent)
    }
}

actor StubVisionPlantGateService: VisionPlantGateServiceProtocol {
    func evaluateTap(detection: TrackedDetection, frame: CameraFrame) async -> VisionPlantGateOutcome? { nil }
}

