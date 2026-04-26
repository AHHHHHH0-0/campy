import Foundation

struct VisionPlantCandidate: Equatable, Sendable {
    let identifier: String
    let confidence: Float
}

struct VisionPlantAuditReport: Equatable, Sendable {
    let yoloClass: String
    let candidates: [VisionPlantCandidate]
}

enum VisionPlantDecision: String, Equatable, Sendable {
    case plantLike
    case nonPlant
    case unsure
}

struct VisionPlantGateOutcome: Equatable, Sendable {
    let decision: VisionPlantDecision
    let report: VisionPlantAuditReport
}

protocol VisionPlantGateServiceProtocol: AnyObject, Sendable {
    /// Runs tap-time Vision classification on the detection crop, logs candidate
    /// outputs, and returns a deterministic plant-likeliness decision.
    func evaluateTap(detection: TrackedDetection, frame: CameraFrame) async -> VisionPlantGateOutcome?
}

