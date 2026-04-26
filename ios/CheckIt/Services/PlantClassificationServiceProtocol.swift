import CoreVideo
import Foundation

struct PlantPrediction: Equatable, Sendable {
    let scientificName: String
    let confidence: Float
}

protocol PlantClassificationServiceProtocol: AnyObject, Sendable {
    /// Runs the offline plant classifier against a YOLO-detected plant crop.
    /// Returns top-1 scientific name + confidence, or `nil` if the model isn't loaded
    /// or another inference is in flight.
    func classify(crop: CGRect, in frame: CameraFrame) async -> PlantPrediction?
}
