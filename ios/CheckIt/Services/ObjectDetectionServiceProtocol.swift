import CoreVideo
import Foundation

protocol ObjectDetectionServiceProtocol: AnyObject, Sendable {
    /// Runs YOLO11 against `frame.pixelBuffer`. Returns the per-frame raw detections
    /// in image coordinates (top-left origin, pixel units). If a previous inference
    /// is still in flight the frame is dropped and `nil` is returned.
    func detect(in frame: CameraFrame) async -> [RawDetection]?
}
