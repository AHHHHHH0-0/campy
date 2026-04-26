import AVFoundation
import CoreVideo
import Foundation

/// Frame produced by the camera capture pipeline; consumed by the inference worker.
/// Reference type so we can share the underlying `CVPixelBuffer` without a deep copy.
final class CameraFrame: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let timestamp: CMTime
    let frameIndex: Int

    init(pixelBuffer: CVPixelBuffer, timestamp: CMTime, frameIndex: Int) {
        self.pixelBuffer = pixelBuffer
        self.timestamp = timestamp
        self.frameIndex = frameIndex
    }
}

protocol CameraServiceProtocol: AnyObject, Sendable {
    /// Async sequence of pixel buffers produced by the active capture session.
    /// Frames are emitted on a background queue.
    var frames: AsyncStream<CameraFrame> { get }

    /// Layer view used by `CameraPreviewView`. Owned by the camera service.
    @MainActor
    var previewLayer: AVCaptureVideoPreviewLayer { get }

    /// Whether the AVCaptureSession is currently running.
    @MainActor
    var isRunning: Bool { get }

    /// Idempotently brings the session up.
    /// `AVCaptureSession.startRunning()` must be called from a background queue
    /// (Apple Developer Documentation: AVCaptureSession), so callers should treat
    /// this as fire-and-forget.
    @MainActor
    func start() async

    /// Idempotently tears the session down.
    @MainActor
    func stop() async

    /// Pauses outputs without tearing down the AVCaptureSession (used while the
    /// detail sheet is open).
    @MainActor
    func pause()

    /// Resumes outputs after `pause()`.
    @MainActor
    func resume()
}
