import AVFoundation
import CoreVideo
import Foundation
import os

/// Production camera pipeline. Rear lens, 1920x1080, 30 fps, BGRA pixel format.
///
/// Apple Developer Documentation: `AVCaptureSession.startRunning()` and
/// `stopRunning()` block the calling thread and must be invoked from a
/// background queue. We honor that by hopping every session mutation onto a
/// dedicated `sessionQueue`.
final class CameraService: NSObject, CameraServiceProtocol, @unchecked Sendable {

    // MARK: Public surface

    let frames: AsyncStream<CameraFrame>
    let previewLayer: AVCaptureVideoPreviewLayer

    @MainActor
    var isRunning: Bool { session.isRunning }

    // MARK: Private

    private let session: AVCaptureSession
    private let videoOutput: AVCaptureVideoDataOutput
    private let sessionQueue = DispatchQueue(label: "ai.checkit.camera.session", qos: .userInitiated)
    private let captureQueue = DispatchQueue(label: "ai.checkit.camera.capture", qos: .userInitiated)
    private let frameContinuation: AsyncStream<CameraFrame>.Continuation
    private let pausedLock = OSAllocatedUnfairLock<Bool>(initialState: false)
    private let frameIndexLock = OSAllocatedUnfairLock<Int>(initialState: 0)
    private let configuredLock = OSAllocatedUnfairLock<Bool>(initialState: false)

    override init() {
        let session = AVCaptureSession()
        let output = AVCaptureVideoDataOutput()
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill

        self.session = session
        self.videoOutput = output
        self.previewLayer = layer

        var capturedContinuation: AsyncStream<CameraFrame>.Continuation!
        // Keep only the single newest frame in the buffer. YOLO inference takes
        // ~150–300 ms; with unbounded buffering the consumer processes a growing
        // backlog of stale frames. bufferingNewest(1) ensures the worker always
        // gets the most recent frame the moment it becomes free.
        let stream = AsyncStream<CameraFrame>(bufferingPolicy: .bufferingNewest(1)) { c in
            capturedContinuation = c
        }
        self.frames = stream
        self.frameContinuation = capturedContinuation

        super.init()

        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: captureQueue)
    }

    @MainActor
    func start() async {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.configureIfNeeded()
            if !self.session.isRunning {
                self.session.startRunning()
            }
        }
    }

    @MainActor
    func stop() async {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if self.session.isRunning {
                self.session.stopRunning()
            }
        }
    }

    @MainActor
    func pause() {
        pausedLock.withLock { $0 = true }
    }

    @MainActor
    func resume() {
        pausedLock.withLock { $0 = false }
    }

    // MARK: Configuration

    private func configureIfNeeded() {
        let alreadyConfigured = configuredLock.withLock { configured -> Bool in
            if configured { return true }
            configured = true
            return false
        }
        if alreadyConfigured { return }

        session.beginConfiguration()
        // 720p is more than enough for YOLO's 640×640 input and cuts pixel-data
        // preprocessing time roughly in half versus the previous 1080p preset.
        session.sessionPreset = .hd1280x720

        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
           let input = try? AVCaptureDeviceInput(device: device),
           session.canAddInput(input) {
            session.addInput(input)
            do {
                try device.lockForConfiguration()
                device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
                device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
                device.unlockForConfiguration()
            } catch {
                // Non-fatal — we'll fall back to the device's default frame range.
            }
        }

        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }

        if let connection = videoOutput.connection(with: .video) {
            connection.videoRotationAngle = 90
        }
        session.commitConfiguration()
    }

    private func nextFrameIndex() -> Int {
        frameIndexLock.withLock { idx in
            idx += 1
            return idx
        }
    }

    private var isPaused: Bool {
        pausedLock.withLock { $0 }
    }
}

// MARK: AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraService: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard !isPaused,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let frame = CameraFrame(
            pixelBuffer: pixelBuffer,
            timestamp: timestamp,
            frameIndex: nextFrameIndex()
        )
        frameContinuation.yield(frame)
    }
}

// MARK: Stub used only for SwiftUI previews / unit tests

final class StubCameraService: CameraServiceProtocol, @unchecked Sendable {
    let frames: AsyncStream<CameraFrame>

    @MainActor
    let previewLayer: AVCaptureVideoPreviewLayer = AVCaptureVideoPreviewLayer()

    @MainActor
    var isRunning: Bool = false

    init() {
        self.frames = AsyncStream { _ in }
    }

    @MainActor func start() async { isRunning = true }
    @MainActor func stop() async { isRunning = false }
    @MainActor func pause() {}
    @MainActor func resume() {}
}
