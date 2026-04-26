import AVFoundation
import Foundation
import os

/// Raw-PCM mic capture via `AVAudioEngine`. Apple's `Speech` framework is NOT
/// involved — transcription is done by `WhisperTranscriberService` after `stop()`.
///
/// Apple Developer Documentation: `AVAudioEngine` taps cannot have an explicit
/// format applied other than `nil` (use the bus's native format), so we resample
/// in the Whisper pipeline rather than at capture.
final class SpeechInputService: SpeechInputServiceProtocol, @unchecked Sendable {

    let levelStream: AsyncStream<Double>
    let cancelStream: AsyncStream<Void>

    private let levelContinuation: AsyncStream<Double>.Continuation
    private let cancelContinuation: AsyncStream<Void>.Continuation
    private let engine = AVAudioEngine()
    private let bufferLock = OSAllocatedUnfairLock<[Float]>(initialState: [])
    private let levelLock = OSAllocatedUnfairLock<Double>(initialState: 0)
    private var sourceFormat: AVAudioFormat?
    private var interruptionObserver: NSObjectProtocol?

    init() {
        var levelContinuation: AsyncStream<Double>.Continuation!
        let levelStream = AsyncStream<Double> { c in levelContinuation = c }
        var cancelContinuation: AsyncStream<Void>.Continuation!
        let cancelStream = AsyncStream<Void> { c in cancelContinuation = c }
        self.levelStream = levelStream
        self.levelContinuation = levelContinuation
        self.cancelStream = cancelStream
        self.cancelContinuation = cancelContinuation

        observeAudioInterruption()
    }

    deinit {
        if let interruptionObserver {
            NotificationCenter.default.removeObserver(interruptionObserver)
        }
    }

    @MainActor
    func start() async throws {
        bufferLock.withLock { $0.removeAll(keepingCapacity: true) }
        levelLock.withLock { $0 = 0 }

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement, options: [])
        try session.setActive(true, options: [])

        let input = engine.inputNode
        let format = input.inputFormat(forBus: 0)
        sourceFormat = format

        // Apple Developer Documentation: pass `nil` for the tap format to use
        // the bus's native format and avoid resample in the engine.
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.processBuffer(buffer)
        }

        engine.prepare()
        try engine.start()
    }

    @MainActor
    func stop() async -> CapturedAudio? {
        let format = sourceFormat
        teardownEngine()
        guard let format else { return nil }
        let pcm = bufferLock.withLock { $0 }
        bufferLock.withLock { $0.removeAll(keepingCapacity: false) }
        guard !pcm.isEmpty else { return nil }
        return CapturedAudio(
            pcm: pcm,
            sampleRate: format.sampleRate,
            channelCount: format.channelCount
        )
    }

    @MainActor
    func cancel() {
        teardownEngine()
        bufferLock.withLock { $0.removeAll(keepingCapacity: false) }
        cancelContinuation.yield(())
    }

    // MARK: Internals

    private func processBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channels = buffer.floatChannelData else { return }
        let frameCount = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)
        guard frameCount > 0 else { return }

        // Mix down to mono and append.
        var monoMutable = [Float](repeating: 0, count: frameCount)
        for ch in 0..<channelCount {
            let ptr = channels[ch]
            for i in 0..<frameCount {
                monoMutable[i] += ptr[i] / Float(channelCount)
            }
        }
        let mono = monoMutable
        bufferLock.withLock { $0.append(contentsOf: mono) }

        // Smoothed RMS for visualizer.
        var sumSq: Float = 0
        for v in mono { sumSq += v * v }
        let rms = sqrtf(sumSq / Float(frameCount))
        let normalized = min(max(Double(rms) * 4.0, 0), 1)
        let smoothed = levelLock.withLock { current -> Double in
            let alpha = UIConfig.AudioVisualizer.levelSmoothing
            current = current * (1 - alpha) + normalized * alpha
            return current
        }
        levelContinuation.yield(smoothed)
    }

    private func teardownEngine() {
        if engine.isRunning {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        sourceFormat = nil
    }

    private func observeAudioInterruption() {
        interruptionObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: nil,
            queue: .main
        ) { [weak self] note in
            guard let self,
                  let info = note.userInfo,
                  let typeRaw = info[AVAudioSessionInterruptionTypeKey] as? UInt,
                  let type = AVAudioSession.InterruptionType(rawValue: typeRaw) else {
                return
            }
            if type == .began {
                Task { @MainActor in self.cancel() }
            }
        }
    }
}

final class StubSpeechInputService: SpeechInputServiceProtocol, @unchecked Sendable {
    let levelStream: AsyncStream<Double>
    let cancelStream: AsyncStream<Void>

    init() {
        self.levelStream = AsyncStream { _ in }
        self.cancelStream = AsyncStream { _ in }
    }

    @MainActor func start() async throws {}
    @MainActor func stop() async -> CapturedAudio? { nil }
    @MainActor func cancel() {}
}
