@preconcurrency import AVFoundation
import Foundation

#if canImport(ZeticMLange)
import ZeticMLange
#endif

/// Two-stage Whisper pipeline:
/// 1. Resample to 16 kHz mono float32 via `AVAudioConverter`.
/// 2. Zero-pad to exactly 30 s.
/// 3. Build an 80-bin log-mel spectrogram via `WhisperLogMel`.
/// 4. Run encoder once, decoder autoregressively (greedy argmax).
/// 5. Detokenize via the bundled BPE tokenizer.
///
/// All internal failures are caught and surfaced as an empty string so the chat
/// system prompt routes to its empty-input canned reply.
actor WhisperTranscriberService: WhisperTranscriberServiceProtocol {

    private let modelLoader: ModelLoader
    private lazy var tokenizer: WhisperBPETokenizer? = WhisperBPETokenizer.loadFromBundle()

    init(modelLoader: ModelLoader) {
        self.modelLoader = modelLoader
    }

    func transcribe(audio: CapturedAudio) async -> String {
#if canImport(ZeticMLange)
        guard let encoder = await modelLoader.whisperEncoder,
              let decoder = await modelLoader.whisperDecoder,
              let tokenizer = tokenizer else {
            return ""
        }

        guard let resampled = Self.resampleTo16k(audio: audio) else { return "" }
        let mel = WhisperLogMel.compute(pcm: resampled)
        let melTensor = Self.tensorFromFloats(
            mel,
            shape: [1, AppConfig.whisperMelBins, WhisperLogMel.melFrameCount]
        )

        do {
            let encoderOutputs = try encoder.run(inputs: [melTensor])
            guard let encoded = encoderOutputs.first else { return "" }

            var generated: [Int] = tokenizer.startPrefixIDs()
            let endID = tokenizer.endOfTextTokenID ?? -1
            let suppress = tokenizer.timestampTokenIDs

            while generated.count < AppConfig.whisperMaxDecodeTokens {
                let tokenTensor = Self.tensorFromInt32(generated.map { Int32($0) })
                let logits = try decoder.run(inputs: [encoded, tokenTensor])
                guard let lastLogits = logits.first else { break }

                let nextID = Self.greedyArgmax(logits: lastLogits, suppress: suppress)
                if nextID == endID || nextID < 0 { break }
                generated.append(nextID)
            }

            // Strip the prefix tokens before decoding.
            let prefixCount = tokenizer.startPrefixIDs().count
            let textTokens = Array(generated.dropFirst(prefixCount))
            return tokenizer.decode(textTokens).trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            return ""
        }
#else
        return ""
#endif
    }

#if canImport(ZeticMLange)
    private static func tensorFromFloats(_ floats: [Float], shape: [Int]) -> Tensor {
        let data = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        return Tensor(data: data, dataType: BuiltinDataType.float32, shape: shape)
    }

    private static func tensorFromInt32(_ ints: [Int32]) -> Tensor {
        let data = ints.withUnsafeBufferPointer { Data(buffer: $0) }
        return Tensor(data: data, dataType: BuiltinDataType.int32, shape: [1, ints.count])
    }

    /// Greedy argmax over the final time-step's logits. The decoder output is
    /// shaped `[batch, time, vocab]`; we take the logits at the last time step.
    private static func greedyArgmax(logits: Tensor, suppress: Set<Int>) -> Int {
        let count = logits.count()
        guard count > 0 else { return -1 }
        let floats: [Float] = logits.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self).prefix(count)) }
        // Time axis is the next-to-last; vocab is last. Without per-model shape
        // metadata we conservatively assume the decoder emits a single timestep
        // per call (Zetic's static-shape decoder export pattern), so the entire
        // output is one logits row.
        let vocab = logits.shape.last ?? floats.count
        let start = max(0, floats.count - vocab)
        var bestIdx = -1
        var bestVal: Float = -.greatestFiniteMagnitude
        for i in 0..<vocab {
            let id = i
            if suppress.contains(id) { continue }
            let v = floats[start + i]
            if v > bestVal {
                bestVal = v
                bestIdx = id
            }
        }
        return bestIdx
    }
#endif

    /// Resample arbitrary device audio to 16 kHz mono float32.
    /// Apple Developer Documentation: `AVAudioConverter` accepts variable-rate
    /// sources; we feed it via the input-block API to avoid manual chunking.
    private static func resampleTo16k(audio: CapturedAudio) -> [Float]? {
        let targetRate = AppConfig.whisperSampleRate
        if abs(audio.sampleRate - targetRate) < 1.0 {
            return audio.pcm
        }

        guard let sourceFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: audio.sampleRate,
            channels: audio.channelCount,
            interleaved: false
        ),
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetRate,
            channels: 1,
            interleaved: false
        ),
        let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
            return nil
        }
        converter.sampleRateConverterQuality = .max

        let sourceFrameCount = AVAudioFrameCount(audio.pcm.count)
        guard let sourceBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: sourceFrameCount) else {
            return nil
        }
        sourceBuffer.frameLength = sourceFrameCount
        if let dst = sourceBuffer.floatChannelData?[0] {
            audio.pcm.withUnsafeBufferPointer { src in
                dst.update(from: src.baseAddress!, count: audio.pcm.count)
            }
        }

        let ratio = targetRate / audio.sampleRate
        let estimatedTargetCount = AVAudioFrameCount(Double(sourceFrameCount) * ratio + 16)
        guard let targetBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: estimatedTargetCount) else {
            return nil
        }

        var error: NSError?
        var providedOnce = false
        let status = converter.convert(to: targetBuffer, error: &error) { _, outStatus in
            if providedOnce {
                outStatus.pointee = .endOfStream
                return nil
            }
            providedOnce = true
            outStatus.pointee = .haveData
            return sourceBuffer
        }
        guard status != .error, error == nil,
              let outChannel = targetBuffer.floatChannelData?[0] else {
            return nil
        }
        let outCount = Int(targetBuffer.frameLength)
        return Array(UnsafeBufferPointer(start: outChannel, count: outCount))
    }
}

actor StubWhisperTranscriberService: WhisperTranscriberServiceProtocol {
    func transcribe(audio: CapturedAudio) async -> String { "" }
}
