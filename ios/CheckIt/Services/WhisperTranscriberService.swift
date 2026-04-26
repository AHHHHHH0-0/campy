@preconcurrency import AVFoundation
import Foundation
import os

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
    private let logger = Logger(subsystem: "CheckIt", category: "Whisper")

    init(modelLoader: ModelLoader) {
        self.modelLoader = modelLoader
    }

    func transcribe(audio: CapturedAudio) async -> String {
        let startedAt = Date()
#if canImport(ZeticMLange)
        let enc = await modelLoader.whisperEncoder
        let dec = await modelLoader.whisperDecoder
        let tok = tokenizer
        guard let encoder = enc, let decoder = dec, let tokenizer = tok else {
            #if DEBUG
            logger.debug("missing whisper prerequisites encoder=\(enc != nil, privacy: .public) decoder=\(dec != nil, privacy: .public) tokenizer=\(tok != nil, privacy: .public)")
            #endif
            return ""
        }

        guard let resampled = Self.resampleTo16k(audio: audio) else {
            #if DEBUG
            logger.debug("failed to resample audio to 16k")
            #endif
            return ""
        }
        #if DEBUG
        logger.debug(
            "transcribe start inputPcmCount=\(audio.pcm.count, privacy: .public) inputSampleRate=\(audio.sampleRate, privacy: .public) resampledCount=\(resampled.count, privacy: .public)"
        )
        #endif
        let mel = WhisperLogMel.compute(pcm: resampled)
        let melTensor = Self.tensorFromFloats(
            mel,
            shape: [1, AppConfig.whisperMelBins, WhisperLogMel.melFrameCount]
        )

        do {
            let encoderOutputs = try encoder.run(inputs: [melTensor])
            guard !encoderOutputs.isEmpty else { return "" }
            #if DEBUG
            logger.debug("encoder outputs count=\(encoderOutputs.count, privacy: .public)")
            #endif

            let encoded = encoderOutputs[0]
            let padLen = AppConfig.whisperStaticDecodeLen

            var generated: [Int] = tokenizer.startPrefixIDs()
            let endID = tokenizer.endOfTextTokenID ?? -1
            let suppress = tokenizer.timestampTokenIDs

            while generated.count < AppConfig.whisperMaxDecodeTokens {
                // Zetic's decoder is compiled with static shapes. Inputs must be:
                //   [0] input_ids          — [1, 448] int32, padded with 0
                //   [1] encoder_hidden_states — [1, 1500, d_model] float32
                //   [2] decoder_attention_mask — [1, 448] int32, 1 for real tokens, 0 for pad
                var paddedIds = generated.map { Int32($0) }
                paddedIds.append(contentsOf: [Int32](repeating: 0, count: max(0, padLen - paddedIds.count)))

                var mask = [Int32](repeating: 0, count: padLen)
                for i in 0..<min(generated.count, padLen) { mask[i] = 1 }

                let tokenTensor = Self.tensorFromInt32(paddedIds)
                let maskTensor  = Self.tensorFromInt32(mask)
                let logits = try decoder.run(inputs: [tokenTensor, encoded, maskTensor])
                guard let outputTensor = logits.first else { break }

                // Predict next token from logits at the last real position.
                let nextID = Self.greedyArgmax(logits: outputTensor, atPosition: generated.count - 1, suppress: suppress)
                if nextID == endID || nextID < 0 { break }
                generated.append(nextID)
            }

            // Strip the prefix tokens before decoding.
            let prefixCount = tokenizer.startPrefixIDs().count
            let textTokens = Array(generated.dropFirst(prefixCount))
            let decoded = tokenizer.decode(textTokens).trimmingCharacters(in: .whitespacesAndNewlines)
            #if DEBUG
            let latencyMs = Date().timeIntervalSince(startedAt) * 1000
            logger.debug(
                "transcribe success latencyMs=\(latencyMs, privacy: .public) tokenCount=\(textTokens.count, privacy: .public) textLength=\(decoded.count, privacy: .public) text=\(decoded, privacy: .public)"
            )
            #endif
            return decoded
        } catch {
            #if DEBUG
            logger.debug("transcribe failed with error: \(String(describing: error), privacy: .public)")
            #endif
            return ""
        }
#else
        #if DEBUG
        logger.debug("canImport(ZeticMLange) is false; returning empty transcript")
        #endif
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

    /// Greedy argmax over the logits at a specific sequence position.
    /// The Zetic decoder emits `[1, seq_len, vocab]`; we read position `atPosition`
    /// (the last real token), which contains the next-token prediction.
    private static func greedyArgmax(logits: Tensor, atPosition position: Int, suppress: Set<Int>) -> Int {
        let count = logits.count()
        guard count > 0 else { return -1 }
        let vocab = logits.shape.last ?? 0
        guard vocab > 0 else { return -1 }
        let floats: [Float] = logits.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self).prefix(count)) }
        let seqLen = count / vocab
        let safePos = min(max(position, 0), seqLen - 1)
        let start = safePos * vocab
        var bestIdx = -1
        var bestVal: Float = -.greatestFiniteMagnitude
        for i in 0..<vocab {
            if suppress.contains(i) { continue }
            let v = floats[start + i]
            if v > bestVal {
                bestVal = v
                bestIdx = i
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
