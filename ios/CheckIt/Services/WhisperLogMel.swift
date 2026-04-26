import Accelerate
import Foundation

/// Log-mel spectrogram preprocessing for Whisper. STFT magnitude → 80-bin mel
/// filterbank → log → clamp to [-1, +0.4] (matching OpenAI's reference).
///
/// vDSP gives us hardware-accelerated FFT and matrix-vector multiplication.
/// The mel filterbank itself is recomputed once on first use.
enum WhisperLogMel {

    static let sampleRate: Float = 16_000
    static let nFFT: Int = AppConfig.whisperFFTSize         // 400
    static let hop: Int = AppConfig.whisperHopSize          // 160
    static let nMels: Int = AppConfig.whisperMelBins        // 80
    static let windowSeconds: Int = AppConfig.whisperWindowSeconds // 30

    /// Number of mel frames in the fixed Whisper window (3000 for a 30 s window).
    static var melFrameCount: Int {
        (Int(sampleRate) * windowSeconds) / hop
    }

    /// Returns a flat `[nMels * melFrameCount]` float32 buffer in row-major order
    /// (mel index outer, time index inner) suitable for direct upload as a
    /// `[1, nMels, melFrameCount]` Whisper encoder input tensor.
    static func compute(pcm: [Float]) -> [Float] {
        let totalSamples = Int(sampleRate) * windowSeconds
        var padded = pcm
        if padded.count < totalSamples {
            padded.append(contentsOf: [Float](repeating: 0, count: totalSamples - padded.count))
        } else if padded.count > totalSamples {
            padded = Array(padded.prefix(totalSamples))
        }

        let frameCount = melFrameCount
        let log2Size = vDSP_Length(log2(Double(nFFT)))
        guard let fft = vDSP_create_fftsetup(log2Size, FFTRadix(kFFTRadix2)) else {
            return [Float](repeating: 0, count: nMels * frameCount)
        }
        defer { vDSP_destroy_fftsetup(fft) }

        let window = hannWindow(length: nFFT)

        let bins = nFFT / 2
        var realPart = [Float](repeating: 0, count: bins)
        var imagPart = [Float](repeating: 0, count: bins)
        var magBuffer = [Float](repeating: 0, count: bins)
        var output = [Float](repeating: 0, count: nMels * frameCount)
        let filters = melFilterbank()

        for frame in 0..<frameCount {
            let start = frame * hop
            let end = start + nFFT
            var windowed = [Float](repeating: 0, count: nFFT)
            if start < padded.count {
                let available = min(nFFT, padded.count - start)
                for i in 0..<available {
                    windowed[i] = padded[start + i] * window[i]
                }
            }
            // Pack into split-complex buffer for vDSP.
            windowed.withUnsafeBufferPointer { ptr in
                ptr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: bins) { complex in
                    realPart.withUnsafeMutableBufferPointer { rPtr in
                        imagPart.withUnsafeMutableBufferPointer { iPtr in
                            var splitComplex = DSPSplitComplex(realp: rPtr.baseAddress!, imagp: iPtr.baseAddress!)
                            vDSP_ctoz(complex, 2, &splitComplex, 1, vDSP_Length(bins))
                        }
                    }
                }
            }
            realPart.withUnsafeMutableBufferPointer { rPtr in
                imagPart.withUnsafeMutableBufferPointer { iPtr in
                    var splitComplex = DSPSplitComplex(realp: rPtr.baseAddress!, imagp: iPtr.baseAddress!)
                    vDSP_fft_zrip(fft, &splitComplex, 1, log2Size, FFTDirection(FFT_FORWARD))
                    vDSP_zvmags(&splitComplex, 1, &magBuffer, 1, vDSP_Length(bins))
                }
            }
            // vDSP scales by 4N for magsq; divide by `(2*nFFT)^2` to get power, but for
            // mel filterbank purposes any constant scale is absorbed by the log.
            for m in 0..<nMels {
                var s: Float = 0
                let row = filters[m]
                for b in 0..<bins {
                    s += row[b] * magBuffer[b]
                }
                output[m * frameCount + frame] = max(s, 1e-10)
            }
        }
        // Log-mel and dynamic range clamp matching the OpenAI Whisper reference.
        for i in 0..<output.count {
            output[i] = log10(output[i])
        }
        let maxVal = output.max() ?? 0
        let floor = maxVal - 8.0
        for i in 0..<output.count {
            output[i] = max(output[i], floor)
            output[i] = (output[i] + 4.0) / 4.0
        }
        return output
    }

    private static func hannWindow(length: Int) -> [Float] {
        var w = [Float](repeating: 0, count: length)
        for i in 0..<length {
            w[i] = 0.5 * (1 - cos(2 * .pi * Float(i) / Float(length - 1)))
        }
        return w
    }

    private static func melFilterbank() -> [[Float]] {
        // Slaney-style mel filterbank (matches `librosa.filters.mel(htk=False)`).
        let fmin: Float = 0
        let fmax: Float = sampleRate / 2
        let nBins = nFFT / 2
        let melLow = hzToMel(fmin)
        let melHigh = hzToMel(fmax)
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0..<melPoints.count {
            melPoints[i] = melLow + (melHigh - melLow) * Float(i) / Float(nMels + 1)
        }
        let hzPoints = melPoints.map { melToHz($0) }
        let binFreqs = (0..<nBins).map { Float($0) * sampleRate / Float(nFFT) }
        var filters = [[Float]](repeating: [Float](repeating: 0, count: nBins), count: nMels)
        for m in 0..<nMels {
            let lower = hzPoints[m]
            let center = hzPoints[m + 1]
            let upper = hzPoints[m + 2]
            for b in 0..<nBins {
                let f = binFreqs[b]
                if f >= lower, f <= center {
                    filters[m][b] = (f - lower) / max(center - lower, 1e-6)
                } else if f > center, f <= upper {
                    filters[m][b] = (upper - f) / max(upper - center, 1e-6)
                }
            }
            // Slaney normalization.
            let weight = 2 / max(upper - lower, 1e-6)
            for b in 0..<nBins { filters[m][b] *= weight }
        }
        return filters
    }

    private static func hzToMel(_ hz: Float) -> Float {
        let f0: Float = 0
        let f_sp: Float = 200.0 / 3
        let logStep: Float = logf(6.4) / 27
        let minLogHz: Float = 1000
        if hz < minLogHz {
            return (hz - f0) / f_sp
        }
        let minLogMel = (minLogHz - f0) / f_sp
        return minLogMel + logf(hz / minLogHz) / logStep
    }

    private static func melToHz(_ mel: Float) -> Float {
        let f0: Float = 0
        let f_sp: Float = 200.0 / 3
        let logStep: Float = logf(6.4) / 27
        let minLogHz: Float = 1000
        let minLogMel = (minLogHz - f0) / f_sp
        if mel < minLogMel {
            return f0 + f_sp * mel
        }
        return minLogHz * expf(logStep * (mel - minLogMel))
    }
}
