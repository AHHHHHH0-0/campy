import Foundation
import Observation

#if canImport(ZeticMLange)
import ZeticMLange
#endif

/// Owns the five Zetic-hosted models (YOLO11, plant classifier, Whisper encoder,
/// Whisper decoder, Gemma) and their aggregated download progress. Eager-loaded
/// on the splash; the resulting handles are pulled lazily by the inference
/// services that depend on them.
///
/// We surface progress as a single `Double` 0…1 weighted-average; `ModelConfig.LoadProgressWeight`
/// is the source of truth for the weights.
@MainActor
@Observable
final class ModelLoader {

    // MARK: Per-model progress sliders, each 0…1

    private(set) var yoloProgress: Double = 0
    private(set) var plantProgress: Double = 0
    private(set) var whisperEncoderProgress: Double = 0
    private(set) var whisperDecoderProgress: Double = 0
    private(set) var gemmaProgress: Double = 0

    private(set) var allReady: Bool = false
    private(set) var startError: Error? = nil

#if canImport(ZeticMLange)
    private(set) var yolo: ZeticMLangeModel?
    private(set) var plantClassifier: ZeticMLangeModel?
    private(set) var whisperEncoder: ZeticMLangeModel?
    private(set) var whisperDecoder: ZeticMLangeModel?
    private(set) var gemma: ZeticMLangeLLMModel?
#endif

    var aggregatedProgress: Double {
        let w = ModelConfig.LoadProgressWeight.self
        let weighted =
            yoloProgress * w.yolo +
            plantProgress * w.plant +
            whisperEncoderProgress * w.whisperEncoder +
            whisperDecoderProgress * w.whisperDecoder +
            gemmaProgress * w.gemma
        return min(max(weighted / w.sum, 0), 1)
    }

    /// Used only by SwiftUI previews to fast-forward the splash.
    static var preview: ModelLoader {
        let m = ModelLoader()
        m.yoloProgress = 1
        m.plantProgress = 1
        m.whisperEncoderProgress = 1
        m.whisperDecoderProgress = 1
        m.gemmaProgress = 1
        m.allReady = true
        return m
    }

    /// Kicks off all five downloads concurrently. Safe to call once after both
    /// camera and microphone permissions have been granted.
    func startEagerLoad() {
        guard !allReady else { return }
#if canImport(ZeticMLange)
        Task.detached(priority: .userInitiated) { [weak self] in
            await self?.loadAll()
        }
#else
        // Simulator path: no ZeticMLange binary; pretend everything loaded.
        yoloProgress = 1
        plantProgress = 1
        whisperEncoderProgress = 1
        whisperDecoderProgress = 1
        gemmaProgress = 1
        allReady = true
#endif
    }

#if canImport(ZeticMLange)
    private nonisolated func loadAll() async {
        let key = SecretsLoader.zeticPersonalKey

        // Five concurrent loads. We update progress on the main actor.
        await withTaskGroup(of: Void.self) { group in
            group.addTask { [weak self] in
                await self?.loadYOLO(personalKey: key)
            }
            group.addTask { [weak self] in
                await self?.loadPlant(personalKey: key)
            }
            group.addTask { [weak self] in
                await self?.loadWhisperEncoder(personalKey: key)
            }
            group.addTask { [weak self] in
                await self?.loadWhisperDecoder(personalKey: key)
            }
            group.addTask { [weak self] in
                await self?.loadGemma(personalKey: key)
            }
        }

        await MainActor.run {
            self.allReady =
                self.yolo != nil &&
                self.plantClassifier != nil &&
                self.whisperEncoder != nil &&
                self.whisperDecoder != nil &&
                self.gemma != nil
        }
    }

    private nonisolated func loadYOLO(personalKey: String) async {
        do {
            let model = try ZeticMLangeModel(
                personalKey: personalKey,
                name: ModelConfig.YOLO.name,
                version: ModelConfig.YOLO.version,
                modelMode: .RUN_SPEED,
                onDownload: { [weak self] progress in
                    Task { @MainActor in self?.yoloProgress = Double(progress) }
                }
            )
            await MainActor.run {
                self.yolo = model
                self.yoloProgress = 1
            }
        } catch {
            await MainActor.run { self.startError = error }
        }
    }

    private nonisolated func loadPlant(personalKey: String) async {
        do {
            let model = try ZeticMLangeModel(
                personalKey: personalKey,
                name: ModelConfig.PlantClassifier.name,
                version: ModelConfig.PlantClassifier.version,
                modelMode: .RUN_AUTO,
                onDownload: { [weak self] progress in
                    Task { @MainActor in self?.plantProgress = Double(progress) }
                }
            )
            await MainActor.run {
                self.plantClassifier = model
                self.plantProgress = 1
            }
        } catch {
            await MainActor.run { self.startError = error }
        }
    }

    private nonisolated func loadWhisperEncoder(personalKey: String) async {
        do {
            let model = try ZeticMLangeModel(
                personalKey: personalKey,
                name: ModelConfig.WhisperEncoder.name,
                version: ModelConfig.WhisperEncoder.version,
                modelMode: .RUN_SPEED,
                onDownload: { [weak self] progress in
                    Task { @MainActor in self?.whisperEncoderProgress = Double(progress) }
                }
            )
            await MainActor.run {
                self.whisperEncoder = model
                self.whisperEncoderProgress = 1
            }
        } catch {
            await MainActor.run { self.startError = error }
        }
    }

    private nonisolated func loadWhisperDecoder(personalKey: String) async {
        do {
            let model = try ZeticMLangeModel(
                personalKey: personalKey,
                name: ModelConfig.WhisperDecoder.name,
                version: ModelConfig.WhisperDecoder.version,
                modelMode: .RUN_SPEED,
                onDownload: { [weak self] progress in
                    Task { @MainActor in self?.whisperDecoderProgress = Double(progress) }
                }
            )
            await MainActor.run {
                self.whisperDecoder = model
                self.whisperDecoderProgress = 1
            }
        } catch {
            await MainActor.run { self.startError = error }
        }
    }

    private nonisolated func loadGemma(personalKey: String) async {
        do {
            let model = try ZeticMLangeLLMModel(
                personalKey: personalKey,
                name: ModelConfig.Gemma.name,
                version: ModelConfig.Gemma.version,
                onDownload: { [weak self] progress in
                    Task { @MainActor in self?.gemmaProgress = Double(progress) }
                }
            )
            await MainActor.run {
                self.gemma = model
                self.gemmaProgress = 1
            }
        } catch {
            await MainActor.run { self.startError = error }
        }
    }
#endif
}
