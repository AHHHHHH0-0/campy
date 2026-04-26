import Foundation
import Observation

/// Single composition root. Owns and wires every service.
/// Built once at app start in `CheckItApp`, propagated via
/// `@Environment(\.appContainer)`. Exposes protocols, not concrete types.
@MainActor
@Observable
final class AppContainer {

    // MARK: Services (protocol-typed)

    let camera: any CameraServiceProtocol
    let tensorFactory: any TensorFactoryProtocol
    let detection: any ObjectDetectionServiceProtocol
    let plantClassifier: any PlantClassificationServiceProtocol
    let visionPlantGate: any VisionPlantGateServiceProtocol
    let speechInput: any SpeechInputServiceProtocol
    let whisper: any WhisperTranscriberServiceProtocol
    let gemini: any GeminiServiceProtocol
    let gemma: any GemmaReasoningServiceProtocol
    let packStore: any PackStoreProtocol
    let plantKnowledge: OfflinePlantKnowledgeService
    let transcriptStore: any TranscriptStoreProtocol
    let telemetry: InferenceTelemetry
    let modelLoader: ModelLoader

    // MARK: Reactive state

    /// 0…1 weighted-average download progress aggregated across all five Zetic models.
    /// `ModelLoadingView` binds directly to this.
    var modelLoadProgress: Double {
        modelLoader.aggregatedProgress
    }

    var allModelsReady: Bool {
        modelLoader.allReady
    }

    // MARK: Init

    init(
        camera: any CameraServiceProtocol,
        tensorFactory: any TensorFactoryProtocol,
        detection: any ObjectDetectionServiceProtocol,
        plantClassifier: any PlantClassificationServiceProtocol,
        visionPlantGate: any VisionPlantGateServiceProtocol,
        speechInput: any SpeechInputServiceProtocol,
        whisper: any WhisperTranscriberServiceProtocol,
        gemini: any GeminiServiceProtocol,
        gemma: any GemmaReasoningServiceProtocol,
        packStore: any PackStoreProtocol,
        plantKnowledge: OfflinePlantKnowledgeService,
        transcriptStore: any TranscriptStoreProtocol,
        telemetry: InferenceTelemetry,
        modelLoader: ModelLoader
    ) {
        self.camera = camera
        self.tensorFactory = tensorFactory
        self.detection = detection
        self.plantClassifier = plantClassifier
        self.visionPlantGate = visionPlantGate
        self.speechInput = speechInput
        self.whisper = whisper
        self.gemini = gemini
        self.gemma = gemma
        self.packStore = packStore
        self.plantKnowledge = plantKnowledge
        self.transcriptStore = transcriptStore
        self.telemetry = telemetry
        self.modelLoader = modelLoader
    }

    // MARK: Default production wiring

    /// Builds the live-app composition root. All services are concrete production types.
    /// Heavy model construction is deferred to `modelLoader.startEagerLoad()` which
    /// runs after splash and permission grants.
    static func makeProduction() -> AppContainer {
        let telemetry = InferenceTelemetry()
        let modelLoader = ModelLoader()

        let tensorFactory = CheckItTensorFactory()
        let camera = CameraService()
        let detection = GeneralObjectDetectionService(tensorFactory: tensorFactory, modelLoader: modelLoader)
        let plantClassifier = PlantClassificationService(tensorFactory: tensorFactory, modelLoader: modelLoader)
        let visionPlantGate = VisionPlantGateService()
        let speechInput = SpeechInputService()
        let whisper = WhisperTranscriberService(modelLoader: modelLoader)
        let packStore = PackStore()
        let plantKnowledge = OfflinePlantKnowledgeService(packStore: packStore)
        let transcriptStore = TranscriptStore()
        let gemini = GeminiPackService(packStore: packStore, transcriptStore: transcriptStore)
        let gemma = GemmaReasoningService(modelLoader: modelLoader)

        return AppContainer(
            camera: camera,
            tensorFactory: tensorFactory,
            detection: detection,
            plantClassifier: plantClassifier,
            visionPlantGate: visionPlantGate,
            speechInput: speechInput,
            whisper: whisper,
            gemini: gemini,
            gemma: gemma,
            packStore: packStore,
            plantKnowledge: plantKnowledge,
            transcriptStore: transcriptStore,
            telemetry: telemetry,
            modelLoader: modelLoader
        )
    }

    /// Lightweight container for SwiftUI previews.
    /// Uses no-op stubs so previews never hit the network or AVFoundation.
    static var preview: AppContainer {
        let telemetry = InferenceTelemetry()
        let modelLoader = ModelLoader.preview
        let tensorFactory = CheckItTensorFactory()
        let camera = StubCameraService()
        let detection = StubObjectDetectionService()
        let plantClassifier = StubPlantClassificationService()
        let visionPlantGate = StubVisionPlantGateService()
        let speechInput = StubSpeechInputService()
        let whisper = StubWhisperTranscriberService()
        let packStore = PackStore()
        let plantKnowledge = OfflinePlantKnowledgeService(packStore: packStore)
        let transcriptStore = TranscriptStore()
        let gemini = StubGeminiService()
        let gemma = GemmaReasoningService(modelLoader: modelLoader)
        return AppContainer(
            camera: camera,
            tensorFactory: tensorFactory,
            detection: detection,
            plantClassifier: plantClassifier,
            visionPlantGate: visionPlantGate,
            speechInput: speechInput,
            whisper: whisper,
            gemini: gemini,
            gemma: gemma,
            packStore: packStore,
            plantKnowledge: plantKnowledge,
            transcriptStore: transcriptStore,
            telemetry: telemetry,
            modelLoader: modelLoader
        )
    }
}
