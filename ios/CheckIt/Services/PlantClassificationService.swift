import CoreVideo
import Foundation

#if canImport(ZeticMLange)
import ZeticMLange
#endif

actor PlantClassificationService: PlantClassificationServiceProtocol {

    private let tensorFactory: any TensorFactoryProtocol
    private let modelLoader: ModelLoader
    private var inFlight: Bool = false
    private lazy var labels: [String] = Self.loadLabels()

    init(tensorFactory: any TensorFactoryProtocol, modelLoader: ModelLoader) {
        self.tensorFactory = tensorFactory
        self.modelLoader = modelLoader
    }

    func classify(crop: CGRect, in frame: CameraFrame) async -> PlantPrediction? {
        guard !inFlight else { return nil }
        inFlight = true
        defer { inFlight = false }

#if canImport(ZeticMLange)
        guard let model = await modelLoader.plantClassifier else { return nil }

        let bytes = tensorFactory.makeNormalizedNCHWBytes(
            from: frame.pixelBuffer,
            crop: crop,
            spec: ModelInputSpec.plant
        )
        let inputTensor = Tensor(
            data: bytes,
            dataType: BuiltinDataType.float32,
            shape: [1, 3, ModelInputSpec.plant.height, ModelInputSpec.plant.width]
        )
        do {
            let outputs = try model.run(inputs: [inputTensor])
            guard let logits = outputs.first else { return nil }
            return Self.topOne(logits: logits, labels: labels, threshold: AppConfig.plantClassificationAcceptanceThreshold)
        } catch {
            return nil
        }
#else
        return nil
#endif
    }

#if canImport(ZeticMLange)
    private static func topOne(logits: Tensor, labels: [String], threshold: Float) -> PlantPrediction? {
        let count = logits.count()
        guard count > 0 else { return nil }
        let floats: [Float] = logits.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self).prefix(count)) }
        // Apply softmax for normalized confidence.
        let maxLogit = floats.max() ?? 0
        var sum: Float = 0
        var exps = [Float](repeating: 0, count: floats.count)
        for i in 0..<floats.count {
            let e = expf(floats[i] - maxLogit)
            exps[i] = e
            sum += e
        }
        guard sum > 0 else { return nil }

        var bestIdx = 0
        var bestProb: Float = 0
        for i in 0..<exps.count {
            let p = exps[i] / sum
            if p > bestProb {
                bestProb = p
                bestIdx = i
            }
        }
        if bestProb < threshold { return nil }
        let name = bestIdx < labels.count ? labels[bestIdx] : "unknown_\(bestIdx)"
        return PlantPrediction(scientificName: name, confidence: bestProb)
    }
#endif

    private static func loadLabels() -> [String] {
        guard let url = Bundle.main.url(forResource: ModelConfig.PlantClassifier.topKLabelsResource, withExtension: "txt"),
              let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8) else {
            return []
        }
        return text.split(whereSeparator: \.isNewline).map { String($0).trimmingCharacters(in: .whitespaces) }
    }
}

actor StubPlantClassificationService: PlantClassificationServiceProtocol {
    func classify(crop: CGRect, in frame: CameraFrame) async -> PlantPrediction? { nil }
}
