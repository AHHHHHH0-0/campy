import CoreVideo
import Foundation
import os

#if canImport(ZeticMLange)
import ZeticMLange
#endif

actor PlantClassificationService: PlantClassificationServiceProtocol {

    private let tensorFactory: any TensorFactoryProtocol
    private let modelLoader: ModelLoader
    private var inFlight: Bool = false
    /// classIndex (Int) -> scientific name (String), built once from the two bundled JSONs.
    private lazy var classIndexToName: [Int: String] = Self.loadClassIndexMap()
    private let logger = Logger(subsystem: "CheckIt", category: "PlantClassifier")

    init(tensorFactory: any TensorFactoryProtocol, modelLoader: ModelLoader) {
        self.tensorFactory = tensorFactory
        self.modelLoader = modelLoader
    }

    func classify(crop: CGRect, in frame: CameraFrame) async -> PlantPrediction? {
        guard !inFlight else { return nil }
        inFlight = true
        defer { inFlight = false }

#if canImport(ZeticMLange)
        guard let model = await modelLoader.plantClassifier else {
            #if DEBUG
            logger.debug("plantClassifier model not loaded")
            #endif
            return nil
        }

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
            return Self.topOne(
                logits: logits,
                classIndexToName: classIndexToName,
                threshold: AppConfig.plantClassificationAcceptanceThreshold,
                logger: logger
            )
        } catch {
            #if DEBUG
            logger.debug("classify run error: \(String(describing: error), privacy: .public)")
            #endif
            return nil
        }
#else
        return nil
#endif
    }

#if canImport(ZeticMLange)
    private static func topOne(
        logits: Tensor,
        classIndexToName: [Int: String],
        threshold: Float,
        logger: Logger
    ) -> PlantPrediction? {
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

        #if DEBUG
        logger.debug(
            "topOne bestIdx=\(bestIdx, privacy: .public) bestProb=\(bestProb, privacy: .public) mapSize=\(classIndexToName.count, privacy: .public)"
        )
        #endif

        if bestProb < threshold {
            #if DEBUG
            logger.debug("topOne below threshold (\(threshold, privacy: .public)), returning nil")
            #endif
            return nil
        }

        let name: String
        if let resolved = classIndexToName[bestIdx] {
            name = resolved
            #if DEBUG
            logger.debug("topOne resolved idx=\(bestIdx, privacy: .public) -> \(name, privacy: .public)")
            #endif
        } else {
            name = "unknown_\(bestIdx)"
            #if DEBUG
            logger.debug("topOne no mapping for idx=\(bestIdx, privacy: .public), falling back to \(name, privacy: .public)")
            #endif
        }

        return PlantPrediction(scientificName: name, confidence: bestProb)
    }
#endif

    /// Builds a flat [classIndex: scientificName] map from two bundled JSONs:
    ///   class_idx_to_species_id.json  — {"0": "1355868", "1": "1355920", ...}
    ///   plantnet300K_species_id_2_name.json — {"1355868": "Lactuca virosa L.", ...}
    private static func loadClassIndexMap() -> [Int: String] {
        guard
            let idxURL = Bundle.main.url(
                forResource: ModelConfig.PlantClassifier.classIndexToSpeciesIdResource,
                withExtension: "json"
            ),
            let nameURL = Bundle.main.url(
                forResource: ModelConfig.PlantClassifier.speciesIdToNameResource,
                withExtension: "json"
            ),
            let idxData = try? Data(contentsOf: idxURL),
            let nameData = try? Data(contentsOf: nameURL),
            let idxRaw = try? JSONSerialization.jsonObject(with: idxData) as? [String: String],
            let nameRaw = try? JSONSerialization.jsonObject(with: nameData) as? [String: String]
        else {
            let log = Logger(subsystem: "CheckIt", category: "PlantClassifier")
            log.debug("loadClassIndexMap: failed to load one or both JSON label files")
            return [:]
        }

        var result: [Int: String] = [:]
        result.reserveCapacity(idxRaw.count)
        for (idxStr, speciesId) in idxRaw {
            guard let idx = Int(idxStr), let name = nameRaw[speciesId] else { continue }
            result[idx] = name
        }

        let log = Logger(subsystem: "CheckIt", category: "PlantClassifier")
        log.debug("loadClassIndexMap: loaded \(result.count, privacy: .public) class->name mappings")
        return result
    }
}

actor StubPlantClassificationService: PlantClassificationServiceProtocol {
    func classify(crop: CGRect, in frame: CameraFrame) async -> PlantPrediction? { nil }
}
