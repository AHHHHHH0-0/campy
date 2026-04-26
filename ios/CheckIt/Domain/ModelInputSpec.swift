import Foundation

/// Preprocessing record passed from `ModelConfig` into `TensorFactoryProtocol`.
struct ModelInputSpec: Equatable, Sendable {
    let width: Int
    let height: Int
    let mean: [Float]
    let std: [Float]

    static let yolo = ModelInputSpec(
        width: ModelConfig.YOLO.inputSize.width,
        height: ModelConfig.YOLO.inputSize.height,
        mean: ModelConfig.YOLO.mean,
        std: ModelConfig.YOLO.std
    )

    static let plant = ModelInputSpec(
        width: ModelConfig.PlantClassifier.inputSize.width,
        height: ModelConfig.PlantClassifier.inputSize.height,
        mean: ModelConfig.PlantClassifier.mean,
        std: ModelConfig.PlantClassifier.std
    )
}
