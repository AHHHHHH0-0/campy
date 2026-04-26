import Foundation

/// Concrete identifiers and preprocessing constants for every Zetic-hosted model.
enum ModelConfig {
    // YOLO11 detector
    enum YOLO {
        static let name = "Steve/YOLO11_comparison"
        static let version = 5
        static let inputSize: (width: Int, height: Int) = (640, 640)
        static let mean: [Float] = [0.0, 0.0, 0.0]
        static let std: [Float] = [1.0, 1.0, 1.0]
        static let plantClassAllowlist: Set<String> = ["potted plant"]
    }

    // Plant species classifier
    enum PlantClassifier {
        static let name = "NumaanFormoli/plant300k"
        static let version = 2
        static let inputSize: (width: Int, height: Int) = (224, 224)
        static let mean: [Float] = [0.485, 0.456, 0.406]
        static let std: [Float] = [0.229, 0.224, 0.225]
        static let topKLabelsResource = "plant300k_labels"
    }

    // Whisper encoder
    enum WhisperEncoder {
        static let name = "OpenAI/whisper-base-encoder"
        static let version = 1
    }

    // Whisper decoder
    enum WhisperDecoder {
        static let name = "OpenAI/whisper-tiny-decoder"
        static let version = 1
    }

    // Bundled tokenizer assets shipped under Resources/Whisper/ (sourced from the
    // official OpenAI Whisper repo; multilingual `whisper-base` / `whisper-tiny`
    // share the same tokenizer).
    enum WhisperTokenizer {
        static let vocabResource = "whisper-vocab"
        static let mergesResource = "whisper-merges"
        static let startOfTranscriptToken = "<|startoftranscript|>"
        static let endOfTextToken = "<|endoftext|>"
        static let timestampTokenPrefix = "<|"
        static let englishTaskPrefixTokens = [
            "<|startoftranscript|>",
            "<|en|>",
            "<|transcribe|>",
            "<|notimestamps|>"
        ]
    }

    // Gemma narrator LLM
    enum Gemma {
        static let name = "changgeun/gemma-4-E2B-it"
        static let version = 1
    }

    /// Weighting used to roll up per-model download progress into a single 0…1 value
    /// for `ModelLoadingView`. Roughly proportional to model size — Gemma dominates.
    enum LoadProgressWeight {
        static let yolo: Double = 0.05
        static let plant: Double = 0.05
        static let whisperEncoder: Double = 0.10
        static let whisperDecoder: Double = 0.10
        static let gemma: Double = 0.70

        static var sum: Double { yolo + plant + whisperEncoder + whisperDecoder + gemma }
    }
}
