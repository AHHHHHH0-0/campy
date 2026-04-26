import CoreVideo
import Foundation

/// Converts pixel buffers (or sub-regions thereof) into Zetic-ready tensor bytes.
/// Returns raw bytes laid out as `[1, 3, height, width]` float32 (NCHW), already
/// normalized by the supplied `ModelInputSpec`. Concrete implementations build a
/// `Tensor` from these bytes inside the inference services.
protocol TensorFactoryProtocol: Sendable {
    func makeNormalizedNCHWBytes(
        from pixelBuffer: CVPixelBuffer,
        crop: CGRect?,
        spec: ModelInputSpec
    ) -> Data
}
