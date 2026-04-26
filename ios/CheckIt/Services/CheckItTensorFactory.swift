import Accelerate
import CoreImage
import CoreVideo
import Foundation

/// Letterbox-resize a `CVPixelBuffer` (BGRA) into an `[1, 3, H, W]` float32 NCHW
/// tensor and apply per-channel mean/std normalization in-place via vDSP.
///
/// We use `CIImage` + `CIContext.render(...)` for the resize because it gives us
/// hardware-accelerated scaling without a Metal device dance, and the resulting
/// pixel data lives in a plain `CVPixelBuffer` we can read back as bytes.
final class CheckItTensorFactory: TensorFactoryProtocol, @unchecked Sendable {

    private let ciContext: CIContext

    init() {
        // Disable color management so we don't unintentionally re-tone-map the input.
        self.ciContext = CIContext(options: [.workingColorSpace: NSNull()])
    }

    func makeNormalizedNCHWBytes(
        from pixelBuffer: CVPixelBuffer,
        crop: CGRect?,
        spec: ModelInputSpec
    ) -> Data {
        let baseImage = CIImage(cvPixelBuffer: pixelBuffer)
        let cropped = crop.flatMap { Self.cropImage(baseImage, to: $0, sourceSize: pixelBuffer.size) } ?? baseImage

        let targetW = spec.width
        let targetH = spec.height

        // Letterbox resize preserving aspect ratio.
        let inputSize = cropped.extent.size
        let scale = min(CGFloat(targetW) / inputSize.width, CGFloat(targetH) / inputSize.height)
        let scaledW = inputSize.width * scale
        let scaledH = inputSize.height * scale
        let dx = (CGFloat(targetW) - scaledW) / 2
        let dy = (CGFloat(targetH) - scaledH) / 2

        let scaled = cropped
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .transformed(by: CGAffineTransform(translationX: dx, y: dy))

        // Render onto a black canvas at the target size.
        var rgbaBytes = [UInt8](repeating: 0, count: targetW * targetH * 4)
        rgbaBytes.withUnsafeMutableBytes { rawBuf in
            ciContext.render(
                scaled,
                toBitmap: rawBuf.baseAddress!,
                rowBytes: targetW * 4,
                bounds: CGRect(x: 0, y: 0, width: targetW, height: targetH),
                format: .RGBA8,
                colorSpace: nil
            )
        }

        return Self.normalizeNCHW(
            rgba: rgbaBytes,
            width: targetW,
            height: targetH,
            mean: spec.mean,
            std: spec.std
        )
    }

    private static func cropImage(_ image: CIImage, to crop: CGRect, sourceSize: CGSize) -> CIImage {
        // CIImage origin is bottom-left; convert from top-left UI rect.
        let flipped = CGRect(
            x: crop.origin.x,
            y: sourceSize.height - crop.origin.y - crop.height,
            width: crop.width,
            height: crop.height
        )
        let bounded = flipped.intersection(image.extent)
        return image.cropped(to: bounded)
            .transformed(by: CGAffineTransform(translationX: -bounded.origin.x, y: -bounded.origin.y))
    }

    private static func normalizeNCHW(
        rgba: [UInt8],
        width: Int,
        height: Int,
        mean: [Float],
        std: [Float]
    ) -> Data {
        let pixelCount = width * height
        var planes = [[Float](repeating: 0, count: pixelCount),
                      [Float](repeating: 0, count: pixelCount),
                      [Float](repeating: 0, count: pixelCount)]

        for i in 0..<pixelCount {
            let r = Float(rgba[i * 4 + 0]) / 255.0
            let g = Float(rgba[i * 4 + 1]) / 255.0
            let b = Float(rgba[i * 4 + 2]) / 255.0
            planes[0][i] = (r - mean[0]) / std[0]
            planes[1][i] = (g - mean[1]) / std[1]
            planes[2][i] = (b - mean[2]) / std[2]
        }

        var data = Data(count: pixelCount * 3 * MemoryLayout<Float>.size)
        data.withUnsafeMutableBytes { raw in
            let dst = raw.bindMemory(to: Float.self)
            for c in 0..<3 {
                planes[c].withUnsafeBufferPointer { src in
                    let dstStart = dst.baseAddress!.advanced(by: c * pixelCount)
                    dstStart.update(from: src.baseAddress!, count: pixelCount)
                }
            }
        }
        return data
    }
}

private extension CVPixelBuffer {
    var size: CGSize {
        CGSize(width: CVPixelBufferGetWidth(self), height: CVPixelBufferGetHeight(self))
    }
}
