import AVFoundation
import SwiftUI
import UIKit

/// `UIViewRepresentable` over `AVCaptureVideoPreviewLayer`.
/// Apple Developer Documentation: AVCaptureVideoPreviewLayer — the layer
/// is owned by `CameraService`, we just embed it inside a host view.
struct CameraPreviewView: UIViewRepresentable {
    let previewLayer: AVCaptureVideoPreviewLayer

    func makeUIView(context: Context) -> PreviewHostView {
        let view = PreviewHostView()
        view.previewLayer = previewLayer
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        return view
    }

    func updateUIView(_ uiView: PreviewHostView, context: Context) {
        // Frame is laid out by `PreviewHostView.layoutSubviews()`.
    }

    final class PreviewHostView: UIView {
        var previewLayer: AVCaptureVideoPreviewLayer?

        override func layoutSubviews() {
            super.layoutSubviews()
            previewLayer?.frame = bounds
        }
    }
}
