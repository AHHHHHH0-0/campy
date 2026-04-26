import AVFoundation
import SwiftUI

/// Draws one bounding box per tracked detection over the live camera preview.
/// Box color is driven by `DetectionState`, with neutral gray for live YOLO-only
/// detections. The box itself is the tap target.
struct OverlayCanvas: View {
    let detections: [TrackedDetection]
    let states: [UUID: DetectionState]
    let frameSize: CGSize
    let previewLayer: AVCaptureVideoPreviewLayer
    let namespace: Namespace.ID
    let onTap: (TrackedDetection, DetectionState) -> Void

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                ForEach(visibleDetections) { det in
                    let state = states[det.id] ?? .blank
                    let rect = previewRect(for: det.bbox, in: geo.size)
                    BoundingBox(
                        color: color(for: state),
                        label: state.label
                    )
                    .frame(width: rect.width, height: rect.height)
                    .contentShape(Rectangle())
                    .position(x: rect.midX, y: rect.midY)
                    .matchedGeometryEffect(id: det.id, in: namespace, isSource: true)
                    .onTapGesture {
                        onTap(det, state)
                    }
                    .animation(UIConfig.Box.positionSpring, value: det.bbox)
                }
            }
        }
    }

    private var visibleDetections: [TrackedDetection] {
        detections.filter { det in
            switch states[det.id] {
            case .blank, .none: return false
            default: return true
            }
        }
    }

    /// Translates a YOLO-frame pixel rect into preview-layer points. We
    /// normalize against the frame's pixel size first, then ask
    /// `AVCaptureVideoPreviewLayer.layerRectConverted(fromMetadataOutputRect:)`
    /// to apply the correct gravity/orientation transform.
    private func previewRect(for bbox: CGRect, in viewSize: CGSize) -> CGRect {
        guard frameSize.width > 0, frameSize.height > 0 else { return .zero }
        let normalized = CGRect(
            x: bbox.minX / frameSize.width,
            y: bbox.minY / frameSize.height,
            width: bbox.width / frameSize.width,
            height: bbox.height / frameSize.height
        )
        return previewLayer.layerRectConverted(fromMetadataOutputRect: normalized)
    }

    private func color(for state: DetectionState) -> Color {
        switch state {
        case .edible: return UIConfig.leafGreen
        case .poisonous: return UIConfig.alertRed
        case .notFood: return UIConfig.neutralGray
        case .inedible, .notFound: return UIConfig.sage
        case .blank: return .clear
        }
    }
}

private struct BoundingBox: View {
    let color: Color
    let label: String

    var body: some View {
        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: UIConfig.Box.cornerRadius)
                .stroke(color, lineWidth: UIConfig.Box.strokeWidth)

            if !label.isEmpty {
                Text(label)
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(UIConfig.paper)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Capsule().fill(color))
                    .offset(x: 6, y: 0)
            }
        }
    }
}
