import SwiftUI

/// Audio-reactive `Circle` driven by a smoothed RMS level (0…1).
/// `scaleEffect` interpolates between `baselineScale` and `peakScale` so the
/// circle "pulses" with mic input.
struct AudioVisualizerCircle: View {
    let level: Double

    var body: some View {
        let clamped = max(0, min(1, level))
        let scale = UIConfig.AudioVisualizer.baselineScale
            + (UIConfig.AudioVisualizer.peakScale - UIConfig.AudioVisualizer.baselineScale) * clamped

        Circle()
            .fill(UIConfig.leafGreen)
            .opacity(0.85)
            .frame(width: UIConfig.AudioVisualizer.baselineSize, height: UIConfig.AudioVisualizer.baselineSize)
            .scaleEffect(scale)
            .shadow(color: UIConfig.leafGreen.opacity(0.4), radius: 24)
            .animation(UIConfig.AudioVisualizer.levelAnimation, value: clamped)
    }
}

#Preview {
    AudioVisualizerCircle(level: 0.6)
}
