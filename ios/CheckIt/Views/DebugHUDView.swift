import SwiftUI

/// Local-only debug overlay. Toggleable via 3-finger tap on `LiveView`.
/// No external sinks; reads from `InferenceTelemetry` directly.
struct DebugHUDView: View {
    @Environment(\.appContainer) private var container

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            row("fps", value: String(format: "%.1f", container.telemetry.fps))
            row("infer", value: String(format: "%.0f ms", container.telemetry.lastInferenceLatencyMs))
            row("yolo", value: container.telemetry.lastDetectionClass)
            row("sciname", value: container.telemetry.lastScientificName)
            row("lookup", value: container.telemetry.lastLookupStatus)
            row("chat", value: String(format: "%.0f ms", container.telemetry.lastChatLatencyMs))
            row("pack", value: String(format: "%.0f ms", container.telemetry.lastPackLatencyMs))
            row("whisper", value: String(format: "%.0f ms", container.telemetry.lastWhisperLatencyMs))
            row("gemma cache", value: "\(container.telemetry.gemmaCacheSize)")
        }
        .font(.system(size: 11, weight: .regular, design: .monospaced))
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
        .foregroundStyle(UIConfig.inkGreen)
        .padding(8)
    }

    @ViewBuilder
    private func row(_ key: String, value: String) -> some View {
        HStack(spacing: 6) {
            Text(key)
                .foregroundStyle(UIConfig.inkGreen.opacity(0.55))
                .frame(width: 86, alignment: .leading)
            Text(value)
        }
    }
}
