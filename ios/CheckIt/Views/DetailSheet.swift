import SwiftUI
import UIKit

/// Full-screen card that animates in via a custom bloom from the tapped box's
/// frame (paired with the `OverlayCanvas` `matchedGeometryEffect`). Streams a
/// Gemma narrative paragraph for every detection: in-pack plants, not-found
/// plants, and general YOLO objects (where Gemma decides food vs. not food).
struct DetailSheet: View {
    let detection: TrackedDetection
    let state: DetectionState
    let frame: CameraFrame?
    let namespace: Namespace.ID
    let onDismiss: () -> Void

    @Environment(\.appContainer) private var container
    @State private var paragraph: String = ""
    @State private var isGenerating: Bool = false
    @State private var streamTask: Task<Void, Never>? = nil
    @State private var resolvedState: DetectionState

    init(
        detection: TrackedDetection,
        state: DetectionState,
        frame: CameraFrame?,
        namespace: Namespace.ID,
        onDismiss: @escaping () -> Void
    ) {
        self.detection = detection
        self.state = state
        self.frame = frame
        self.namespace = namespace
        self.onDismiss = onDismiss
        _resolvedState = State(initialValue: state)
    }

    var body: some View {
        ZStack {
            detailBackgroundColor
                .ignoresSafeArea()
                .matchedGeometryEffect(id: detection.id, in: namespace, isSource: false)

            VStack(alignment: .leading, spacing: UIConfig.Spacing.lg) {
                header

                ScrollView {
                    VStack(alignment: .leading, spacing: UIConfig.Spacing.md) {
                        Text(displayText)
                            .font(.system(size: 16, weight: .regular, design: .serif))
                            .foregroundStyle(UIConfig.inkGreen)
                            .multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .opacity(paragraph.isEmpty && !isGenerating ? 0.4 : 1)

                        if let rationale = rationale {
                            Text(rationale)
                                .font(.system(size: 14, weight: .regular))
                                .foregroundStyle(rationaleColor)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding(.horizontal, UIConfig.Spacing.lg)
                }
                .frame(maxHeight: .infinity, alignment: .top)

                dismissHint
            }
            .padding(.top, windowSafeAreaTop + UIConfig.Spacing.lg)
            .padding(.bottom, UIConfig.Spacing.lg)
        }
        .contentShape(Rectangle())
        .onTapGesture {
            streamTask?.cancel()
            onDismiss()
        }
        .task(id: detection.id) {
            await startStreaming()
        }
        .onDisappear {
            streamTask?.cancel()
        }
    }

    @ViewBuilder
    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(titleLine)
                .font(.system(size: 26, weight: .bold, design: .rounded))
                .foregroundStyle(UIConfig.inkGreen)
            if let subtitle = subtitleLine {
                Text(subtitle)
                    .font(.system(size: 14, weight: .medium, design: .rounded))
                    .foregroundStyle(UIConfig.inkGreen.opacity(0.6))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, UIConfig.Spacing.lg)
    }

    @ViewBuilder
    private var dismissHint: some View {
        Text(UIStrings.dismissHint)
            .font(.system(size: 12, weight: .regular))
            .foregroundStyle(UIConfig.inkGreen.opacity(0.45))
            .frame(maxWidth: .infinity)
    }

    private var titleLine: String {
        switch resolvedState {
        case .edible(let e), .inedible(let e), .poisonous(let e):
            return e.commonName
        case .notFound(let name):
            return name
        case .notFood(let cls):
            return cls
        case .blank:
            return "—"
        }
    }

    private var subtitleLine: String? {
        switch resolvedState {
        case .edible(let e), .inedible(let e), .poisonous(let e):
            return e.scientificName
        case .notFound:
            return UIStrings.notInLocalDatabase
        case .notFood, .blank:
            return nil
        }
    }

    private var rationale: String? {
        switch resolvedState {
        case .edible(let e), .inedible(let e), .poisonous(let e):
            return e.rationale
        case .notFood, .notFound, .blank:
            return nil
        }
    }

    private var rationaleColor: Color {
        switch resolvedState {
        case .poisonous: return UIConfig.alertRed
        default: return UIConfig.inkGreen.opacity(0.75)
        }
    }

    private var detailBackgroundColor: Color {
        switch resolvedState {
        case .edible:
            return UIConfig.lightVerdictGreen
        case .inedible, .poisonous, .notFood:
            return UIConfig.lightVerdictRed
        case .notFound, .blank:
            return UIConfig.paper
        }
    }

    private func startStreaming() async {
        resolvedState = state
        if let frame {
            if let outcome = await container.visionPlantGate.evaluateTap(detection: detection, frame: frame) {
                switch outcome.decision {
                case .plantLike:
                    if let prediction = await container.plantClassifier.classify(crop: detection.bbox, in: frame) {
                        resolvedState = await container.plantKnowledge.resolve(
                            yoloClass: detection.yoloClass,
                            scientificName: prediction.scientificName
                        )
                    } else {
                        paragraph = UIStrings.plantUnsureRetry
                        return
                    }
                case .unsure:
                    paragraph = UIStrings.plantUnsureRetry
                    return
                case .nonPlant:
                    resolvedState = .notFood(yoloClass: detection.yoloClass)
                }
            } else {
                paragraph = UIStrings.plantUnsureRetry
                return
            }
        } else {
            print("[Gemma] no_frame_for_tap yolo_class=\(detection.yoloClass)")
            paragraph = UIStrings.plantUnsureRetry
            return
        }

        if case .blank = resolvedState { return }

        let cacheKey = self.cacheKey
        if let cached = container.gemma.cachedNarrative(forKey: cacheKey) {
            print("[Gemma] cache_hit key=\(cacheKey)")
            paragraph = cached.paragraph
            return
        }

        streamTask?.cancel()
        let stream: AsyncStream<String>
        switch resolvedState {
        case .edible(let e), .inedible(let e), .poisonous(let e):
            let blurb = await container.plantKnowledge.currentPrepBlurb()
            print("[Gemma] path=in_pack yolo_class=\(detection.yoloClass) scientific_name=\(e.scientificName) category=\(e.category.rawValue)")
            stream = container.gemma.narrate(plant: e, prepBlurb: blurb)
        case .notFound(let name):
            print("[Gemma] path=not_found scientific_name=\(name)")
            stream = container.gemma.narrateNotFound(scientificName: name)
        case .notFood(let cls):
            print("[Gemma] path=object yolo_class=\(cls)")
            stream = container.gemma.narrateObject(yoloClass: cls)
        default:
            return
        }

        isGenerating = true
        paragraph = ""
        streamTask = Task { @MainActor in
            var buffer = ""
            for await token in stream {
                if Task.isCancelled { return }
                buffer.append(token)
            }
            let cleaned = stripThinkingTags(buffer)
            print("[Gemma] done key=\(cacheKey) raw=\(buffer.count)chars cleaned=\(cleaned.count)chars")
            paragraph = cleaned
            isGenerating = false
            container.gemma.cache(narrative: GemmaNarrative(key: cacheKey, paragraph: cleaned))
        }
        await streamTask?.value
    }

    /// Strips reasoning/thinking blocks that Gemma emits before its final answer.
    /// Handles two formats:
    ///   • `<think>…</think>` / `<thinking>…</thinking>`
    ///   • `<|WORD>…<WORD|>` (mirrored pipe-token blocks, e.g. `<|channel>…<channel|>`)
    private func stripThinkingTags(_ raw: String) -> String {
        var result = raw

        // Pattern 1: <think>…</think> and <thinking>…</thinking>
        if let re = try? NSRegularExpression(
            pattern: #"<think(?:ing)?>\s*[\s\S]*?</think(?:ing)?>"#,
            options: .caseInsensitive
        ) {
            let r = NSRange(result.startIndex..., in: result)
            result = re.stringByReplacingMatches(in: result, range: r, withTemplate: "")
        }

        // Pattern 2: <|WORD>…<WORD|>  (backreference ensures open/close tags match)
        if let re = try? NSRegularExpression(
            pattern: #"<\|(\w+)>[\s\S]*?<\1\|>"#,
            options: .caseInsensitive
        ) {
            let r = NSRange(result.startIndex..., in: result)
            result = re.stringByReplacingMatches(in: result, range: r, withTemplate: "")
        }

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var displayText: String {
        if isGenerating { return "give me a sec..." }
        return paragraph.isEmpty ? "  " : paragraph
    }

    private var windowSafeAreaTop: CGFloat {
        UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .first?.keyWindow?.safeAreaInsets.top ?? 59
    }

    private var cacheKey: String {
        switch resolvedState {
        case .edible(let e), .inedible(let e), .poisonous(let e):
            return e.scientificName.lowercased()
        case .notFound(let name):
            return "notfound:" + name.lowercased()
        case .notFood(let cls):
            return "notfood:" + cls
        case .blank:
            return "blank"
        }
    }

}
