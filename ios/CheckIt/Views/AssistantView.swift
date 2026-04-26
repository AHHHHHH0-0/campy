import Network
import SwiftUI

/// Prepare mode root. Renders the model-only transcript on the paper background,
/// drives press-and-hold raw-PCM capture, and dispatches Whisper → Gemini.
/// Triple-tap on the body opens the delete-conversation confirm sheet.
struct AssistantView: View {
    @Binding var selection: AppTab
    @Environment(\.appContainer) private var container
    @Environment(\.scenePhase) private var scenePhase

    @State private var replies: [ModelReply] = []
    @State private var isHolding: Bool = false
    @State private var isOnline: Bool = true
    @State private var audioLevel: Double = 0
    @State private var isPending: Bool = false
    @State private var showClearConfirm: Bool = false

    @State private var holdTask: Task<Void, Never>? = nil
    @State private var holdCapTask: Task<Void, Never>? = nil
    @State private var levelTask: Task<Void, Never>? = nil
    @State private var cancelTask: Task<Void, Never>? = nil
    @State private var pathMonitor: NWPathMonitor? = nil

    var body: some View {
        ZStack(alignment: .bottom) {
            UIConfig.paper.ignoresSafeArea()

            // The press-and-hold target is the body above the tab bar.
            // The tab bar is added on top so its taps stay routable.
            ZStack {
                transcriptScroll

                if replies.isEmpty && !isHolding && !isPending {
                    Text(UIStrings.tapAndHoldToSpeak)
                        .font(.system(size: 16, weight: .regular, design: .rounded))
                        .foregroundStyle(UIConfig.inkGreen.opacity(0.5))
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                        .allowsHitTesting(false)
                }

                if isHolding {
                    holdOverlay
                }
            }
            .contentShape(Rectangle())
            .simultaneousGesture(holdGesture)
            .simultaneousGesture(tripleTapGesture)

            TabBar(selection: $selection)
        }
        .task {
            await loadInitial()
        }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active && isHolding {
                cancelHold()
            }
        }
        .alert(UIStrings.clearConfirmTitle, isPresented: $showClearConfirm) {
            Button(UIStrings.clearConfirmYes, role: .destructive) {
                Task { await performClear() }
            }
            Button(UIStrings.clearConfirmNo, role: .cancel) {}
        } message: {
            Text(UIStrings.clearConfirmMessage)
        }
    }

    // MARK: Transcript

    @ViewBuilder
    private var transcriptScroll: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(replies) { reply in
                        replyRow(reply: reply)
                            .id(reply.id)
                    }
                    if isPending {
                        pendingRow
                    }
                }
                .padding(.horizontal, UIConfig.Spacing.lg)
                .padding(.top, UIConfig.Spacing.lg)
                .padding(.bottom, UIConfig.TabBar.height + UIConfig.TabBar.bottomPadding + UIConfig.Spacing.lg)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .onChange(of: replies.count) { _, _ in
                guard let last = replies.last else { return }
                withAnimation(.easeOut(duration: 0.25)) {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
    }

    @ViewBuilder
    private func replyRow(reply: ModelReply) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            divider
                .padding(.top, AppConfig.transcriptDividerPaddingTop)
                .padding(.bottom, AppConfig.transcriptDividerPaddingBottom)
            Text(reply.text)
                .font(.system(size: 16, weight: .regular, design: .serif))
                .foregroundStyle(UIConfig.inkGreen)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder
    private var pendingRow: some View {
        VStack(alignment: .leading, spacing: 0) {
            divider
                .padding(.top, AppConfig.transcriptDividerPaddingTop)
                .padding(.bottom, AppConfig.transcriptDividerPaddingBottom)
            ShimmerLine()
                .frame(height: 14)
                .padding(.vertical, 4)
            ShimmerLine()
                .frame(height: 14)
                .frame(maxWidth: 220, alignment: .leading)
                .padding(.vertical, 4)
        }
    }

    @ViewBuilder
    private var divider: some View {
        // Default to a small leaf-shaped logo until the asset is supplied.
        if UIImage(named: UIStrings.transcriptDividerAsset) != nil {
            Image(UIStrings.transcriptDividerAsset)
                .resizable()
                .renderingMode(.template)
                .scaledToFit()
                .frame(height: 24)
                .foregroundStyle(UIConfig.leafGreen)
        } else {
            HStack(spacing: 6) {
                Image(systemName: "leaf.fill")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(UIConfig.leafGreen)
                Spacer()
            }
            .frame(height: 24)
        }
    }

    // MARK: Hold overlay

    @ViewBuilder
    private var holdOverlay: some View {
        ZStack {
            Rectangle()
                .fill(.ultraThinMaterial)
                .ignoresSafeArea()
            if isOnline {
                AudioVisualizerCircle(level: audioLevel)
            } else {
                Text(UIStrings.noNetworkHoldMessage)
                    .font(.system(size: 18, weight: .medium, design: .rounded))
                    .foregroundStyle(UIConfig.inkGreen)
            }
        }
        .transition(.opacity.animation(.easeInOut(duration: 0.18)))
    }

    // MARK: Gestures

    private var holdGesture: some Gesture {
        // LongPress fires once at 0.5 s; we sequence it with a 0-distance Drag so
        // we can detect release / drag-off events. SwiftUI sequencing semantics
        // are documented under `Gesture/sequenced(before:)`.
        LongPressGesture(minimumDuration: AppConfig.holdMinimumDuration)
            .sequenced(before: DragGesture(minimumDistance: 0))
            .onChanged { value in
                switch value {
                case .first:
                    break
                case .second(let firstFinished, _):
                    guard firstFinished, !isPending, !isHolding else { return }
                    startHoldIfReady()
                }
            }
            .onEnded { _ in
                guard isHolding else { return }
                finalizeHoldAndDispatch()
            }
    }

    private var tripleTapGesture: some Gesture {
        // Triple-tap is invisible; the long-press's threshold is 0.5 s so a
        // sustained hold cannot accidentally satisfy `count: 3` because each
        // tap subgesture cancels once the touch exceeds tap timeouts.
        TapGesture(count: 3)
            .onEnded {
                guard !isHolding, !isPending else { return }
                showClearConfirm = true
            }
    }

    // MARK: Hold lifecycle

    private func startHoldIfReady() {
        // Called once the LongPress threshold (0.5 s) fires. We start mic
        // capture immediately when online, otherwise we just show the
        // "No network connection" overlay until the user releases.
        isHolding = true
        isOnline = currentlyOnline()
        guard isOnline else { return }

        holdTask?.cancel()
        holdTask = Task { @MainActor in
            do {
                try await container.speechInput.start()
                if !isHolding { return }
                listenLevels()
                listenCancels()
                armHardCap()
            } catch {
                cancelHold()
            }
        }
    }

    private func finalizeHoldAndDispatch() {
        guard isHolding else { return }
        let wasOnline = isOnline
        teardownHoldUI()
        guard wasOnline else { return }

        Task { @MainActor in
            guard let captured = await container.speechInput.stop() else { return }
            await dispatch(captured: captured)
        }
    }

    private func cancelHold() {
        guard isHolding else { return }
        teardownHoldUI()
        container.speechInput.cancel()
    }

    private func teardownHoldUI() {
        withAnimation(.easeInOut(duration: 0.18)) {
            isHolding = false
        }
        audioLevel = 0
        holdTask?.cancel(); holdTask = nil
        holdCapTask?.cancel(); holdCapTask = nil
        levelTask?.cancel(); levelTask = nil
        cancelTask?.cancel(); cancelTask = nil
    }

    private func armHardCap() {
        holdCapTask?.cancel()
        holdCapTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: UInt64(AppConfig.holdHardCapSeconds * 1_000_000_000))
            if Task.isCancelled || !isHolding { return }
            finalizeHoldAndDispatch()
        }
    }

    private func listenLevels() {
        levelTask?.cancel()
        levelTask = Task { @MainActor in
            for await level in container.speechInput.levelStream {
                if Task.isCancelled { return }
                audioLevel = level
            }
        }
    }

    private func listenCancels() {
        cancelTask?.cancel()
        cancelTask = Task { @MainActor in
            for await _ in container.speechInput.cancelStream {
                if Task.isCancelled { return }
                cancelHold()
                return
            }
        }
    }

    // MARK: Dispatch

    private func dispatch(captured: CapturedAudio) async {
        isPending = true
        let transcript = await container.whisper.transcribe(audio: captured)
        let result = await container.gemini.dispatch(transcript: transcript)
        let reply = ModelReply(text: result.chatReply)
        await container.transcriptStore.append(reply)
        replies = await container.transcriptStore.snapshot()
        if result.updatedPack != nil {
            // GeminiPackService already persisted via PackStore. Refresh the
            // offline knowledge cache so the next frame uses the new pack.
            container.plantKnowledge.invalidate()
        }
        await MainActor.run {
            container.telemetry.lastChatLatencyMs = result.chatLatencyMs
            container.telemetry.lastPackLatencyMs = result.packLatencyMs ?? 0
        }
        isPending = false
    }

    // MARK: Permissions, network, lifecycle

    private func currentlyOnline() -> Bool {
        guard let monitor = pathMonitor else { return true }
        return monitor.currentPath.status == .satisfied
    }

    private func loadInitial() async {
        replies = await container.transcriptStore.load()
        if pathMonitor == nil {
            let monitor = NWPathMonitor()
            monitor.start(queue: .global(qos: .utility))
            pathMonitor = monitor
        }
    }

    private func performClear() async {
        await container.transcriptStore.clear()
        await container.packStore.deleteAll()
        container.plantKnowledge.invalidate()
        container.gemma.clearCache()
        replies = []
    }
}

private struct ShimmerLine: View {
    @State private var phase: CGFloat = 0
    var body: some View {
        Capsule()
            .fill(UIConfig.sage.opacity(0.35))
            .overlay(
                LinearGradient(
                    colors: [
                        UIConfig.sage.opacity(0.0),
                        UIConfig.paper.opacity(0.7),
                        UIConfig.sage.opacity(0.0)
                    ],
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .mask(Capsule())
                .offset(x: phase)
            )
            .clipShape(Capsule())
            .onAppear {
                withAnimation(.linear(duration: 1.2).repeatForever(autoreverses: false)) {
                    phase = 240
                }
            }
    }
}
