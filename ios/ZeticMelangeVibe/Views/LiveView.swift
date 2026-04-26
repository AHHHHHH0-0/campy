import SwiftUI

/// Identify mode root. Hosts the camera preview, runs the per-frame inference
/// worker, renders bounding boxes via `OverlayCanvas`, and presents the bloom
/// detail sheet on box tap.
struct LiveView: View {
    @Binding var selection: AppTab
    @Environment(\.appContainer) private var container
    @Environment(\.scenePhase) private var scenePhase

    @State private var detections: [TrackedDetection] = []
    @State private var states: [UUID: DetectionState] = [:]
    @State private var frameSize: CGSize = .zero
    @State private var presentedDetection: TrackedDetection? = nil
    @State private var presentedState: DetectionState = .blank
    @State private var presentedFrame: CameraFrame? = nil
    @State private var showHUD: Bool = false
    @State private var streamTask: Task<Void, Never>? = nil
    @State private var worker: InferenceWorker? = nil
    @State private var lastFrame: CameraFrame? = nil

    @Namespace private var bloom

    var body: some View {
        ZStack {
            CameraPreviewView(previewLayer: container.camera.previewLayer)
                .ignoresSafeArea()

            OverlayCanvas(
                detections: detections,
                states: states,
                frameSize: frameSize,
                previewLayer: container.camera.previewLayer,
                namespace: bloom,
                onTap: handleBoxTap
            )
            .ignoresSafeArea()
            .allowsHitTesting(presentedDetection == nil)

            if showHUD {
                VStack {
                    HStack {
                        DebugHUDView()
                        Spacer()
                    }
                    Spacer()
                }
                .allowsHitTesting(false)
            }

            VStack {
                Spacer()
                TabBar(selection: $selection)
            }
            .opacity(presentedDetection == nil ? 1 : 0)

            if let det = presentedDetection {
                DetailSheet(
                    detection: det,
                    state: presentedState,
                    frame: presentedFrame,
                    namespace: bloom,
                    onDismiss: dismissDetail
                )
                .transition(.asymmetric(
                    insertion: .scale(scale: 0.85).combined(with: .opacity),
                    removal: .scale(scale: 0.92).combined(with: .opacity)
                ))
                .zIndex(10)
            }
        }
        .simultaneousGesture(hudToggleGesture)
        .task {
            await startCameraAndInference()
        }
        .onDisappear {
            stopCameraAndInference()
        }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active {
                container.camera.pause()
            } else if presentedDetection == nil {
                container.camera.resume()
            }
        }
    }

    private var hudToggleGesture: some Gesture {
        // Two-second long-press anywhere on the live preview toggles the HUD.
        // Used only during threshold tuning / demo rehearsal.
        LongPressGesture(minimumDuration: 2.0)
            .onEnded { _ in showHUD.toggle() }
    }

    private func handleBoxTap(_ det: TrackedDetection, _ state: DetectionState) {
        guard presentedDetection == nil else { return }
        // Resolve highest current-frame confidence at tap location is implicit:
        // `OverlayCanvas` only forwards a tap from the topmost rendered box.
        container.camera.pause()
        withAnimation(UIConfig.Bloom.openCurve) {
            presentedDetection = det
            presentedState = state
            presentedFrame = lastFrame
        }
    }

    private func dismissDetail() {
        withAnimation(UIConfig.Bloom.closeCurve) {
            presentedDetection = nil
            presentedState = .blank
            presentedFrame = nil
        }
        container.camera.resume()
    }

    private func startCameraAndInference() async {
        let worker = InferenceWorker(
            detection: container.detection,
            tracker: BoxTracker(),
            telemetry: container.telemetry
        )
        self.worker = worker

        await container.camera.start()

        streamTask?.cancel()
        streamTask = Task { @MainActor [worker] in
            for await frame in container.camera.frames {
                if Task.isCancelled { return }
                self.lastFrame = frame
                guard let outcome = await worker.process(frame: frame) else { continue }
                self.detections = outcome.detections
                self.states = outcome.states
                self.frameSize = outcome.frameSize
            }
        }
    }

    private func stopCameraAndInference() {
        streamTask?.cancel()
        streamTask = nil
        Task { await container.camera.stop() }
    }
}
