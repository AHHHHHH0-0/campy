import AVFoundation
import SwiftUI

/// Boot sequence root.
/// 1. WelcomeView for `AppConfig.welcomeDurationSeconds`.
/// 2. Camera permission. Denied → `CameraRequiredView` indefinitely.
/// 3. Mic permission. Denied → `MicRequiredView` indefinitely.
/// 4. ModelLoadingView until `AppContainer.modelLoadProgress == 1.0`.
/// 5. Tabbed shell with horizontal slide between identify/prepare.
struct ContentView: View {

    enum Phase {
        case welcome
        case requestingCamera
        case cameraDenied
        case requestingMic
        case micDenied
        case loadingModels
        case ready
    }

    @Environment(\.appContainer) private var container
    @State private var phase: Phase = .welcome
    @State private var selection: AppTab = .identify
    @State private var lastTab: AppTab = .identify

    var body: some View {
        ZStack {
            switch phase {
            case .welcome:
                WelcomeView()
                    .transition(.opacity)
            case .requestingCamera, .requestingMic, .loadingModels:
                ModelLoadingView()
                    .transition(.opacity)
            case .cameraDenied:
                CameraRequiredView()
                    .transition(.opacity)
            case .micDenied:
                MicRequiredView()
                    .transition(.opacity)
            case .ready:
                tabbedShell
                    .transition(.opacity)
            }
        }
        .task {
            await runBootSequence()
        }
        .onChange(of: container.modelLoader.allReady) { _, ready in
            guard ready, phase == .loadingModels else { return }
            withAnimation(.easeInOut(duration: 0.25)) {
                phase = .ready
            }
        }
    }

    @ViewBuilder
    private var tabbedShell: some View {
        ZStack {
            switch selection {
            case .identify:
                LiveView(selection: $selection)
                    .transition(slideTransition(to: .identify))
            case .prepare:
                AssistantView(selection: $selection)
                    .transition(slideTransition(to: .prepare))
            }
        }
        .animation(.easeInOut(duration: UIConfig.tabTransitionDuration), value: selection)
        .onChange(of: selection) { _, new in
            lastTab = new
        }
    }

    private func slideTransition(to tab: AppTab) -> AnyTransition {
        let goingForward = tab.orderIndex >= lastTab.orderIndex
        let edgeIn: Edge = goingForward ? .trailing : .leading
        let edgeOut: Edge = goingForward ? .leading : .trailing
        return .asymmetric(insertion: .move(edge: edgeIn), removal: .move(edge: edgeOut))
    }

    // MARK: Boot

    private func runBootSequence() async {
        // 1. Welcome splash.
        try? await Task.sleep(nanoseconds: UInt64(AppConfig.welcomeDurationSeconds * 1_000_000_000))
        if Task.isCancelled { return }

        // 2. Camera permission.
        await MainActor.run {
            withAnimation(.easeInOut(duration: 0.25)) {
                phase = .requestingCamera
            }
        }
        let cameraGranted = await requestCameraPermission()
        guard cameraGranted else {
            await MainActor.run { phase = .cameraDenied }
            return
        }

        // 3. Mic permission.
        await MainActor.run { phase = .requestingMic }
        let micGranted = await requestMicrophonePermission()
        guard micGranted else {
            await MainActor.run { phase = .micDenied }
            return
        }

        // 4. Eager-load all five Zetic models.
        await MainActor.run {
            phase = .loadingModels
            container.modelLoader.startEagerLoad()
        }
    }

    private func requestCameraPermission() async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized: return true
        case .denied, .restricted: return false
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
        @unknown default: return false
        }
    }

    private func requestMicrophonePermission() async -> Bool {
        // Apple Developer Documentation: AVAudioApplication.requestRecordPermission
        // (iOS 17+). Older `AVAudioSession.recordPermission` is deprecated.
        let app = AVAudioApplication.shared
        switch app.recordPermission {
        case .granted: return true
        case .denied: return false
        case .undetermined:
            return await AVAudioApplication.requestRecordPermission()
        @unknown default: return false
        }
    }
}
