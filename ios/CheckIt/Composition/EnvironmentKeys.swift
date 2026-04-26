import SwiftUI

private struct AppContainerKey: EnvironmentKey {
    /// `defaultValue` must be evaluable from a nonisolated context, so we lazily
    /// build the preview container on first access from the main actor.
    static var defaultValue: AppContainer {
        MainActor.assumeIsolated { AppContainer.preview }
    }
}

extension EnvironmentValues {
    var appContainer: AppContainer {
        get { self[AppContainerKey.self] }
        set { self[AppContainerKey.self] = newValue }
    }
}
