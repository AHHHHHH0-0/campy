import SwiftUI

@main
struct CheckItApp: App {
    @State private var container: AppContainer = AppContainer.makeProduction()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.appContainer, container)
                .preferredColorScheme(.light)
        }
    }
}
