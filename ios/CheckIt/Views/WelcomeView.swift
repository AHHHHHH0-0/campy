import SwiftUI

/// Splash shown for `AppConfig.welcomeDurationSeconds` (3 s) before any system
/// permission prompts. Paper background, centered "Welcome to Campy" wordmark.
struct WelcomeView: View {
    var body: some View {
        ZStack {
            UIConfig.paper.ignoresSafeArea()
            Text(UIStrings.welcomeWordmark)
                .font(.system(size: 32, weight: .semibold, design: .rounded))
                .foregroundStyle(UIConfig.inkGreen)
                .multilineTextAlignment(.center)
        }
    }
}

#Preview {
    WelcomeView()
}
