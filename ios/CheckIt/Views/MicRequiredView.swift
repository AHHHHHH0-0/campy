import SwiftUI

/// Replaces the entire app body indefinitely when `AVAudioApplication` mic
/// permission is `.denied` or `.restricted`. No `exit(0)`, no settings deep-link.
struct MicRequiredView: View {
    var body: some View {
        ZStack {
            UIConfig.paper.ignoresSafeArea()
            Text(UIStrings.micRequiredCopy)
                .font(.system(size: 18, weight: .medium, design: .rounded))
                .foregroundStyle(UIConfig.inkGreen)
                .multilineTextAlignment(.center)
                .padding(UIConfig.Spacing.lg)
        }
    }
}

#Preview {
    MicRequiredView()
}
