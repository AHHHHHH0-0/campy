import SwiftUI

/// Splash held until `AppContainer.modelLoadProgress` hits 1.0. Layout:
/// logo + progress bar + "Powered by Zetic" grouped at vertical center with
/// equal spacing between each element; beta-disclaimer fine-print at the
/// bottom safe-area edge.
struct ModelLoadingView: View {
    @Environment(\.appContainer) private var container

    private let itemSpacing: CGFloat = 16

    var body: some View {
        ZStack {
            UIConfig.paper.ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer()
                    .frame(maxHeight: 140)

                Image(UIStrings.checkitLogoAsset)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 240)

                Spacer().frame(height: itemSpacing)

                progressBar
                    .frame(width: 266, height: 6)

                Spacer().frame(height: itemSpacing)

                poweredBy

                Spacer()

                Text(UIStrings.betaDisclaimer)
                    .font(.system(size: 11, weight: .regular))
                    .foregroundStyle(UIConfig.inkGreen.opacity(0.55))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, UIConfig.Spacing.lg)
                    .padding(.bottom, UIConfig.Spacing.md)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    @ViewBuilder
    private var progressBar: some View {
        // Fixed-rect track + ink-green dot fill. We render the dot with a
        // rounded rectangle so the leading edge stays crisp at low progress.
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(UIConfig.sage.opacity(0.2))
                Capsule()
                    .fill(UIConfig.inkGreen)
                    .frame(width: max(6, proxy.size.width * container.modelLoadProgress))
                    .animation(.easeInOut(duration: 0.2), value: container.modelLoadProgress)
            }
        }
    }

    @ViewBuilder
    private var poweredBy: some View {
        HStack(spacing: 6) {
            Text(UIStrings.poweredByPrefix)
                .font(.system(size: 16, weight: .regular))
                .foregroundStyle(UIConfig.inkGreen.opacity(0.7))
            Image(UIStrings.poweredByZeticAsset)
                .resizable()
                .scaledToFit()
                .frame(height: 22)
        }
    }
}

#Preview {
    ModelLoadingView()
        .environment(\.appContainer, AppContainer.preview)
}
