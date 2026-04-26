import SwiftUI

/// Floating glass pill rendered above the camera preview / paper body.
/// Two fixed tabs: identify (left), prepare (right). Selection pill animates
/// between the two tab positions on selection change.
struct TabBar: View {
    @Binding var selection: AppTab
    @Namespace private var pillSpace

    var body: some View {
        HStack(spacing: 0) {
            tabButton(for: .identify)
            tabButton(for: .prepare)
        }
        .padding(6)
        .background(
            ZStack {
                Capsule()
                    .fill(UIConfig.sage.opacity(UIConfig.TabBar.fillOpacity))
                Capsule()
                    .fill(.ultraThinMaterial)
                    .opacity(0.6)
            }
        )
        .overlay(
            Capsule()
                .stroke(UIConfig.inkGreen.opacity(0.05), lineWidth: 1)
        )
        .shadow(
            color: UIConfig.inkGreen.opacity(0.15),
            radius: UIConfig.TabBar.shadowRadius,
            x: 0,
            y: UIConfig.TabBar.shadowYOffset
        )
        .padding(.horizontal, UIConfig.TabBar.horizontalPadding)
        .padding(.bottom, UIConfig.TabBar.bottomPadding)
    }

    @ViewBuilder
    private func tabButton(for tab: AppTab) -> some View {
        let isSelected = (selection == tab)
        Button {
            guard selection != tab else { return }
            withAnimation(UIConfig.TabBar.selectionAnimation) {
                selection = tab
            }
        } label: {
            HStack(spacing: 8) {
                Image(systemName: iconName(for: tab))
                    .font(.system(size: 16, weight: .semibold))
                Text(label(for: tab))
                    .font(.system(size: 12, weight: .semibold))
            }
            .foregroundStyle(isSelected ? UIConfig.inkGreen : UIConfig.inkGreen.opacity(0.55))
            .frame(maxWidth: .infinity)
            .frame(height: UIConfig.TabBar.height - 12)
            .background(
                Group {
                    if isSelected {
                        Capsule()
                            .fill(UIConfig.paper.opacity(0.9))
                            .matchedGeometryEffect(id: "tabPill", in: pillSpace)
                    }
                }
            )
        }
        .buttonStyle(.plain)
    }

    private func iconName(for tab: AppTab) -> String {
        switch tab {
        case .identify: return "viewfinder"
        case .prepare: return "checklist"
        }
    }

    private func label(for tab: AppTab) -> String {
        switch tab {
        case .identify: return UIStrings.identifyTabLabel
        case .prepare: return UIStrings.prepareTabLabel
        }
    }
}

#Preview {
    @Previewable @State var sel: AppTab = .identify
    return ZStack {
        UIConfig.paper.ignoresSafeArea()
        VStack {
            Spacer()
            TabBar(selection: $sel)
        }
    }
}
