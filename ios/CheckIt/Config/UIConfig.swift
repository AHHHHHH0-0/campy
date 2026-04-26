import SwiftUI

/// Color tokens, spacing scale, animation durations.
/// Light mode only — the app forces `UIUserInterfaceStyle = Light`.
enum UIConfig {

    // MARK: Brand palette (locked, exhaustive)

    static let inkGreen = Color(red: 0x1F / 255.0, green: 0x3D / 255.0, blue: 0x2B / 255.0)
    static let leafGreen = Color(red: 0x4F / 255.0, green: 0x7C / 255.0, blue: 0x45 / 255.0)
    static let sage = Color(red: 0xA8 / 255.0, green: 0xBF / 255.0, blue: 0xA3 / 255.0)
    static let neutralGray = Color(red: 0xB8 / 255.0, green: 0xBC / 255.0, blue: 0xC2 / 255.0)
    static let paper = Color(red: 0xF6 / 255.0, green: 0xF1 / 255.0, blue: 0xE7 / 255.0)
    static let lightVerdictGreen = Color(red: 0xE8 / 255.0, green: 0xF4 / 255.0, blue: 0xE6 / 255.0)
    static let lightVerdictRed = Color(red: 0xF7 / 255.0, green: 0xE7 / 255.0, blue: 0xE5 / 255.0)

    /// Dedicated semantic safety token, deliberately outside the brand palette.
    static let alertRed = Color(red: 0xC0 / 255.0, green: 0x39 / 255.0, blue: 0x2B / 255.0)

    // MARK: Spacing scale
    enum Spacing {
        static let xs: CGFloat = 4
        static let sm: CGFloat = 8
        static let md: CGFloat = 16
        static let lg: CGFloat = 24
        static let xl: CGFloat = 32
    }

    // MARK: Tab bar tokens
    enum TabBar {
        static let fillOpacity: Double = 0.9
        static let height: CGFloat = 56
        static let horizontalPadding: CGFloat = 24
        static let bottomPadding: CGFloat = 16
        static let shadowRadius: CGFloat = 12
        static let shadowYOffset: CGFloat = 4
        static let selectionAnimation = Animation.spring(response: 0.32, dampingFraction: 0.8)
    }

    // MARK: Tab transition
    static let tabTransitionDuration: Double = 0.30

    // MARK: Detail-sheet bloom curve
    enum Bloom {
        static let openDuration: Double = 0.42
        static let closeDuration: Double = 0.32
        static let openCurve = Animation.interpolatingSpring(stiffness: 220, damping: 22)
        static let closeCurve = Animation.easeInOut(duration: 0.32)
    }

    // MARK: Box stroke
    enum Box {
        static let strokeWidth: CGFloat = 2.5
        static let cornerRadius: CGFloat = 6
        static let positionSpring = Animation.spring(response: 0.18, dampingFraction: 0.85)
    }

    // MARK: Audio visualizer
    enum AudioVisualizer {
        static let baselineSize: CGFloat = 96
        static let peakScale: CGFloat = 1.7
        static let baselineScale: CGFloat = 1.0
        static let levelSmoothing: Double = 0.18
        static let levelAnimation = Animation.linear(duration: 0.05)
    }
}
