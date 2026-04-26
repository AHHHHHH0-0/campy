import Foundation

/// Root navigation tab. Drives `ContentView`'s tab state and the `TabBar` selection.
enum AppTab: String, CaseIterable, Sendable, Codable {
    case identify
    case prepare

    /// Order index used by the horizontal slide transition direction.
    var orderIndex: Int {
        switch self {
        case .identify: return 0
        case .prepare: return 1
        }
    }
}
