import Foundation

/// What the camera pipeline produces for a single tracked detection. Drives the
/// per-box color in `OverlayCanvas` and the contents of `DetailSheet` on tap.
enum DetectionState: Equatable, Sendable {
    case blank
    case edible(RegionPack.Entry)
    case inedible(RegionPack.Entry)
    case poisonous(RegionPack.Entry)
    case notFound(scientificName: String)
    case notFood(yoloClass: String)

    /// Human-readable single-line label rendered inside the bounding box.
    var label: String {
        switch self {
        case .blank: return ""
        case .edible(let e): return "identified as: \(e.commonName)"
        case .inedible(let e): return "inedible: \(e.commonName)"
        case .poisonous(let e): return "poisonous: \(e.commonName)"
        case .notFound(let name): return "\(name) — not in local database"
        case .notFood(let cls): return cls
        }
    }
}
