import Foundation

/// Persisted region pack. Identical schema to `PromptConfig.geminiPackResponseSchema`
/// so what the model emits round-trips losslessly through `JSONDecoder`.
struct RegionPack: Codable, Equatable, Sendable {
    let schemaVersion: Int
    let destinationSlug: String
    let generatedAt: Date?
    let prepBlurb: String?
    var entries: [Entry]

    struct Entry: Codable, Equatable, Sendable, Identifiable {
        var id: String { scientificName.lowercased() }

        let scientificName: String
        let commonName: String
        let aliases: [String]?
        let category: Category
        let rationale: String
        let prepNotes: String?

        enum Category: String, Codable, Equatable, Sendable, CaseIterable {
            case edible
            case inedible
            case poisonous
        }
    }

    /// O(1) lookup view. Built once when the pack is loaded; `RegionPack` itself
    /// keeps `entries` as an array because the schema is array-shaped.
    func indexed() -> [String: Entry] {
        var dict: [String: Entry] = [:]
        for entry in entries {
            let key = entry.scientificName.lowercased()
            switch dict[key]?.category {
            case .none:
                dict[key] = entry
            case .some(let existing):
                if Self.severity(existing) < Self.severity(entry.category) {
                    dict[key] = entry
                }
            }
        }
        return dict
    }

    /// Pack conflict rule: poisonous beats inedible beats edible.
    private static func severity(_ c: Entry.Category) -> Int {
        switch c {
        case .edible: return 0
        case .inedible: return 1
        case .poisonous: return 2
        }
    }
}

extension RegionPack {
    static let currentSchemaVersion: Int = 1
}
