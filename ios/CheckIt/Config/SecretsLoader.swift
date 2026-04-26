import Foundation

/// Reads API keys from Info.plist (substituted at build time from Secrets.xcconfig).
/// Fatal-errors fast on missing keys so we never silently ship a broken binary.
enum SecretsLoader {

    static var geminiAPIKey: String { value(forKey: "GEMINI_API_KEY") }
    static var zeticPersonalKey: String { value(forKey: "ZETIC_PERSONAL_KEY") }

    private static func value(forKey key: String) -> String {
        guard let raw = Bundle.main.object(forInfoDictionaryKey: key) as? String else {
            fatalError("Missing Info.plist key '\(key)'. Configure ios/CheckIt/Config/Secrets.xcconfig.")
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              !trimmed.hasPrefix("REPLACE_WITH_") else {
            fatalError("Info.plist key '\(key)' is empty or unreplaced. Edit Secrets.xcconfig.")
        }
        return trimmed
    }
}
