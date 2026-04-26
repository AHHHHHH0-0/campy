import Foundation

protocol PackStoreProtocol: AnyObject, Sendable {
    /// Returns the freshest on-disk pack written by the Gemini pack-update call.
    /// Falls back to the bundled seed pack at
    /// `Resources/SeedPacks/angeles-nf/pack.json` so identify can produce
    /// edible / poisonous verdicts on day-zero installs and after triple-tap clears.
    func activePack() async -> RegionPack?

    /// Persist a pack to `Application Support/RegionPacks/{slug}/pack.json`.
    func write(pack: RegionPack) async throws

    /// Recursively remove `Application Support/RegionPacks/`. The bundled seed
    /// pack lives in the app binary and is unaffected.
    func deleteAll() async
}
