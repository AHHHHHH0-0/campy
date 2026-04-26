import Foundation
import os

/// Deterministic scientific-name lookup against the active region pack.
/// Owns the result-engine path that maps `(yoloClass, scientificName)` →
/// `DetectionState`.
final class OfflinePlantKnowledgeService: @unchecked Sendable {

    struct CacheState: Sendable {
        var slug: String?
        var indexed: [String: RegionPack.Entry]
        var blurb: String?
    }

    private let packStore: any PackStoreProtocol
    private let cacheLock = OSAllocatedUnfairLock<CacheState>(initialState: CacheState(slug: nil, indexed: [:], blurb: nil))

    init(packStore: any PackStoreProtocol) {
        self.packStore = packStore
    }

    /// Resolves a `DetectionState` for one detection. `scientificName` is `nil`
    /// for non-plant detections (which short-circuit to the static template).
    func resolve(yoloClass: String, scientificName: String?) async -> DetectionState {
        guard let scientificName, !scientificName.isEmpty else {
            print("[KnowledgeService] no scientific name — routing yolo_class=\(yoloClass) as notFood")
            return .notFood(yoloClass: yoloClass)
        }
        await refreshCacheIfNeeded()
        let (entry, packSize) = cacheLock.withLock { state in
            (state.indexed[scientificName.lowercased()], state.indexed.count)
        }
        print("[KnowledgeService] lookup key=\(scientificName.lowercased()) pack_size=\(packSize)")
        guard let entry else {
            print("[KnowledgeService] miss — \(scientificName) not in pack")
            return .notFound(scientificName: scientificName)
        }
        print("[KnowledgeService] hit common_name=\(entry.commonName) category=\(entry.category.rawValue)")
        switch entry.category {
        case .edible: return .edible(entry)
        case .inedible: return .inedible(entry)
        case .poisonous: return .poisonous(entry)
        }
    }

    func currentPrepBlurb() async -> String? {
        await refreshCacheIfNeeded()
        return cacheLock.withLock { $0.blurb }
    }

    /// Forces a re-read; called after the pack-update call writes a fresh pack.
    func invalidate() {
        cacheLock.withLock { $0 = CacheState(slug: nil, indexed: [:], blurb: nil) }
    }

    // MARK: Internals

    private func refreshCacheIfNeeded() async {
        let needsRefresh = cacheLock.withLock { $0.slug == nil }
        if !needsRefresh { return }
        guard let pack = await packStore.activePack() else { return }
        let indexed = pack.indexed()
        cacheLock.withLock { state in
            state.slug = pack.destinationSlug
            state.indexed = indexed
            state.blurb = pack.prepBlurb
        }
    }
}
