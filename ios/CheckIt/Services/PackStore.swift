import Foundation

actor PackStore: PackStoreProtocol {

    private let fileManager: FileManager
    private let bundle: Bundle

    init(fileManager: FileManager = .default, bundle: Bundle = .main) {
        self.fileManager = fileManager
        self.bundle = bundle
    }

    func activePack() async -> RegionPack? {
        if let mostRecent = mostRecentOnDiskPack() {
            return mostRecent
        }
        return bundledSeedPack()
    }

    func write(pack: RegionPack) async throws {
        let directory = try regionPacksDirectory().appendingPathComponent(pack.destinationSlug, isDirectory: true)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("pack.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(pack)
        try data.write(to: url, options: [.atomic])
    }

    func deleteAll() async {
        guard let dir = try? regionPacksDirectory() else { return }
        try? fileManager.removeItem(at: dir)
    }

    // MARK: Internals

    private func regionPacksDirectory() throws -> URL {
        let support = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let dir = support.appendingPathComponent("RegionPacks", isDirectory: true)
        try fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func mostRecentOnDiskPack() -> RegionPack? {
        guard let dir = try? regionPacksDirectory(),
              let entries = try? fileManager.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey]) else {
            return nil
        }
        let candidates: [URL] = entries.flatMap { slug -> [URL] in
            let pack = slug.appendingPathComponent("pack.json")
            return fileManager.fileExists(atPath: pack.path) ? [pack] : []
        }
        let sorted = candidates.sorted { lhs, rhs in
            let l = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
            let r = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
            return l > r
        }
        for url in sorted {
            if let pack = decode(url: url) { return pack }
        }
        return nil
    }

    private func bundledSeedPack() -> RegionPack? {
        guard let url = bundle.url(forResource: "pack", withExtension: "json", subdirectory: "SeedPacks/angeles-nf") else {
            return nil
        }
        return decode(url: url)
    }

    private func decode(url: URL) -> RegionPack? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(RegionPack.self, from: data)
    }
}
