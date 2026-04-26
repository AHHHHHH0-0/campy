import Foundation

actor TranscriptStore: TranscriptStoreProtocol {

    private let fileManager: FileManager
    private var inMemory: [ModelReply] = []
    private var loaded: Bool = false

    init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    func load() async -> [ModelReply] {
        if loaded { return inMemory }
        loaded = true
        guard let url = transcriptURL(),
              fileManager.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url) else {
            inMemory = []
            return inMemory
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        if let decoded = try? decoder.decode([ModelReply].self, from: data) {
            inMemory = decoded
        } else {
            inMemory = []
        }
        return inMemory
    }

    func snapshot() async -> [ModelReply] {
        if !loaded { _ = await load() }
        return inMemory
    }

    func append(_ reply: ModelReply) async {
        if !loaded { _ = await load() }
        inMemory.append(reply)
        persist()
    }

    func clear() async {
        inMemory.removeAll()
        if let url = transcriptURL() {
            try? fileManager.removeItem(at: url)
        }
        loaded = true
    }

    // MARK: Internals

    private func transcriptURL() -> URL? {
        guard let support = try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ) else {
            return nil
        }
        return support.appendingPathComponent("transcript.json")
    }

    private func persist() {
        guard let url = transcriptURL() else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(inMemory) else { return }
        try? data.write(to: url, options: [.atomic])
    }
}
