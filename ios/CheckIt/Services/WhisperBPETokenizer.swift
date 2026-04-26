import Foundation

/// Minimal BPE tokenizer just sufficient for Whisper detokenization.
///
/// Whisper's vocabulary maps integer token ids to their decoded UTF-8 surface
/// form (with the GPT-2 byte-level encoding). For decoding we only need
/// id → string, which is what `vocab.json` contains. We DO NOT implement the
/// merge step here because the decoder runs autoregressively on already-tokenized
/// outputs — we only translate integer ids back into text.
final class WhisperBPETokenizer: @unchecked Sendable {

    private let idToToken: [Int: String]
    private let endOfTextID: Int?
    private let timestampIDs: Set<Int>
    private let suppressIDs: Set<Int>

    static func loadFromBundle() -> WhisperBPETokenizer? {
        guard let vocabURL = Bundle.main.url(forResource: ModelConfig.WhisperTokenizer.vocabResource, withExtension: "json"),
              let vocabData = try? Data(contentsOf: vocabURL),
              let parsed = try? JSONSerialization.jsonObject(with: vocabData) as? [String: Int] else {
            return nil
        }
        var idToToken: [Int: String] = [:]
        idToToken.reserveCapacity(parsed.count)
        for (token, id) in parsed { idToToken[id] = token }
        return WhisperBPETokenizer(idToToken: idToToken)
    }

    init(idToToken: [Int: String]) {
        self.idToToken = idToToken
        self.endOfTextID = idToToken.first { $0.value == ModelConfig.WhisperTokenizer.endOfTextToken }?.key
        var ts: Set<Int> = []
        for (id, tok) in idToToken {
            // Whisper timestamp tokens look like `<|0.00|>`.
            if tok.hasPrefix("<|") && tok.hasSuffix("|>") {
                let inner = tok.dropFirst(2).dropLast(2)
                if Double(inner) != nil { ts.insert(id) }
            }
        }
        self.timestampIDs = ts
        self.suppressIDs = ts
    }

    var endOfTextTokenID: Int? { endOfTextID }
    var timestampTokenIDs: Set<Int> { timestampIDs }

    func startPrefixIDs() -> [Int] {
        var ids: [Int] = []
        let inverse = idToToken
        let lookup: (String) -> Int? = { name in
            for (id, tok) in inverse where tok == name { return id }
            return nil
        }
        for prefix in ModelConfig.WhisperTokenizer.englishTaskPrefixTokens {
            if let id = lookup(prefix) { ids.append(id) }
        }
        return ids
    }

    /// Translate an array of token ids back into a UTF-8 string, applying
    /// GPT-2 byte-level decoding.
    func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            guard let tok = idToToken[id] else { continue }
            // Special tokens (anything wrapped in <|…|>) don't decode to text.
            if tok.hasPrefix("<|") && tok.hasSuffix("|>") { continue }
            for ch in tok.unicodeScalars {
                if let byte = Self.byteToUnicodeReverse[ch] {
                    bytes.append(byte)
                } else if let scalarByte = UInt8(exactly: ch.value) {
                    bytes.append(scalarByte)
                }
            }
        }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }

    // GPT-2 byte → printable Unicode reversibility map (Whisper inherits this).
    private static let byteToUnicodeReverse: [Unicode.Scalar: UInt8] = {
        var bytes: [UInt8] = []
        var unicodes: [Int] = []
        let printableRanges: [(Int, Int)] = [(0x21, 0x7E), (0xA1, 0xAC), (0xAE, 0xFF)]
        for (lo, hi) in printableRanges {
            for v in lo...hi { bytes.append(UInt8(v)); unicodes.append(v) }
        }
        var n = 0
        for b in 0...255 {
            if !bytes.contains(UInt8(b)) {
                bytes.append(UInt8(b))
                unicodes.append(256 + n)
                n += 1
            }
        }
        var dict: [Unicode.Scalar: UInt8] = [:]
        for i in 0..<bytes.count {
            if let scalar = Unicode.Scalar(unicodes[i]) {
                dict[scalar] = bytes[i]
            }
        }
        return dict
    }()
}
