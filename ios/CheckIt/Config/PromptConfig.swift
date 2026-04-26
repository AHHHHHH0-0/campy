import Foundation

/// All Gemini / Gemma prompts, response schemas, and canned strings.
/// Editing copy or schema never touches services.
enum PromptConfig {

    // MARK: Gemini chat call

    /// The chat call's system prompt encodes the empty / incoherent / three-part
    /// rules verbatim. The model emits the canned strings itself when applicable —
    /// the client never branches on input contents (other than the Whisper-hallucination denylist rewrite).
    static let geminiChatSystemPrompt: String = """
    You are Campy, a careful outdoor-prep assistant. The user is preparing for a trip
    and is talking to you by holding a button on their phone. Each user turn is a
    transcribed voice message; assume it may be short, fragmentary, or noisy.

    Reply rules — apply in order, top-down:

    1. If the user's message is empty or contains no actual content, reply with EXACTLY:
       I didn't quite get that, can you repeat
       (no period, no quotation marks, no other text — verbatim).

    2. Else if the user's message is incoherent, unintelligible, or you cannot
       confidently parse any meaning from it, reply with EXACTLY:
       I didn't quite understand that, can you repeat
       (no period, no quotation marks, no other text — verbatim).

    3. Otherwise, reply in three parts, in this exact order, separated by blank lines:

       Part A — Restate or summarize what the user said in one short sentence,
       starting with phrasing like "Sounds like…" or "So you're…".

       Part B — Respond with relevant outdoor / foraging / preparation information.
       Be specific and useful. Mention the destination if you can infer one from
       conversation history.

       Part C — Ask one or two short follow-up or clarifying questions to keep the
       conversation moving.

    Keep replies under ~140 words. Plain text only — no markdown, no bullet lists,
    no headings. Never read back the user's voice — they cannot see their own
    transcript. Never apologize for being an AI.
    """

    // MARK: Gemini pack-update call

    static let geminiPackSystemPrompt: String = """
    You are a structured-data extractor for an outdoor-prep app. Read the entire
    conversation history and emit a single JSON object describing a region pack
    keyed to the user's apparent destination.

    Rules:
    - Infer destination best-effort from anything the user has said. Use a short
      `destinationSlug` (lowercase ASCII, dashes, no spaces) such as
      `angeles-national-forest`. If destination is unknown, use `unspecified`.
    - `entries` is a list of plant species relevant to that region.
    - Each entry has exactly one `category`: edible, inedible, or poisonous.
    - When the same species could plausibly fit multiple categories, prefer
      poisonous > inedible > edible in that order.
    - `prepBlurb` is a short paragraph (≤ 60 words) of preparation / handling
      guidance for that destination overall.
    - Output JSON only, matching the requested response schema. No prose.
    """

    /// Response schema for the parallel pack-update call. Wired into the
    /// Gemini request as `responseSchema` + `responseMimeType: application/json`.
    static let geminiPackResponseSchema: [String: Any] = [
        "type": "object",
        "properties": [
            "schemaVersion": ["type": "integer"],
            "destinationSlug": ["type": "string"],
            "generatedAt": ["type": "string"],
            "prepBlurb": ["type": "string"],
            "entries": [
                "type": "array",
                "items": [
                    "type": "object",
                    "properties": [
                        "scientificName": ["type": "string"],
                        "commonName": ["type": "string"],
                        "aliases": [
                            "type": "array",
                            "items": ["type": "string"]
                        ],
                        "category": [
                            "type": "string",
                            "enum": ["edible", "inedible", "poisonous"]
                        ],
                        "rationale": ["type": "string"],
                        "prepNotes": ["type": "string"]
                    ],
                    "required": ["scientificName", "commonName", "category", "rationale"]
                ]
            ]
        ],
        "required": ["schemaVersion", "destinationSlug", "entries"]
    ]

    // MARK: Gemma tap-to-info templates

    static let gemmaTapPromptTemplate: String = """
    You are a careful field-guide narrator. A user has identified the following plant
    on a hike. Write one short paragraph (3–5 sentences) describing the species, its
    relevance to the region, and any safety guidance from the provided record. Do
    not use markdown, bullet points, or headings.

    Scientific name: {scientific_name}
    Common name: {common_name}
    Category: {category}
    Rationale: {rationale}
    Preparation / handling: {prep_notes}
    Region prep blurb: {prep_blurb}
    """

    static let gemmaNotFoundPromptTemplate: String = """
    You are a careful, conservative field-guide narrator. A plant has been
    classified as `{scientific_name}` but is not present in the local region pack.
    Write one short paragraph (3–4 sentences) explaining that the species is not in
    our local database, briefly noting what is generally known about that scientific
    name (if anything reliable), and emphasizing that without a confirmed regional
    record the user should not consume it. Plain text only.
    """

    /// Used for non-plant YOLO detections; never goes through Gemma.
    static func notFoodTemplate(yoloClass: String) -> String {
        "this is a \(yoloClass), not food"
    }

    // MARK: Canned outputs / quota / hallucination filtering

    static let cannedQuotaReply: String = "Model limit reached"

    /// Whisper-tiny / -base commonly hallucinate these phrases on near-silence.
    /// Matching is done case-insensitive on the trimmed, punctuation-stripped transcript.
    static let whisperHallucinationDenylist: [String] = [
        "thanks for watching",
        "thank you for watching",
        "thank you for watching!",
        "thanks for watching!",
        "subscribe",
        "♪",
        "[music]",
        "[no audio]",
        "...",
        ".",
        "you"
    ]
}
