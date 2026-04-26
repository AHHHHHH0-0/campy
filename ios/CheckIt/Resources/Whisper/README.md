# Whisper tokenizer assets

This directory hosts the BPE tokenizer files needed to detokenize the Zetic-hosted
Whisper decoder output back into UTF-8 text:

- `whisper-vocab.json` — vocabulary map, keys are surface forms (with GPT-2 byte
  encoding) and values are integer ids. Same file across `whisper-tiny`,
  `whisper-base`, and `whisper-small` (multilingual variants).
- `whisper-merges.txt` — BPE merge table. Not strictly required for decoding
  (we go id → text only) but kept here for completeness if a future encoding
  path is needed on-device.

Source: [openai/whisper](https://github.com/openai/whisper) repository,
`whisper/assets/multilingual.tiktoken` rebundled as JSON, or the
`tokenizer.json` / `vocab.json` shipped with HuggingFace's `whisper-tiny`.

If neither file is present at runtime, `WhisperBPETokenizer.loadFromBundle()`
returns `nil` and `WhisperTranscriberService.transcribe(...)` falls through to
an empty-string return. That routes through the chat-call system prompt's
empty-input path so the user sees the canned `"I didn't quite get that, can
you repeat"` reply rather than an internal error. Demo rehearsal therefore
requires the real assets to be checked in — the build does not fail without them.
