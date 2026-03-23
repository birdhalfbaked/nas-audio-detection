# Flight Audio Detection

Detect ATC-style key phrases from WAV audio using phoneme transcription, fuzzy phrase matching, and phraseology tries.

## What this project does

- Transcribes audio into timestamped phoneme segments with a Wav2Vec2 CTC model.
- Matches key phrases in phoneme space using edit-distance based scoring.
- Returns structured `KeyPhraseSegment` detections (phrase, time span, confidence fields).
- Resolves overlapping detections by keeping the most confident match.
- Supports a globally loaded common phraseology trie, with optional facility-specific extension.

## Requirements

- Python `>=3.14`
- `espeak-ng` installed and available on PATH (required for phonemizer)
- Python dependencies installed with `uv`

`espeak-ng` release page: [https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0)

## Install

```bash
uv sync --group dev
```

## Build common phraseology trie

Generate the shared generic phraseology trie:

```bash
uv run python scripts/build_common_tries.py
```

This creates:

- `data/common_trie.json`

If `espeak-ng` is missing, this script fails by design (to avoid silently generating a non-phoneme trie).

## CLI

Main entrypoint:

```bash
uv run python -m app.main detect --audio-file <path-to-wav> [options]
```

### Common examples

Use generic global phraseology only:

```bash
uv run python -m app.main detect --audio-file tests/fixtures/test_audio1.wav
```

Use facility context (common + facility-specific if available):

```bash
uv run python -m app.main detect --audio-file tests/fixtures/test_audio1.wav --facility ksea
```

Add explicit phrases:

```bash
uv run python -m app.main detect \
  --audio-file tests/fixtures/test_audio1.wav \
  --phrase "contact departure" \
  --phrase "contact approach"
```

Use phrase file:

```bash
uv run python -m app.main detect \
  --audio-file tests/fixtures/test_audio1.wav \
  --phrases-file phrases.txt
```

Emit JSON:

```bash
uv run python -m app.main detect \
  --audio-file tests/fixtures/test_audio1.wav \
  --json
```

### Detect options

- `--audio-file` (required): input WAV path
- `--facility`: facility ID for facility-specific phraseology
- `--phrase`: repeatable direct phrase input
- `--phrases-file`: newline-delimited phrase list
- `--target-sample-rate`: model input sample rate (default `16000`)
- `--chunk-seconds`: chunk size in seconds (default `10.0`)
- `--stride-seconds`: overlap stride in seconds (default `2.0`)
- `--max-normalized-distance`: phrase match acceptance threshold (default `0.45`)
- `--json`: print structured JSON output

## Key components

- `app/audio/processor.py`
  - `AudioTranscriber.transcribe_phonemes(...)`: timestamped phoneme segments
  - `AudioTranscriber.__call__(...)`: key phrase detection pipeline
  - overlap resolution by confidence on colliding detections

- `app/audio/phrase_tagger.py`
  - `KeyPhraseSegment` dataclass
  - `find_best_phrase_match(...)` fuzzy phoneme matcher

- `app/utils.py`
  - `PhonemeTrie` with phoneme insertion/search
  - trie serialization (`to_dict`/`from_dict`) and word extraction

- `app/phraseology.py`
  - globally cached common trie loader
  - facility trie loader

- `app/facilities.py`
  - facility wrapper combining common + facility phraseology

## Testing

Run all current tests:

```bash
uv run --group dev pytest
```

Run specific audio tests:

```bash
uv run --group dev pytest tests/audio/test_processor.py tests/audio/test_phrase_tagger.py
```

## Notes

- Phrase matching is phoneme/edit-distance based (approximate), not strict forced alignment.
- Timestamp quality is suitable for detection/tagging workflows, but may drift slightly at boundaries.