# Music Analyze

## Purpose

Analyze an audio file with ACE-Step 5 Hz LM audio understanding and print
structured metadata for cover, remix, or catalog workflows. The command decodes
regular audio files to 48 kHz stereo, converts the source into ACE audio-code
tokens, runs the LM understanding phase, and returns JSON on stdout.

## Required Models

Managed ids:

- `music-acestep`: ACE-Step turbo, VAE, and 5 Hz LM.
- `music-acestep-xl-turbo-lm4b`: ACE-Step XL turbo plus the optional 4B 5 Hz
  LM.

## Install And Check

```bash
mere.run model pull music-acestep
mere.run model pull music-acestep-xl-turbo-lm4b
mere.run music analyze --help
mere.run guide music analyze --model music-acestep-xl-turbo-lm4b
```

## Parameters

- positional audio: audio file to analyze.
- `--model`, `-m`: managed ACE-Step id, model root, or checkpoints root.
- `--checkpoints-root`: root containing ACE-Step subdirectories.
- `--turbo-subdirectory`, `--vae-subdirectory`, `--lm-subdirectory`: component
  layout overrides.
- `--duration`: analyze the first N seconds instead of the full decoded input.
- `--max-new-tokens`: maximum LM tokens for the understanding pass.
- `--lm-temperature`, `--lm-top-k`, `--lm-top-p`: LM sampling controls.
- `--include-raw-lm`: include the raw LM output in the JSON.
- `--include-audio-codes`: include serialized ACE audio-code tokens in the
  JSON. This can be very large for full songs.
- `--quiet`, `-q`: suppress diagnostics.

## Output

stdout is JSON. Diagnostics go to stderr.

Important fields:

- `metadata.bpm`
- `metadata.keyscale`
- `metadata.timesignature`
- `metadata.language`
- `metadata.caption`
- `metadata.lyrics`
- `inputDurationSeconds`
- `analyzedDurationSeconds`

## Examples

```bash
mere.run music analyze ./song.mp3 \
  --model music-acestep-xl-turbo-lm4b \
  --lm-subdirectory acestep-5Hz-lm-4B
```

```bash
mere.run music analyze ./song.mp3 \
  --duration 30 \
  --include-raw-lm \
  > ./song-analysis.json
```

```bash
mere.run music generate \
  "modern reggaeton dance club remix, dembow rhythm, punchy bass" \
  --model music-acestep-xl-turbo-lm4b \
  --source-audio ./song.mp3 \
  --analyze-source-audio \
  --audio-cover-strength 0.20 \
  --cover-noise-strength 0.0 \
  --output ./reggaeton-cover.wav
```

## Iteration Tips

- Use `music analyze` before a cover when you want to inspect source tempo,
  key/scale, and time signature without generating audio.
- Use `--duration 30` for quick probes on long songs, then run without
  `--duration` when you want whole-song understanding.
- Keep user-provided generation metadata explicit when you want to override the
  source analysis.
- Use `--include-raw-lm` when debugging parser behavior or unexpected missing
  metadata.

## Troubleshooting

- Missing LM: pull `music-acestep` or `music-acestep-xl-turbo-lm4b`, or pass
  `--lm-subdirectory`.
- XL turbo without LM: use `music-acestep-xl-turbo-lm4b` for analysis.
- Huge JSON: avoid `--include-audio-codes` for full songs unless you are
  debugging the token stream.
- Slow analysis: pass `--duration` to inspect a shorter prefix first.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/MusicAnalyzeCommand.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCore/ACEStep/ACEStepMusicUnderstanding.swift
