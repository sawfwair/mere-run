# Music Generate

## Purpose

Generate a WAV music clip from a caption and optional structured lyrics using ACE-Step. This is the main cookbook for "make a good song" requests.

## Required Models

Managed id: `music-acestep`. The expected layout includes ACE-Step turbo, VAE, Qwen3 text encoder, and optionally the 5 Hz LM subdirectory.

## Install And Check

```bash
mere.run model pull music-acestep
mere.run music generate --help
mere.run guide music generate --model music-acestep
```

## Parameters

- positional caption: musical target description.
- `--lyrics`: inline lyrics.
- `--lyrics-file`: lyrics file; cannot be used with `--lyrics`.
- `--output`, `-o`: WAV path.
- `--model`, `-m`: managed id, model root, or checkpoints root.
- `--checkpoints-root`: root containing ACE-Step subdirectories.
- `--turbo-subdirectory`, `--vae-subdirectory`, `--lm-subdirectory`, `--text-subdirectory`: component layout overrides.
- `--use-lm`: enable 5 Hz constrained LM.
- `--duration`: output seconds.
- `--steps`, `-s`: turbo denoise steps.
- `--shift`: turbo scheduler shift.
- `--seed`: deterministic generation.
- `--audio-cover-strength`: cover conditioning strength from `0` to `1`.
- `--vocal-language`: language tag for lyric formatting.
- `--instruction`: caption instruction prefix.
- `--task-type`, `--task`: `text2music`, `cover`, `repaint`, `extract`, `lego`, or `complete`.
- `--track-name`: target track for extract/lego.
- `--complete-track-classes`: comma-separated classes for complete.
- `--non-cover`: set `isCover=false`.
- `--bpm`, `--keyscale`, `--timesignature`: musical metadata.
- `--lm-top-k`, `--lm-top-p`: constrained LM sampling controls.
- `--metadata-duration`, `--metadata-language`: metadata overrides for LM.
- `--no-tiled-vae`, `--vae-chunk-size`, `--vae-overlap`: VAE decode memory controls.
- `--quiet`, `-q`: suppress diagnostics.

## Prompting Patterns

- Caption formula: genre + instrumentation + vocal style + mood + tempo + mix/production references.
- Lyrics work best with tags like `[verse]`, `[chorus]`, `[bridge]`.
- Match `--vocal-language` to the lyrics language.
- Use `--bpm`, `--keyscale`, and `--timesignature` when rhythm or harmony must be stable.
- Use `--duration 10` to draft, then extend once the caption works.

## Examples

```bash
mere.run music generate \
  "bright indie pop, jangly electric guitars, live drums, warm female vocal, 128 bpm, summer road-trip chorus" \
  --lyrics "[verse]\nwindows down, the city disappears\n[chorus]\nwe keep driving into golden light" \
  --duration 18 \
  --bpm 128 \
  --keyscale "D major" \
  --seed 72 \
  --output ./road-trip.wav
```

```bash
mere.run music generate \
  "dark cinematic synthwave instrumental, pulsing bass, spacious drums, neon tension" \
  --duration 12 \
  --steps 8 \
  --output ./cue.wav
```

## Iteration Tips

- First iterate caption and lyrics at 10 to 20 seconds.
- Lock `--seed` after a promising groove, then adjust metadata.
- If vocals are garbled, simplify lyrics and add section tags.
- If structure drifts, try `--use-lm` with BPM/key/time metadata.

## Troubleshooting

- `--lyrics` and `--lyrics-file` conflict: use only one.
- Text encoder missing: set `--text-subdirectory` or keep the default layout.
- `--use-lm` fails: ensure the LM subdirectory exists or pass `--lm-subdirectory`.
- Audio decode memory pressure: keep tiled VAE enabled, reduce duration, or tune VAE chunk size.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/MusicGenerateCommand.swift
- https://huggingface.co/docs/diffusers/api/pipelines/ace_step
- https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md
- https://huggingface.co/ACE-Step/Ace-Step1.5
