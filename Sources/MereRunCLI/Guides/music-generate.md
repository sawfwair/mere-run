# Music Generate

## Purpose

Generate a WAV music clip from a caption. ACE-Step supports optional structured
lyrics, while Magenta RT2 supports native Apple Silicon offline and realtime
prompt-to-music generation.

## Required Models

Managed ids:

- `music-acestep`: ACE-Step turbo, VAE, Qwen3 text encoder, and optionally the
  5 Hz LM subdirectory.
- `music-acestep-xl-turbo`: ACE-Step 1.5 XL turbo DiT, VAE, and Qwen3 text
  encoder.
- `music-acestep-xl-turbo-lm4b`: ACE-Step 1.5 XL turbo plus the optional 4B
  5 Hz LM subdirectory.
- `music-magenta-rt2-small`: Magenta RealTime 2 small exported runtime assets.
- `music-magenta-rt2-base`: Magenta RealTime 2 base exported runtime assets.

## Install And Check

```bash
mere.run model pull music-acestep
mere.run model pull music-acestep-xl-turbo
mere.run model pull music-acestep-xl-turbo-lm4b
mere.run model pull music-magenta-rt2-small
mere.run music generate --help
mere.run guide music generate --model music-acestep
mere.run guide music generate --model music-acestep-xl-turbo
mere.run guide music generate --model music-magenta-rt2-small
```

## Parameters

- positional caption: musical target description.
- `--lyrics`: inline lyrics.
- `--lyrics-file`: lyrics file; cannot be used with `--lyrics`.
- `--output`, `-o`: WAV path.
- `--model`, `-m`: managed id, model root, or checkpoints root.
- `--checkpoints-root`: root containing ACE-Step subdirectories.
- `--turbo-subdirectory`, `--vae-subdirectory`, `--lm-subdirectory`, `--text-subdirectory`: component layout overrides.
- `--use-lm`: enable 5 Hz constrained LM for supported text-to-music tasks.
- `--lm-subdirectory acestep-5Hz-lm-4B`: force the optional 4B LM when using
  `music-acestep-xl-turbo-lm4b`.
- `--duration`: output seconds.
- `--steps`, `-s`: turbo denoise steps.
- `--shift`: turbo scheduler shift; ACE-Step CLI default is `3.0`, matching upstream.
- `--seed`: deterministic generation.
- `--source-audio`: source song for ACE-Step cover conditioning; implies cover mode unless `--non-cover` is set.
- `--reference-audio`: optional ACE-Step timbre reference audio file(s).
- `--audio-cover-strength`: cover conditioning strength from `0` to `1`.
- `--cover-noise-strength`: source-latent noise initialization strength from
  `0` to `1` for ACE-Step covers. `0` starts from pure noise; higher values
  start closer to the source song.
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
- `--temperature`, `--top-k`: Magenta RT2 sampling controls.
- `--style-conditioning streaming|full`: choose realtime C++ style-token
  masking or Python-like full MusicCoCa style conditioning.
- `--cfg-musiccoca`, `--cfg-notes`, `--cfg-drums`: Magenta RT2 guidance scales.
- `--drumless`: Magenta RT2 drumless generation.
- `--unmask-width`, `--seed-rotation`: Magenta RT2 generation controls.
- `--prefill-silence`, `--prefill-duration`: Magenta RT2 realtime prefill controls.
- `--quiet`, `-q`: suppress diagnostics.

ACE-Step uses upstream-style native Haar DCW sampler correction by default for
cleaner diffusion latents before VAE decode.

The default ACE-Step managed ID uses the smaller 1.5 turbo DiT. Use
`music-acestep-xl-turbo` for the ACE-Step 1.5 XL turbo DiT on larger machines,
or `music-acestep-xl-turbo-lm4b` with `--use-lm` and
`--lm-subdirectory acestep-5Hz-lm-4B` when you want the optional 4B 5 Hz LM.
ACE-Step cover/repaint/extract tasks skip the LM phase, matching upstream,
because they use source-audio conditioning directly.

For realtime Magenta RT2 runs, use `mere.run music realtime`. It accepts the
same Magenta controls plus `--play` or `--no-play` and optional `--output` WAV
capture. Add `--interactive` to steer while it runs with stdin commands such as
`prompt <text>`, `temp <value>`, `noteon <0-131>`, `noteoff <0-131>`,
`style streaming|full`, `drumless on|off`, `reset`, and `quit`.

## Prompting Patterns

- Caption formula: genre + instrumentation + vocal style + mood + tempo + mix/production references.
- Lyrics work best with tags like `[verse]`, `[chorus]`, `[bridge]`.
- Match `--vocal-language` to the lyrics language.
- Use `--bpm`, `--keyscale`, and `--timesignature` when rhythm or harmony must be stable.
- Use `--duration 10` to draft, then extend once the caption works.
- For Magenta RT2, put all musical direction in the prompt; lyrics and ACE-Step
  task modes are not supported by that runtime.

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

```bash
mere.run music generate \
  "dream-pop cover with soft vocals, lush guitars, wide chorus" \
  --source-audio ./song.mp3 \
  --lyrics-file ./cover-lyrics.txt \
  --audio-cover-strength 1.0 \
  --duration 20 \
  --output ./cover.wav
```

```bash
mere.run music generate \
  "modern reggaeton dance club remix, 96 bpm dembow rhythm, syncopated kick-snare groove, punchy 808 sub bass, bright Latin percussion, perreo club energy, chopped pop vocals, glossy synth stabs" \
  --model music-acestep-xl-turbo \
  --source-audio ./song.mp3 \
  --lyrics-file ./lyrics.txt \
  --audio-cover-strength 0.20 \
  --cover-noise-strength 0.0 \
  --output ./reggaeton-cover.wav
```

```bash
mere.run music generate \
  "ambient modular synths with brushed drums, slow evolving harmony" \
  --model music-magenta-rt2-small \
  --duration 4 \
  --temperature 0.8 \
  --output ./magenta-cue.wav
```

```bash
mere.run music realtime \
  "drumless glassy arpeggios with soft tape hiss" \
  --model music-magenta-rt2-small \
  --duration 2 \
  --output ./magenta-live.wav \
  --no-play
```

## Iteration Tips

- First iterate caption and lyrics at 10 to 20 seconds.
- Lock `--seed` after a promising groove, then adjust metadata.
- If vocals are garbled, simplify lyrics and add section tags.
- If a cover drifts, keep `--audio-cover-strength 1.0`, avoid `--use-lm`, and
  make the caption explicitly ask to preserve melody, tempo, phrasing, and structure.
- For style-transfer covers, start around `--audio-cover-strength 0.2` and keep
  `--cover-noise-strength 0.0` so the prompt can steer genre and arrangement.
  Lower `--audio-cover-strength` if the original still dominates; only raise
  `--cover-noise-strength` when you want to re-anchor the result to the source
  song's contour.
- For stronger style transfer, generate or provide a short style/timbre example
  and pass it with `--reference-audio` while keeping the original song in
  `--source-audio`.
- For Magenta RT2, use `music realtime --output --no-play --duration 2` for a
  fast headless smoke before running an audible session.

## Troubleshooting

- `--lyrics` and `--lyrics-file` conflict: use only one.
- Text encoder missing: set `--text-subdirectory` or keep the default layout.
- `--use-lm` fails: ensure the LM subdirectory exists or pass `--lm-subdirectory`.
  Cover/repaint/extract tasks skip LM even if the flag is present.
- Audio decode memory pressure: keep tiled VAE enabled, reduce duration, or tune VAE chunk size.
- Magenta RT2 unsupported runtime: build `vendor/magentart.xcframework` with
  `scripts/rebuild_magentart_xcframework.sh` on Apple Silicon macOS, then
  rebuild `mere.run`.
- Magenta RT2 missing assets: pull the managed model or provide a local root
  with exported `.mlxfn` files plus `resources/musiccoca` and
  `resources/spectrostream`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/MusicGenerateCommand.swift
- https://huggingface.co/docs/diffusers/api/pipelines/ace_step
- https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md
- https://huggingface.co/ACE-Step/Ace-Step1.5
- https://huggingface.co/ACE-Step/acestep-v15-xl-turbo
- https://huggingface.co/ACE-Step/acestep-5Hz-lm-4B
- https://github.com/magenta/magenta-realtime
- https://huggingface.co/google/magenta-realtime-2
