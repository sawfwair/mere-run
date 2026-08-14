# Music Generate

## Purpose

Generate a WAV music clip from a caption. MiniMax Music 3 generates complete
44.1 kHz stereo songs from a caption plus structured lyrics, ACE-Step supports
generation and source-audio editing, and Magenta RT2 supports native Apple
Silicon offline and realtime prompt-to-music generation.

## Required Models

Managed ids:

- `music-acestep`: ACE-Step turbo, VAE, Qwen3 text encoder, and optionally the
  5 Hz LM subdirectory.
- `music-acestep-xl-turbo`: ACE-Step 1.5 XL turbo DiT, VAE, and Qwen3 text
  encoder.
- `music-acestep-xl-turbo-lm4b`: ACE-Step 1.5 XL turbo plus the optional 4B
  5 Hz LM subdirectory.
- `music-acestep-xl-sft`: ACE-Step 1.5 XL-SFT with full non-distilled
  CFG/APG/ADG inference.
- `music-acestep-xl-base`: ACE-Step 1.5 XL-Base, including extract, lego, and
  complete tasks.
- `music-acestep-lm-1.7b`: independently pullable upstream-default 5 Hz
  planner; pairs with any ACE-Step DiT.
- `music-acestep-lm-4b`: independently pullable optional 4B 5 Hz planner.
- `music-minimax-music3`: MiniMax Music 3 language model, RVQ depth decoder,
  flow transformer, condition encoder, and stereo vocoder.
- `music-magenta-rt2-small`: Magenta RealTime 2 small exported runtime assets.
- `music-magenta-rt2-base`: Magenta RealTime 2 base exported runtime assets.

## Install And Check

```bash
mere.run model pull music-acestep
mere.run model pull music-acestep-xl-turbo
mere.run model pull music-acestep-xl-turbo-lm4b
mere.run model pull music-acestep-lm-1.7b
mere.run model pull music-minimax-music3 --accept-model-license
mere.run model pull music-magenta-rt2-small
mere.run music generate --help
mere.run music analyze --help
mere.run guide music generate --model music-acestep
mere.run guide music generate --model music-acestep-xl-turbo
mere.run guide music generate --model music-minimax-music3
mere.run guide music generate --model music-magenta-rt2-small
```

## Parameters

- positional caption: musical target description.
- `--lyrics`: inline lyrics.
- `--lyrics-file`: lyrics file; cannot be used with `--lyrics`.
- `--output`, `-o`: WAV path.
- `--quality draft|song|final|edit`: model-aware steps, sampler, guidance, LM,
  automatic-duration, and best-of-N policy.
- `--candidates`, `--best-of`: warm-session candidate count; results are
  technically scored and stably ranked.
- `--retake-seed`, `--retake-variance`: reproducible spherical retake noise.
- `--flow-edit`, `--source-caption`, `--source-lyrics`,
  `--flow-edit-n-min`, `--flow-edit-n-max`, `--flow-edit-n-average`: semantic
  source-to-target flow editing.
- `--adapter`, `--adapter-kind`, `--adapter-scale`: stack PEFT LoRA and
  LyCORIS LoKr adapters.
- `--export-format pcm16|pcm24|float32`, `--normalize`, fades, and dither:
  mastering/export controls.
- `--recipe-output`, `--no-recipe`: exact reproducibility sidecar controls.
- `--lrc-file`, `--lrc-output`: synchronized lyrics input/output.
- `--daw-bundle`, `--stems`: portable DAW session and Base-only extraction.
- `--model`, `-m`: managed id, model root, or checkpoints root.
- `--checkpoints-root`: root containing ACE-Step subdirectories.
- `--turbo-subdirectory`, `--vae-subdirectory`, `--text-subdirectory`:
  component layout overrides. `--lm-subdirectory` is the legacy same-root LM
  override; prefer `--lm-model` for an independent planner.
- `--lm-model`: independently select a managed planner id or local planner
  root. If the DiT root has no LM, the default is `music-acestep-lm-1.7b`.
- `--use-lm`: enable 5 Hz constrained LM for supported text-to-music tasks.
- `--lm-model music-acestep-lm-4b`: explicitly select the optional 4B planner.
- `--duration`: output seconds.
- `--minimum-duration`: MiniMax Music 3 decoded-audio duration floor.
- `--min-frames`, `--max-frames`: MiniMax Music 3 exact 25 Hz floor and limit.
- `--performance-mode reference|optimized|q8|q4`: MiniMax Music 3 execution tier;
  `optimized` is the BF16 default and `q8` is the recommended turbo tier.
- `--steps`, `-s`: denoise steps; MiniMax Music 3 defaults to `30`.
- `--shift`: turbo scheduler shift; ACE-Step CLI default is `3.0`, matching upstream.
- `--seed`: deterministic generation.
- `--source-audio`: source song for ACE-Step cover, repaint, extract, lego, or
  complete conditioning. With the default task it implies `cover`.
- `--analyze-source-audio`: run ACE-Step 5 Hz LM audio understanding before a
  cover and fill missing BPM, key/scale, language, and time signature metadata
  from the source audio. Explicit `--bpm`, `--keyscale`, `--timesignature`, and
  `--vocal-language` values always win.
- `--reference-audio`: optional ACE-Step timbre reference audio file(s).
- `--audio-cover-strength`: cover conditioning strength from `0` to `1`.
- `--cover-noise-strength`: source-latent noise initialization strength from
  `0` to `1` for ACE-Step covers. `0` starts from pure noise; higher values
  start closer to the source song.
- `--vocal-language`: one language tag used by both LM metadata constraints and
  lyric formatting. The compatibility `--metadata-language` alias overrides it
  when both are present.
- `--instruction`: caption instruction prefix.
- `--task-type`, `--task`: `text2music`, `repaint`, `cover`, `cover-nofsq`,
  `extract`, `lego`, or `complete`. Unknown values are rejected during parsing.
- `--track-name`: target track for extract/lego.
- `--complete-track-classes`: comma-separated classes for complete.
- `--non-cover`: compatibility alias for the explicit `cover-nofsq` task.
- `--repaint-start`, `--repaint-end`: edit range in seconds; `-1` uses the
  source end.
- `--chunk-mask-mode auto|explicit`: use learned automatic masking or pass the
  exact edit span into the DiT chunk-mask channels.
- `--repaint-mode conservative|balanced|aggressive`: source-preservation
  policy. Conservative injects the source throughout denoising and uses the
  widest boundary fades; aggressive performs pure diffusion.
- `--repaint-strength`: balanced-mode aggressiveness from `0` to `1`.
- `--bpm`, `--keyscale`, `--timesignature`: musical metadata.
- `--instrumental`: use upstream's explicit `[Instrumental]` lyric marker;
  cannot be combined with lyrics input.
- `--lm-temperature`, `--lm-top-k`, `--lm-top-p`,
  `--lm-repetition-penalty`: constrained planner sampling controls. Temperature
  defaults to upstream's `0.85`; repetition penalty defaults to the neutral
  upstream value `1.0`.
- `--lm-cfg-scale`, `--lm-negative-prompt`: classifier-free guidance for the
  semantic-code phase. Upstream defaults are `2.0` and `NO USER INPUT`;
  metadata planning always remains unguided at `1.0`.
- `--candidates`: explicitly opt into local best-of-N technical ranking.
  Quality presets generate one candidate, matching upstream's disabled
  auto-score default.
- `--metadata-duration`: compatibility alias for `--duration`; conflicting
  values are rejected so the planner and renderer cannot diverge.
- `--metadata-language`: compatibility override for `--vocal-language`.
- `--no-tiled-vae`, `--vae-chunk-size`, `--vae-overlap`: VAE decode memory controls.
- `--temperature`, `--top-k`: Magenta RT2 sampling controls.
- `--style-conditioning streaming|full`: choose realtime C++ style-token
  masking or Python-like full MusicCoCa style conditioning.
- `--cfg-musiccoca`, `--cfg-notes`, `--cfg-drums`: Magenta RT2 guidance scales.
- `--drumless`: Magenta RT2 drumless generation.
- `--unmask-width`, `--seed-rotation`: Magenta RT2 generation controls.
- `--prefill-silence`, `--prefill-duration`: Magenta RT2 realtime prefill controls.
- `--list-midi-inputs`: list CoreMIDI input sources and exit.
- `--midi-monitor`: monitor a CoreMIDI input source without loading Magenta RT2.
- `--midi-log-events`: log parsed MIDI note and CC events to stderr.
- `--midi-log-raw`: log raw CoreMIDI packet bytes to stderr.
- `--midi-input`: CoreMIDI source name or unique ID for live note/control steering.
- `--midi-channel`: MIDI channel to listen to, `1` through `16` or `all`.
- `--midi-note-offset`: transpose incoming MIDI notes before sending them to Magenta RT2.
- `--midi-cc`: repeatable CC mapping in the form `cc=target:min:max`; targets
  include `temp`, `topk`, `mc`, `notes`, `drums`, `drumless`, `unmask`,
  `seed`, and `onset`.
- `--quiet`, `-q`: suppress diagnostics.

MiniMax Music 3 requires `--lyrics`, `--lyrics-file`, `--lrc-file`, or
`--instrumental`. Its native staged runtime autoregressively generates 25 Hz
semantic and residual codes, conditions a flow transformer, and decodes stereo
audio without a Python process. `--duration` accepts up to 360 seconds;
`--minimum-duration` prevents semantic EOS and rounds for the vocoder hop so
the decoded WAV meets the requested floor. `--guidance-scale` defaults to `1.7`.
The default `--performance-mode optimized` keeps BF16 quality while using
compact/fused projections, cached depth decode, and batched flow guidance;
`q8` is the recommended quantized turbo tier, with `q4` available for maximum
memory reduction. ACE-Step source audio, adapters, ranked
candidates, stems, DAW bundles, and LRC export do not apply to this model. A
reproducibility recipe is written beside the WAV unless `--no-recipe` is used.
The selective managed download is approximately 26.6 GiB and requires
explicit acknowledgement of the upstream MiniMax-Music3 Community License.

ACE-Step uses upstream-style native Haar DCW sampler correction by default for
cleaner diffusion latents before VAE decode.

The default ACE-Step managed ID uses the smaller 1.5 turbo DiT and upstream
default 1.7B planner. Use `music-acestep-xl-turbo`, `music-acestep-xl-sft`, or
`music-acestep-xl-base` for a different DiT while keeping the same planner.
Pass `--lm-model music-acestep-lm-4b` only when you explicitly want the optional
4B planner. Planner metadata is merged with the upstream rule that explicit
user values win; diagnostics print the effective metadata, not discarded LM
suggestions.
ACE-Step cover, cover-nofsq, repaint, and extract tasks skip the LM phase,
matching upstream. Turbo and SFT checkpoints support text-to-music, repaint,
cover, and cover-nofsq. Extract, lego, and complete are Base-only; the CLI
rejects those tasks on Turbo/SFT checkpoints before loading model weights.
XL-SFT and XL-Base use native continuous scheduling, CFG/APG/ADG, velocity
stabilization, and Euler or Heun integration.

Use a warm resident server when generating repeatedly:

```bash
mere.run music serve \
  --model music-acestep-xl-sft \
  --lm-model music-acestep-lm-1.7b \
  --adapter ./house-style.safetensors \
  --port 8081

curl http://127.0.0.1:8081/v1/audio/music \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"sleek nocturnal house","quality":"final","candidates":4}'
```

Train a directly reloadable adapter from a JSON/JSONL manifest containing
`audio`, `caption`, and optional `lyrics`:

```bash
mere.run music train-adapter \
  --model music-acestep-xl-turbo \
  --dataset ./music-training.jsonl \
  --kind lokr \
  --rank 8 \
  --alpha 16 \
  --output ./my-style.safetensors
```

For realtime Magenta RT2 runs, use `mere.run music realtime`. It accepts the
same Magenta controls plus `--play` or `--no-play` and optional `--output` WAV
capture. Add `--interactive` to steer while it runs with stdin commands such as
`prompt <text>`, `temp <value>`, `noteon <0-131>`, `noteoff <0-131>`,
`style streaming|full`, `drumless on|off`, `reset`, and `quit`. On macOS, add
`--midi-input` to steer notes and mapped controls from a CoreMIDI device such
as OP-1, and use `--midi-monitor --midi-log-raw` to verify that controller
events arrive before loading Magenta RT2.

## Workflow Choices

Use this guide as a command topic, then focus it with `--model` when needed:

```bash
mere.run guide music generate --model music-acestep
mere.run guide music generate --model music-acestep-xl-turbo
mere.run guide music generate --model music-minimax-music3
mere.run guide music generate --model music-magenta-rt2-small
```

For a MiniMax Music 3 song, make the caption describe the production and pass
sectioned lyrics separately:

```bash
mere.run music generate \
  "cinematic synth-pop, female lead, 118 bpm, wide guitars" \
  --model music-minimax-music3 \
  --lyrics-file ./lyrics.txt \
  --duration 30 \
  --steps 30 \
  --seed 7 \
  --output ./minimax-song.wav
```

For ACE-Step text-to-music, start with a direct caption and optional structured
lyrics. Add `--use-lm` only when you want the 5 Hz LM planning phase.

For ACE-Step covers, keep the source song in `--source-audio` and put the target
arrangement in the caption. Add `--analyze-source-audio` when you want missing
BPM, key/scale, language, and time signature filled from the source before
generation. Keep `--audio-cover-strength 1.0` for a faithful cover.

For ACE-Step style-transfer covers or remixes, lower `--audio-cover-strength`
so the caption can steer genre and arrangement. Start around `0.20`, keep
`--cover-noise-strength 0.0`, and only raise source noise when the result needs
more of the original song's contour.

For a surgical edit, select `--task repaint`, pass the original song, and set
the edit range. The runtime silences only that latent span for conditioning,
re-injects appropriately noised clean source latents outside it during
denoising, blends the latent boundaries, then restores the original pre-VAE
waveform outside the range:

```bash
mere.run music generate \
  "replace the bridge with a bigger live-drum chorus" \
  --model music-acestep-xl-turbo \
  --task repaint \
  --source-audio ./song.wav \
  --repaint-start 42.0 \
  --repaint-end 58.5 \
  --repaint-mode balanced \
  --repaint-strength 0.4 \
  --output ./song-repaint.wav
```

For analysis-first workflows, run `mere.run music analyze` separately, inspect
the JSON, then pass explicit `--bpm`, `--keyscale`, `--timesignature`, or
`--vocal-language` values to pin the generated result. Explicit generation
flags always win over source analysis.

## Prompting Patterns

- Caption formula: genre + instrumentation + vocal style + mood + tempo + mix/production references.
- Lyrics work best with tags like `[verse]`, `[chorus]`, `[bridge]`.
- Match `--vocal-language` to the lyrics language.
- Use `--bpm`, `--keyscale`, and `--timesignature` when rhythm or harmony must be stable.
- Use `music analyze` first when you want to inspect source BPM, key/scale, and
  time signature before choosing generation overrides.
- For faithful covers, say what must stay fixed: melody, tempo, phrasing,
  structure, language, and vocal range.
- For style-transfer covers, make the new genre concrete with rhythm, drums,
  bass, percussion, mix, and club/stage/room context.
- Keep source lyrics sectioned with tags like `[verse]`, `[chorus]`, and
  `[bridge]`; the model responds better to visible song structure than to a
  plain paragraph.
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
  --instrumental \
  --duration 12 \
  --steps 8 \
  --output ./cue.wav
```

```bash
mere.run music generate \
  "dream-pop cover with soft vocals, lush guitars, wide chorus" \
  --source-audio ./song.mp3 \
  --analyze-source-audio \
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
  --analyze-source-audio \
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

```bash
mere.run music realtime --list-midi-inputs
mere.run music realtime \
  --midi-monitor \
  --midi-input "OP-1 Bluetooth" \
  --midi-log-raw \
  --duration 30
mere.run music realtime \
  "minimal synth pop, dry drums, tape-warped bass" \
  --model music-magenta-rt2-small \
  --duration 120 \
  --midi-input "OP-1 Bluetooth" \
  --midi-channel all \
  --midi-log-events \
  --midi-cc 1=temp:0.2:1.4 \
  --midi-cc 2=drums:0:2
```

## Iteration Tips

- First iterate caption and lyrics at 10 to 20 seconds.
- Lock `--seed` after a promising groove, then adjust metadata.
- If vocals are garbled, simplify lyrics and add section tags.
- If a cover drifts, keep `--audio-cover-strength 1.0`, avoid `--use-lm`, and
  make the caption explicitly ask to preserve melody, tempo, phrasing, and structure.
- For covers with unknown metadata, add `--analyze-source-audio` so the 5 Hz LM
  can infer BPM, key/scale, language, and time signature from the source's audio
  codes before direct DiT cover generation. User-provided metadata stays pinned.
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
- For MIDI control, run `music realtime --list-midi-inputs` first, then pass a
  stable source name or unique ID with `--midi-input`. If the controller is
  silent, run `music realtime --midi-monitor --midi-log-raw` against that
  source and confirm note-on/note-off bytes before loading the model. Keep
  prompt changes on stdin; MIDI is intended for notes and continuous controls.

## Troubleshooting

- `--lyrics` and `--lyrics-file` conflict: use only one.
- Text encoder missing: set `--text-subdirectory` or keep the default layout.
- `--use-lm` fails: pull `music-acestep-lm-1.7b` or pass `--lm-model` to a
  managed planner id/local root. Use `--lm-subdirectory` only for legacy
  same-root layouts.
  Cover, cover-nofsq, repaint, and extract skip LM even if the flag is present.
- Base-only task rejected: extract, lego, and complete require
  `music-acestep-xl-base`; Turbo and SFT intentionally reject them.
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
