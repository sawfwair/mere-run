# Music Runtime

Write a track from a text prompt, cover an existing song in a different genre,
play a model live from a MIDI controller, split a mix into vocal and
instrumental WAVs, or turn it back into instrument-separated MIDI with tempo,
meter, and key already filled in.
Generation, analysis, adapter training, resident serving, realtime steering,
source separation, and transcription are native Swift and MLX paths — Magenta
RT2 takes CC knobs while it is still generating.

## Commands

| Command | What it does |
| --- | --- |
| `mere.run music generate` | Generate audio from a music prompt. |
| `mere.run music analyze` | Analyze source audio with ACE-Step audio understanding. |
| `mere.run music serve` | Keep an ACE-Step pipeline resident behind the native music API. |
| `mere.run music train-adapter` | Train a reloadable ACE-Step LoRA or LoKr adapter. |
| `mere.run music realtime` | Run Magenta RealTime 2 music generation, steerable from stdin or CoreMIDI. |
| `mere.run music separate` | Separate stems or restore audio with native BS/MelBand RoFormer models. |
| `mere.run music transcribe` | Transcribe a full music mix into instrument-separated MIDI with MuScriptor. |

The macOS Studio app exposes its music workflows through first-class
workspaces backed by `MereRunContract`. Its primary Music composer covers the
normal production loop—create or edit, source/reference audio, quality and LM
planning, ranked candidates, adapter stacks, stems, LRC, recipes, and DAW
export. **Music Tools** provides structured audio analysis, MuScriptor controls
and an embedded MIDI piano roll, plus resident-server start/stop and health.
**Realtime** owns Magenta playback and MIDI steering. **Training Studio** owns
LoRA/LoKr dataset inspection, launch, metrics, and run comparison. Advanced
remains available for raw command-level control. App-to-CLI tests reject any
emitted flag absent from `mere.run catalog --json`.

## Model family

- `music-acestep`
- `music-acestep-xl-turbo`
- `music-acestep-xl-turbo-lm4b`
- `music-acestep-xl-sft`
- `music-acestep-xl-base`
- `music-magenta-rt2-small`
- `music-magenta-rt2-base`
- `music-muscriptor-small`
- `music-muscriptor-medium`
- `music-muscriptor-large`
- `music-separate-bs-roformer-viperx-1297`
- `music-separate-bs-roformer-4stem`
- `music-separate-mel-roformer-dereverb`
- `music-separate-mel-roformer-denoise`

## Guides

Music guidance follows the command/cookbook shape used by the rest of
`mere.run`: choose the command first, then focus the guide with a model id.

```bash
mere.run guide music generate --model music-acestep
mere.run guide music generate --model music-acestep-xl-turbo
mere.run guide music analyze --model music-acestep-xl-turbo-lm4b
mere.run guide music generate --model music-magenta-rt2-small
mere.run guide music separate --model music-separate-bs-roformer-viperx-1297
mere.run guide music separate --model music-separate-bs-roformer-4stem
mere.run guide music separate --model music-separate-mel-roformer-dereverb
mere.run guide music separate --model music-separate-mel-roformer-denoise
mere.run guide music transcribe --model music-muscriptor-medium
```

There is no separate `ace-step` guide topic. ACE-Step text-to-music, covers,
style-transfer covers, and source-audio understanding are documented under
`music generate` and `music analyze`, because those are the public CLI surfaces.

## Typical workflow

```bash
swift run mere.run music generate \
  "upbeat electronic groove" \
  --output ./track.wav

swift run mere.run music generate \
  "dream-pop cover with soft vocals" \
  --source-audio ./song.mp3 \
  --analyze-source-audio \
  --audio-cover-strength 1.0 \
  --output ./cover.wav

swift run mere.run music generate \
  "modern reggaeton dance club remix, 96 bpm dembow rhythm, syncopated kick-snare groove, punchy 808 sub bass, bright Latin percussion" \
  --model music-acestep-xl-turbo \
  --source-audio ./song.mp3 \
  --analyze-source-audio \
  --audio-cover-strength 0.20 \
  --cover-noise-strength 0.0 \
  --output ./reggaeton-cover.wav

swift run mere.run music analyze ./song.mp3 \
  --model music-acestep-xl-turbo-lm4b \
  --lm-subdirectory acestep-5Hz-lm-4B \
  > ./song-analysis.json

swift run mere.run model pull music-acestep-xl-turbo
swift run mere.run music generate \
  "cinematic synth pop with bright vocal harmonies" \
  --model music-acestep-xl-turbo \
  --output ./xl-track.wav

swift run mere.run model pull music-acestep-xl-turbo-lm4b
swift run mere.run music generate \
  "arena-scale rock anthem with stacked vocals" \
  --model music-acestep-xl-turbo-lm4b \
  --use-lm \
  --lm-subdirectory acestep-5Hz-lm-4B \
  --output ./xl-lm4b-track.wav

swift run mere.run music realtime \
  "ambient modular synths with brushed drums" \
  --model music-magenta-rt2-small \
  --duration 4 \
  --output ./live.wav \
  --no-play

swift run mere.run model pull music-muscriptor-medium --accept-model-license
swift run mere.run music transcribe ./song.mp3 --output ./song.mid

swift run mere.run model pull music-separate-bs-roformer-viperx-1297
swift run mere.run music separate ./song.mp3 --output-dir ./song-stems

swift run mere.run model pull music-separate-bs-roformer-4stem
swift run mere.run music separate ./song.mp3 \
  --model music-separate-bs-roformer-4stem \
  --output-dir ./song-4stems

swift run mere.run model pull music-separate-mel-roformer-dereverb
swift run mere.run music separate ./room.wav \
  --model music-separate-mel-roformer-dereverb \
  --output-dir ./room-restored

swift run mere.run model pull music-separate-mel-roformer-denoise
swift run mere.run music separate ./noisy.wav \
  --model music-separate-mel-roformer-denoise \
  --output-dir ./noise-restored
```

`music separate` decodes the source at 44.1 kHz stereo, runs the pinned ViperX
1297 BS-RoFormer checkpoint, and writes `vocals.wav`, `instrumental.wav`, and
`separation.json`. The instrumental is computed from the decoded mixture minus
the vocal estimate, so the two stems close back to the exact working mixture.
The JSON manifest is also emitted on stdout; progress stays on stderr.

The `music-separate-bs-roformer-4stem` profile uses the same native
band-split graph with its separately pinned 384-dimensional, eight-layer
checkpoint. It writes `drums.wav`, `bass.wav`, `other.wav`, `vocals.wav`, and
`separation.json` in the checkpoint's published output order.

The MelBand profiles use a distinct 60-band native graph. Dereverb writes
`noreverb.wav`; denoise writes `dry.wav`. Each mask is scattered back to its
original STFT bins and averaged where mel bands overlap. The models share an
exact 684-tensor, 228,203,172-scalar geometry but retain different weight and
source-config hashes. The dereverb inference config publishes overlap `2`; the
denoise config publishes overlap `4`.

The accepted AEmotion Studio snapshot is explicitly MIT-licensed and pinned at
revision `d323194290f8488ea51814143806609bfbd7a1e5`. mere.run verifies the exact
model, upstream YAML, model-card README, and license hashes before loading.
Inference preserves each profile's published chunk geometry, centered
2,048-point STFT, 441-sample hop, DC filtering, 10% linear fades, and overlap
default. Use a higher valid `--overlap` for denser chunk blending at additional
compute cost, or `--dtype float32` for an all-float32 model path.

MuScriptor predicts notes, instruments, and absolute event times. For MIDI
output, mere.run adds a native musical-context pass that estimates tempo and
beat phase from source-audio accents plus decoded note onsets, estimates meter
from the beat-accent cycle, and estimates key from duration-weighted pitch
classes. Standard tempo, time-signature, and key-signature events are embedded,
with the meter repeated at the first downbeat to establish the bar boundary,
without quantizing or moving notes. Use `--context-output` to inspect the
values, confidence scores, and beat positions as JSON, or
`--no-musical-context` for the legacy fixed-120-BPM writer.

MuScriptor treats `--chunk-batch-size` (default `4`) as an upper bound, not a
forced allocation. On Apple Silicon, the runtime subtracts current MLX active
and cache allocations plus a reserve of the greater of 4 GiB or one-eighth of
physical memory. It estimates each requested chunk at
`numLayers * dim * 65,536 * max(1, beamSize)` bytes for bfloat16 and float16;
float32 doubles that lane estimate. A second, model-complexity-scaled limit
caps useful live beam lanes at 8 for the large checkpoint and 32 for medium and
small, so large with `--beam-size 4` uses at most two chunks together even when
memory allows more. This limit controls cross-chunk grouping. If one requested
beam is wider than the budget, its live-beam forwards are microbatched while
preserving the requested search width. The effective chunk group is selected
once after model load at transcription start. The memory clamp is skipped when
a unified-memory profile is unavailable, including non-Apple Linux hosts; the
model-complexity limit still applies. Explicit `--chunk-batch-size 1` always
preserves the lowest-memory single-chunk path. Persistent cache state still
scales with the requested beam width, so the policy reduces cross-chunk and
forward pressure but does not guarantee admission when one beam cannot fit.

The cap reflects a matched warm M4 Max 128 GB measurement with
`music-muscriptor-large`, 20 seconds/four chunks, beam size 4, and 64 maximum
tokens per chunk. The pre-batching baseline took 22.38 seconds at 11.15 GB peak
physical footprint. Chunk batch 1 took 4.70 seconds at 28.17 GB; chunk batch 2
had a 3.69-second median at 50.38 GB; and chunk batch 4 took 6.16 seconds at
87.48 GB. All JSON outputs had identical SHA-256 hashes. The adaptive
two-chunk path was therefore about 6.1x faster than baseline at about 4.5x the
peak footprint, while the four-chunk group was both slower and substantially
larger. Lower-headroom systems fall back to one chunk, which measured about
4.8x faster than baseline at about 2.5x its peak footprint.

Greedy and sampling pipeline selected tokens into the next model step. Beam
search packs live beams across the effective chunk group into as few bounded
forwards per step as the lane budget allows, keeps an independently forked
typed cache lane for every beam, and removes ended beams from later forwards.

ACE-Step generation uses the upstream CLI turbo shift default (`--shift 3.0`)
and the native Haar DCW sampler correction (`double`, low `0.05`, high `0.02`)
before VAE decode. The XL turbo managed ID installs the 4B DiT decoder plus the
base ACE-Step VAE and Qwen3 text encoder; the `-lm4b` variant also installs the
optional 4B 5 Hz LM for `--use-lm` runs. ACE-Step cover/repaint/extract tasks
follow upstream and skip the 5 Hz LM phase so source-audio conditioning stays
faithful.

ACE-Step task routing is typed and checkpoint-aware. Turbo/SFT support
text-to-music, repaint, cover, and cover-nofsq; extract, lego, and complete are
accepted only for Base checkpoints. Unknown task names fail in argument
parsing, while incompatible checkpoint/task pairs fail before weight loading.
XL-SFT and XL-Base use the native continuous flow schedule, real conditional
and unconditional decoder batches, CFG/APG/ADG guidance intervals, velocity
norm/EMA stabilization, and Euler or Heun ODE integration. Turbo remains the
fast distilled path. Base uniquely enables extract, lego, and complete.

`--quality draft|song|final|edit` is model-aware. It selects checkpoint-safe
steps, sampler, guidance, velocity stabilization, LM planning policy, automatic
duration behavior, and a warm best-of-N count. Explicit flags still override
the preset. `final` prefers the 4B planner when it is installed. Best-of-N
ranking checks finite samples, level, clipping, DC offset, crest factor,
spectral flatness, frame-energy movement, periodicity, time-varying spectral
structure, and tail continuity. This prevents loud stationary noise or a
prematurely dead ending from winning on level statistics alone.

Every ACE-Step generation writes 48 kHz stereo 24-bit WAV by default plus a
schema 2 reproducible recipe JSON. The recipe records exact checkpoint
repositories and immutable revisions, adapter hashes and scales,
prompt/lyrics/instruction, the final effective BPM, duration, key/scale, vocal
language and time signature, task/edit configuration, inference controls,
candidate seeds and technical scores, export policy, and input/output hashes.
When the 5 Hz LM is active, each candidate also records its semantic audio-code
count. The generation seed drives both LM sampling and diffusion.
`--export-format float32` preserves a floating-point master; `--daw-bundle`
adds candidates, extracted stems, synchronized lyric markers, and a portable
REAPER project.

Retakes use exact spherical noise interpolation between `--seed` and
`--retake-seed`. `--flow-edit` implements the upstream source/target velocity
difference field over a configurable normalized window, including Monte Carlo
forward-noise averaging and target-only finishing denoise. It is distinct from
repaint: repaint preserves audio outside a time range, while flow edit morphs
the whole source toward a new semantic target.

PEFT LoRA and LyCORIS LoKr adapters load natively with `--adapter`; multiple
files stack and may use one shared or per-adapter scale. LoKr uses factored
Kronecker evaluation instead of materializing full decoder deltas. Train either
format with `music train-adapter`; its objective matches ACE-Step flow
matching, and its output is directly reloadable by `music generate` or the
resident server. Adapter training writes the same durable `run_started`,
per-step loss/progress, `run_finished`, and `run_failed` event stream used by
Training Studio, so a music run has live feedback and survives app relaunch.

`music serve` holds the complete pipeline and its adapters in memory. It
provides `GET /health`, `POST /v1/audio/music`, and serialized
`POST /v1/audio/music/batches`, with JSON/base64 or raw WAV responses. Binding
outside loopback requires a bearer token. The API mirrors the CLI controls for
checkpoint-aware tasks, quality, steps and scheduler, CFG/APG/ADG, retakes,
cover strength/noise, repaint, flow edit, reference audio, LM metadata and
sampling, complete-track classes, and tiled VAE decode. Song/final requests
without `duration_seconds` use the resident LM planner; every JSON result
returns `conditioning_metadata` with the values actually used.

Batch items may select independent `candidates` values. The server serializes
them through the warm session, returns every ranked candidate and exactly one
selected winner per item, and rejects a request whose `model` does not match
the resident model. Batch responses are JSON; raw `response_format: "wav"` is
available on the single-generation endpoint.

Repaint is a real bounded edit, not a cover alias. `--repaint-start` and
`--repaint-end` produce the upstream 25 Hz latent mask. The requested span is
replaced by the checkpoint silence latent for conditioning, clean source
latents outside it are re-injected at the appropriate noise level during early
denoising, and latent crossfades soften both boundaries. After VAE decode the
runtime splices the original pre-VAE waveform back outside the edit span, with
a short waveform crossfade. `--repaint-mode` selects conservative, balanced,
or aggressive preservation; `--repaint-strength` tunes balanced mode.

For covers, `--analyze-source-audio` runs ACE-Step audio understanding before
the direct DiT cover pass. It converts the source audio to 5 Hz audio codes,
asks the LM for source BPM, key/scale, language, and time signature, and fills
only metadata fields you did not pass explicitly.

Use `music analyze` when you want that same ACE-Step audio-understanding result
as a standalone JSON artifact before deciding how to prompt a cover or remix.
It accepts the same ACE-Step model/checkpoint layout flags plus an optional
`--duration` prefix limit for fast probes.

For faithful covers, keep `--audio-cover-strength 1.0` and leave
`--cover-noise-strength` at its default `0.0`. For style-transfer covers, lower
`--audio-cover-strength` so the text prompt can steer genre, keep
`--cover-noise-strength 0.0` while exploring, and use `--reference-audio` for an
optional target-style/timbre example.

For demo-style steering, pass `--interactive`. The command reads stdin while it
runs, paces generation to realtime, and applies changes between native frames:

```bash
swift run mere.run music realtime \
  "ambient modular synths" \
  --model music-magenta-rt2-small \
  --duration 30 \
  --interactive
```

Supported steering commands are `prompt <text>`, `style streaming|full`,
`temp <value>`, `topk <value>`, `mc <value>`, `notes <value>`,
`drums <value>`, `noteon <0-131>`, `noteoff <0-131>`, `onset 0|1`,
`drumless on|off`, `unmask <value>`, `seed <value>`, `reset`, `quit`, and
`help`.

On macOS, `music realtime` can also listen to CoreMIDI input. Use
`--list-midi-inputs` to find the source name or unique ID, then pass
`--midi-input` to map incoming note-on/note-off messages to the same Magenta
RT2 note controls used by stdin. Use `--midi-monitor` with `--midi-log-raw`
when checking a controller before loading Magenta RT2:

```bash
swift run mere.run music realtime --list-midi-inputs
swift run mere.run music realtime \
  --midi-monitor \
  --midi-input "OP-1 Bluetooth" \
  --midi-log-raw \
  --duration 30
swift run mere.run music realtime \
  "minimal synth pop, dry drums, tape-warped bass" \
  --model music-magenta-rt2-small \
  --duration 120 \
  --midi-input "OP-1 Bluetooth" \
  --midi-channel all \
  --midi-log-events \
  --midi-cc 1=temp:0.2:1.4 \
  --midi-cc 2=drums:0:2
```

`--midi-cc` mappings use `cc=target:min:max`. Supported targets are `temp`,
`topk`, `mc`, `notes`, `drums`, `drumless`, `unmask`, `seed`, and `onset`.
`--midi-log-events` writes parsed note and CC events to stderr during realtime
runs, while `--midi-log-raw` writes raw packet bytes. Prompt changes still use
stdin and may briefly stall while the prompt encoder runs; MIDI is intended for
notes and continuous controls.

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/MusicAnalyzeCommand.swift`
- `Sources/MereRunCLI/Commands/MusicGenerateCommand.swift`
- `Sources/MereRunCLI/Commands/MusicServeCommand.swift`
- `Sources/MereRunCLI/Commands/MusicTrainAdapterCommand.swift`
- `Sources/MereRunCLI/Commands/MusicRealtimeCommand.swift`
- `Sources/MereRunCLI/Commands/MusicTranscribeCommand.swift`

### Runtime

- `Sources/MereRunCore/ACEStep/ACEStepPipeline.swift`
- `Sources/MereRunCore/ACEStep/ACEStepPipeline+Prompting.swift`
- `Sources/MereRunCore/ACEStep/ACEStepPipeline+Generation.swift`
- `Sources/MereRunCore/ACEStep/ACEStepTask.swift`
- `Sources/MereRunCore/ACEStep/ACEStepRepaint.swift`
- `Sources/MereRunCore/ACEStep/ACEStepFlowEdit.swift`
- `Sources/MereRunCore/ACEStep/ACEStepGenerationSession.swift`
- `Sources/MereRunCore/ACEStep/ACEStepAdapter.swift`
- `Sources/MereRunCore/ACEStep/ACEStepAdapterTrainer.swift`
- `Sources/MereRunCore/MagentaRT2/MagentaRT2Resources.swift`
- `Sources/MereRunCore/MagentaRT2/MagentaRT2Renderer.swift`
- `Sources/MereRunCore/MagentaRT2/MagentaRT2RealtimeSession.swift`
- `Sources/MereRunCore/MuScriptor/MuScriptorTranscriber.swift`

## Reading order

The ACEStep runtime now follows a clean phase split:

1. `ACEStepPipeline.swift` for the public pipeline and orchestration
2. `ACEStepPipeline+Prompting.swift` for prompt preparation and conditioning
3. `ACEStepPipeline+Generation.swift` for the generation path itself

See [ACE-Step validation](./acestep-validation.md) for immutable checkpoint
pins, parity coverage, installed-model evidence, listening review fixtures,
and measured performance.

That makes it much easier to follow than a single pipeline monolith.

Magenta RT2 is a native Apple Silicon macOS runtime. The managed model layout
contains exported `.mlxfn` models, matching state files, and shared MusicCoCa and
SpectroStream resources; raw upstream checkpoint files are not enough for
`mere.run`.

`--style-conditioning streaming` matches upstream's realtime C++ path by using
the coarsest MusicCoCa style tokens. `--style-conditioning full` uses all style
tokens, matching upstream's high-level Python `.mlxfn` generator more closely.

## Contributor notes

- this is a native Swift/MLX path, not a Python bridge
- Magenta RT2 uses a pinned C ABI bridge built by
  `scripts/rebuild_magentart_xcframework.sh`; Linux builds keep compiling with
  an unsupported-runtime error for Magenta
- model resolution and storage still follow the same canonical public rules as
  the rest of the repo
