# Music Runtime

This page covers the native music-generation paths exposed through
`mere.run music analyze`, `mere.run music generate`, `mere.run music realtime`,
and `mere.run music transcribe`.

## Public surface

- `mere.run music generate`
- `mere.run music analyze`
- `mere.run music realtime`
- `mere.run music transcribe`

## Model family

- `music-acestep`
- `music-acestep-xl-turbo`
- `music-acestep-xl-turbo-lm4b`
- `music-magenta-rt2-small`
- `music-magenta-rt2-base`
- `music-muscriptor-small`
- `music-muscriptor-medium`
- `music-muscriptor-large`

## Guides

Music guidance follows the command/cookbook shape used by the rest of
`mere.run`: choose the command first, then focus the guide with a model id.

```bash
mere.run guide music generate --model music-acestep
mere.run guide music generate --model music-acestep-xl-turbo
mere.run guide music analyze --model music-acestep-xl-turbo-lm4b
mere.run guide music generate --model music-magenta-rt2-small
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

swift run mere.run model pull music-muscriptor-medium
swift run mere.run music transcribe ./song.mp3 --output ./song.mid
```

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
preserves the lowest-memory single-chunk path.

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
search packs all live beams across the effective chunk group into one forward
per step, keeps an independently forked typed cache lane for every beam, and
removes ended beams from later forwards.

ACE-Step generation uses the upstream CLI turbo shift default (`--shift 3.0`)
and the native Haar DCW sampler correction (`double`, low `0.05`, high `0.02`)
before VAE decode. The XL turbo managed ID installs the 4B DiT decoder plus the
base ACE-Step VAE and Qwen3 text encoder; the `-lm4b` variant also installs the
optional 4B 5 Hz LM for `--use-lm` runs. ACE-Step cover/repaint/extract tasks
follow upstream and skip the 5 Hz LM phase so source-audio conditioning stays
faithful.

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
- `Sources/MereRunCLI/Commands/MusicRealtimeCommand.swift`
- `Sources/MereRunCLI/Commands/MusicTranscribeCommand.swift`

### Runtime

- `Sources/MereRunCore/ACEStep/ACEStepPipeline.swift`
- `Sources/MereRunCore/ACEStep/ACEStepPipeline+Prompting.swift`
- `Sources/MereRunCore/ACEStep/ACEStepPipeline+Generation.swift`
- `Sources/MereRunCore/MagentaRT2/MagentaRT2Resources.swift`
- `Sources/MereRunCore/MagentaRT2/MagentaRT2Renderer.swift`
- `Sources/MereRunCore/MagentaRT2/MagentaRT2RealtimeSession.swift`
- `Sources/MereRunCore/MuScriptor/MuScriptorTranscriber.swift`

## Reading order

The ACEStep runtime now follows a clean phase split:

1. `ACEStepPipeline.swift` for the public pipeline and orchestration
2. `ACEStepPipeline+Prompting.swift` for prompt preparation and conditioning
3. `ACEStepPipeline+Generation.swift` for the generation path itself

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
