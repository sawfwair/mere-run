# Music Runtime

This page covers the native music-generation paths exposed through
`mere.run music generate` and `mere.run music realtime`.

## Public surface

- `mere.run music generate`
- `mere.run music realtime`

## Model family

- `music-acestep`
- `music-magenta-rt2-small`
- `music-magenta-rt2-base`

## Typical workflow

```bash
swift run mere.run music generate \
  "upbeat electronic groove" \
  --output ./track.wav

swift run mere.run music realtime \
  "ambient modular synths with brushed drums" \
  --model music-magenta-rt2-small \
  --duration 4 \
  --output ./live.wav \
  --no-play
```

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

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/MusicGenerateCommand.swift`
- `Sources/MereRunCLI/Commands/MusicRealtimeCommand.swift`

### Runtime

- `Sources/MereRunCore/ACEStep/ACEStepPipeline.swift`
- `Sources/MereRunCore/ACEStep/ACEStepPipeline+Prompting.swift`
- `Sources/MereRunCore/ACEStep/ACEStepPipeline+Generation.swift`
- `Sources/MereRunCore/MagentaRT2/MagentaRT2Resources.swift`
- `Sources/MereRunCore/MagentaRT2/MagentaRT2Renderer.swift`
- `Sources/MereRunCore/MagentaRT2/MagentaRT2RealtimeSession.swift`

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
