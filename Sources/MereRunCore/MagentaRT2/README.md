# Magenta RT2

Native Magenta RealTime 2 support for Apple Silicon macOS.

## Files

- `MagentaRT2Resources.swift` resolves managed or local Magenta RT2 model roots,
  validates exported `.mlxfn` assets, and defines generation controls.
- `MagentaRT2Renderer.swift` owns the C ABI bridge to `magentart.xcframework`
  and renders deterministic frame batches.
- `MagentaRT2RealtimeSession.swift` adapts the renderer into a paced realtime
  frame stream for CLI playback, capture, and live steering.

## Runtime Contract

The runtime expects an exported Magenta RT2 layout with `models/mrt2_small` or
`models/mrt2_base`, plus shared `resources/musiccoca` and
`resources/spectrostream` assets. Raw upstream checkpoints are not enough.

Swift owns CLI stdout and stderr. Native C++ stdout is suppressed inside the
renderer so command stdout can stay machine-readable.

## Style Conditioning

`MagentaRT2StyleConditioning.streaming` keeps the upstream realtime C++ policy
of using the coarsest MusicCoCa style tokens. `full` uses all MusicCoCa style
tokens, matching the upstream high-level Python `.mlxfn` generator.
