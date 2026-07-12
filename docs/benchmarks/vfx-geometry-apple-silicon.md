# VFX geometry Apple Silicon benchmarks

These measurements exercise the native Swift/MLX runtime and the audited
managed checkpoints. They are implementation evidence, not cross-device
performance promises.

## Host

- Apple M4 Max, 16 CPU cores, 128 GB unified memory
- macOS 26.4 (25E246)
- Debug build from `codex/vfx-primitives`, 2026-07-11
- Timed with `/usr/bin/time -lp`; `peak memory footprint` includes Metal/MLX
  unified-memory allocations that process RSS does not fully represent.

## Video Depth Anything Small

Synthetic 128x128 H.264 source at 8 fps, decoded and exported at source
resolution. The managed relative checkpoint was used through
`vision-depth-vda-small`.

| Frames | Network input | Windows | Model load | Inference | Export/review | Wall | Max RSS | Peak footprint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 56 | 2 | 10.736 s | 0.527 s | 0.717 s | 13.53 s | 538.5 MiB | 683.0 MiB |
| 100 | 56 | 5 | 10.581 s | 1.447 s | 2.474 s | 16.91 s | 540.2 MiB | 690.2 MiB |
| 24 | 518 | 2 | 10.639 s | 1.891 s | 0.754 s | 14.68 s | 538.8 MiB | 8.66 GiB |

The 24-to-100-frame run increases peak footprint by about 7 MiB at the same
network size, demonstrating that finalized depth frames no longer accumulate
in memory. The fixed 32-frame temporal graph dominates inference memory; the
518 run is therefore the relevant production-size memory warning. The CLI is a
one-shot process and pays strict `.pth` mapping/loading on every invocation;
the actor-backed generator retains the verified model across calls until
explicitly unloaded.

Every run produced one EXR and normalized PNG per source frame, a frame-accurate
manifest, and a source-FPS review MP4. The pre-streaming and streaming 24-frame
depth/preview artifact hashes were byte-identical.
