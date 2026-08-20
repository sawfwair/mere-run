# VFX geometry Apple Silicon benchmarks

These measurements exercise the native Swift/MLX runtime and the audited
managed checkpoints. They are implementation evidence, not cross-device
performance promises.

Use this report to compare the measured native geometry paths and understand
their validation boundaries.

## Host

- Apple M4 Max, 16 CPU cores, 128 GB unified memory
- macOS 26.4 (25E246)
- Debug build from `codex/vfx-primitives`, 2026-07-12
- Timed with `/usr/bin/time -lp`; `peak memory footprint` includes Metal/MLX
  unified-memory allocations that process RSS does not fully represent.

## MoGe-2 ViT-S Normal

A 128 x 128 RGB PNG was processed at the production 3,600-token setting through
the managed `vision-geometry-moge2-small` checkpoint. Each row represents a
fresh CLI process. The measurements include strict ONNX digest verification,
native MLX loading, metric camera recovery, and all artifact exports.

| Input | Tokens | Valid points | Model load | Inference | Postprocess | Wall | Max RSS | Peak footprint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 x 128 | 3,600 | 13,186 | 0.179 s | 0.397 s | 0.016 s | 0.88 s | 748.9 MiB | 8.50 GiB |

The run emitted metric depth and confidence EXRs, normal EXR, preview PNGs,
validity, camera JSON, and a colored PLY. Its schema-2 manifest binds the exact
27,898-byte input SHA-256 to the pinned 140,852,051-byte ONNX digest and every
output artifact hash. The large gap between process RSS and peak footprint is
the MLX/Metal unified-memory allocation; 3,600 tokens should therefore be
treated as a high-quality production setting, not a small-memory default.

## Video Depth Anything Small

The test used a synthetic 128 x 128 H.264 source at 8 frames per second. It
decoded and exported the source at its original resolution. The test used the
managed relative checkpoint through
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

## Depth Anything 3 Small

Two 128 x 128 PNG views were processed through the managed
`vision-geometry-da3-small` checkpoint. Each CLI invocation was a fresh
process; later rows benefited from filesystem and Metal cache warming. Peak
footprint includes the MLX/Metal allocations that are not visible in RSS.

| Camera mode | Process resolution | Exported points | Preprocess | Model load | Inference | Wall | Max RSS | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Predicted | 56 | 3,764 | 0.099 s | 0.267 s | 0.288 s | 2.65 s | 610.8 MiB | 634.8 MiB |
| Supplied / pose-conditioned | 56 | 3,764 | 0.055 s | 0.162 s | 0.056 s | 0.54 s | 609.9 MiB | 635.1 MiB |
| Predicted | 504 | 76,208 | 0.639 s | 0.148 s | 0.506 s | 4.52 s | 619.2 MiB | 2.83 GiB |

The 504 run emitted two depth EXRs, two confidence EXRs, review and processed
PNGs, predicted camera JSON, a 76,208-point colored binary PLY, a glTF 2.0 GLB
point cloud, and a Nerfstudio/3DGS initialization handoff. The handoff is
explicitly contains only camera data and colored points; it contains no learned
Gaussian parameters and no triangle mesh. A separate pose-conditioned run preserved the
supplied cameras, used the checkpoint's camera encoder, and recorded the
pairwise camera-baseline scale divisor in both structured output and the durable
scene manifest.

## TripoSR

The deterministic 512x512 parity image was reconstructed through the managed
`image-3d-triposr` checkpoint. These rows use the exact pinned upstream CKPT,
full native MLX scene encoding, learned vertex colors, and the native marching
tetrahedra exporter. Each invocation was a fresh process.

| Density grid | Vertices | Triangles | Checkpoint verify | Model load | Scene encode | Mesh/color | Export | Wall | Max RSS | Peak footprint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1,238 | 2,444 | 0.772 s | 0.971 s | 0.483 s | 0.069 s | 0.025 s | 3.83 s | 6.42 GiB | 5.22 GiB |
| 32 | 5,742 | 11,346 | 2.682 s | 1.314 s | 0.435 s | 0.160 s | 0.114 s | 7.34 s | 6.42 GiB | 5.37 GiB |

The runtime verifies the whole 1.67 GB checkpoint SHA-256 before parsing it.
Because that exact content digest supersedes ZIP entry CRCs, the pinned loader
does not rescan every tensor byte in a pure-Swift CRC loop; managed cold model
load fell from 153.3 seconds to about 1.0 to 1.3 seconds without weakening the
restricted non-executing state-dict grammar. Arbitrary checkpoint readers keep
entry CRC verification enabled by default.

At grid 32, the managed CKPT and deterministic converted safetensors package
produced byte-identical OBJ, PLY, and GLB outputs. A repeated converted-package
run also reproduced all three hashes. The durable TripoSR run manifest records
checkpoint format/source pins, foreground processing, density controls,
topology algorithm, mesh summary, and the hash of each mesh plus the shared
mesh manifest. Extraction cost grows cubically with grid resolution; the CLI's
default 256 grid is a quality setting, not a claim that the smaller benchmark
rows match production topology density.

## InstantMesh Base reconstruction

A user-controlled Suzanne object was rendered into six licensed 320x320 views
at the released conditioning rig: azimuth/elevation `30/+20`, `90/-10`,
`150/+20`, `210/-10`, `270/+20`, and `330/-10` degrees. The managed
`image-3d-instantmesh-base` id resolved the exact verified converted
safetensors package from its `native` child. Every invocation was a fresh
native Swift/MLX process with learned vertex colors enabled.

| SDF grid | Views | Vertices | Triangles | Checkpoint verify | Model load | Scene encode | Mesh/color | Export/manifests | Wall | Max RSS | Peak footprint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 6 | 1,586 | 3,164 | 0.528 s | 0.550 s | 0.716 s | 0.049 s | 0.034 s | 2.42 s | 3.62 GiB | 3.97 GiB |
| 24 | 6 | 3,746 | 7,488 | 0.494 s | 0.529 s | 0.499 s | 0.068 s | 0.067 s | 2.16 s | 3.62 GiB | 4.03 GiB |

The upstream 1,253,574,354-byte Lightning checkpoint was pulled and verified
first. A managed run against that raw-only root failed closed with the explicit
offline-conversion requirement; runtime never interpreted Pickle. After the
audited 1,253,463,832-byte safetensors package was installed under `native`,
the same managed model id completed both CLI and multipart API inference.

Two independent grid-24 runs produced identical OBJ, PLY, and GLB hashes:
`47e94a1f...`, `0239415b...`, and `583e819d...`. The authoritative run
manifest separately hashes the shared mesh manifest and durably records view
order/sizes, camera mode, source and converted pins, extraction controls,
Zero123++, runtime Python, and proprietary FlexiCubes exclusions, and the native
marching-tetrahedra topology caveat. Learned field parity is covered by the
reference fixture; triangle topology is not claimed to match upstream
FlexiCubes.
