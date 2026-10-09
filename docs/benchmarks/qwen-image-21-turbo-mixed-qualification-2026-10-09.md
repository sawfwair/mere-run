# Qwen Image 2.1 Turbo mixed-precision qualification

Built with Qwen. This report covers a mixed-precision conversion of the official
Qwen Image 2.1 Turbo checkpoint, not the base Qwen Image 2.1 model or a Turbo LoRA.
The original Qwen Research License applies.

Conversion, artifact verification and public publication are complete. All 25
bundle files were verified by size and SHA-256 at immutable revision
`d8276b786f94a4e8944189cc181e839e31759ab5`, including anonymous access after
publication. Limited trained-checkpoint execution passes the five cases below.
The report and receipt include bounded runtime observations.
The [machine-readable receipt](/qualification/qwen-image-21-turbo-mixed-2026-10-09.json)
records weight hashes, header-audit results, native binary identity and these gates.

## Provenance and format

- Source: `Qwen/Qwen-Image-2.1-Turbo` at
  `d65dbc9a7e8f6b5479e33dee6030eaab2a906509`.
- Artifact: [Sawfwair/Qwen-Image-2.1-Turbo-MLX-Mixed-4bit](https://huggingface.co/Sawfwair/Qwen-Image-2.1-Turbo-MLX-Mixed-4bit).
- Conversion: `scripts/model-conversion/convert_qwen_image21_turbo_mlx.py`,
  MLX 0.32.2 CUDA, cuDNN 9.12.0.46; conversion ran on an ephemeral RunPod GPU.
- Transformer block linear weights: MLX affine Q4/group-64.
- Encoder linear weights: MLX affine Q8/group-64.
- Embeddings, VAE, normalization, transformer input/output, timestep and
  modulation layers: original BF16. The unused language-model output head is
  omitted because native conditioning returns hidden states.
- Original eight-step sigma grid and CFG 1 defaults are preserved.

The conversion records source file hashes and per-layer weight reconstruction
errors in `QWEN21_CONVERSION.json`. Weight reconstruction does not measure image
quality. The bundle includes the upstream license, required Notice, upstream
model card, modification notices, executed converter and SHA-256 checksums.

The final redistribution bundle is 14,252,544,800 bytes (14.25 GB). All transferred
files matched the conversion host's SHA-256 checksums before packaging corrections.
The corrected package records those metadata changes without changing weight
payloads. The header audit passed against the original pinned tensor schemas:
341 Q8 encoder layers, 224 Q4 transformer layers, and 238 unchanged VAE tensors.
Tensor closure, packed dtype/shapes and shard ownership were checked separately
from checksum integrity.

The temporary RunPod pod and its attached temporary disk were terminated after
local preservation; the provider inventory confirmed the pod absent. Its elapsed
compute estimate was about US$0.058, excluding storage and provider billing rounding.

Inference uses native Swift/MLX. Packed weights stay UInt32 and execute with
quantized matrix multiplication; no full-precision model expansion or Python
inference sidecar is needed. Components load and release in stages.

## Local code verification

The Qwen-focused tests passed: 22 tests, two opt-in skips, zero failures. The
packed Q4/Q8 transformer path matches explicitly dequantized weights on common
inputs and rejects malformed packed weights, scales, biases and unsupported
packing. The loader test verifies preservation of UInt32 values through disk
loading. These checks use small fixtures rather than trained-checkpoint images.

The repository gate passed 5,363 XCTest cases (440 opt-in skips) plus 192 Swift
Testing cases, strict lint, CLI help and hygiene checks. The documentation site
also built successfully. Hosted CI and a released application are separate gates.

## Trained-checkpoint local execution

The native Swift/MLX CLI completed all five cases. Each used the public
checkpoint revision above,
eight steps and CFG 1. Earlier admission checks were blocked while other workloads
held memory; the successful runs passed the standard memory-admission guard.

| Case | Seconds | Peak process footprint (GiB) |
| --- | ---: | ---: |
| smoke-512 | 21.5 | 9.11 |
| replay-512 | 18.7 | 9.11 |
| quality-1024 | 81.5 | 20.63 |
| text-1024 | 86.6 | 20.60 |
| edit-512 | 44.4 | 10.42 |

The 512-pixel replay produced identical pixel and PNG SHA-256 hashes. Direct visual
inspection found coherent blue teapots, a legible MERE poster heading, and a
successful blue-to-red reference edit retaining overall shape, composition, table
and lighting with small texture changes. All outputs are correctly sized RGBA PNGs.

The highest observed OS process footprint was 20.63 GiB. This includes GPU
allocations and caches; it is not whole-machine usage or a minimum-memory
guarantee. These cases demonstrate 512/1024 generation and one-reference editing
with sufficient admission headroom. They do not qualify
2048-pixel generation, multiple references, transparent output, broad prompt
quality, or equivalence to BF16 Turbo. `quality_qualified` remains false.

The earlier full repository gate used binary SHA-256
`366f2ca6ec636ec81db1468cb6f1824ceb8ba2d343fc80f90db6eb9496f8c88e`.
These local checkpoint cases all used the later unreleased binary
`59ce312e3041652484a41661185055e791746190a2478cf152e36bb475e108e1`.
The source base is unchanged and the checkout contains uncommitted work; neither
set of results is a released-app or hosted-CI claim. Raw case receipts, logs and
progress events are retained locally; the public-site receipt records case
prompts, seeds, checkpoint/binary hashes, image hashes and memory/timing results.

### Inspected samples

![1024-pixel blue teapot](/qualification/qwen-image-21-turbo-mixed-2026-10-09/quality-1024.png)

![Legible MERE poster](/qualification/qwen-image-21-turbo-mixed-2026-10-09/text-1024.png)

![Blue teapot reference](/qualification/qwen-image-21-turbo-mixed-2026-10-09/smoke-512.png)

![Red teapot edit](/qualification/qwen-image-21-turbo-mixed-2026-10-09/edit-512.png)
