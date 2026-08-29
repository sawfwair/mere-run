# MiniMax-H3 h3.c transfer program

This document defines the evidence contract for transferring useful h3.c
techniques into `mere.run`. It is not a performance receipt. Production behavior
does not change until a candidate passes its tensor, checkpoint, quality,
memory, and fallback gates.

## Promotion conclusion

The original scalar-SIMD affine-Q8 matrix cores remain rejected: they are
roughly four to five times slower than stock MLX at H3's production shapes.
The FastH3 Q8 MLP now uses a separate matrix-tiled Metal schedule. K4a stages a
32-row by 32-output tile and keeps both FC1 projections in registers through
the SwiGLU epilogue; K4b stages a 32-row by 64-output tile. Both dequantize each
Q8/group-64 weight tile once into threadgroup memory and accumulate BF16 matrix
operands in FP32.

The tiled K4 path is the default only for the managed FastH3 VSA Q8 recipe.
Other H3 variants retain stock MLX unless `MERERUN_H3_EXACT_KERNELS=affine-q8-mlp`
is selected explicitly. `MERERUN_H3_EXACT_KERNELS=disabled` opts the FastH3
recipe out. K2b and K3 remain experimental because their scalar matrix cores
did not win.

At 89,188 rows on the packaged FastH3 checkpoint, the correct chunked portable
oracle measured 3,813.785 ms for FC1 plus SwiGLU and 2,176.992 ms for FC2. The
tiled kernels measured 3,445.900 ms and 2,073.032 ms respectively: 1.107x and
1.050x. FC1 relative L2 was 0.000767212, with a worst 32,768-row chunk of
0.000767620; FC2 was bit-identical. K4a also avoids the 5.11 GiB
`[1, 89188, 28672]` FC1 projection. The unchunked stock path wraps that
intermediate after its 4 GiB boundary, so high-resolution validation must use
the chunked oracle rather than treating the corrupt tail as a baseline.

Reproduce the installed-checkpoint stage gate with:

```bash
MERERUN_H3_EXACT_KERNEL_MODEL_ROOT=/path/to/video-minimax-h3-fasth3-vsa-datafree-mlx \
  scripts/h3-kernel-lab.sh affine-mlp-real
```

Resident BF16 has a separate M3/M4 opportunity. The refreshed MLX source has
a Metal 4 NAX GEMM implementation, but its runtime capability gate
requires Apple GPU generation 18 or newer, so it does not dispatch on the local
generation-16 M4 Max. The H3 lab now includes an independently gated MPP BF16
projection candidate, adapted from WeeTodd's measured MiniMax-H3 tiles. Both
tile configurations pass the deterministic small-shape M4 Max bit-exact canary.
Production-shape timing remains unqualified because the first attempt correctly
stopped at the clean-host guard while unrelated builds and ML workloads were
active. The candidate is therefore not wired into model dispatch. Run the
isolated release arm only on a clean host:

```bash
scripts/h3-kernel-lab.sh mpp-projections
```

This round deliberately targets the resident-BF16 M3/M4 path; M5-specific
TensorOps work is out of scope.

The quality-sensitive algorithm arms remain non-default. Reduced canvas,
layer thinning, complete velocity reuse, and token reduction all produced
material same-seed trajectory changes. The three-seed Ref2VA follow-up closed
the strongest candidate: interval-2 velocity reuse saved 31.53% to 36.04% and
preserved audio closely, but changed the visual trajectory in every seed and
introduced a conspicuous train-light artifact in one. It remains an explicit
research arm and is not a production default.

The larger production win came from serving parity rather than approximate
denoising. Preserving each reference image's aspect ratio, refusing to upscale
it, and area-matching it to the render canvas reduced this fixture's packed
sequence and MLX peak without changing the dense-quality algorithm. The final
zero-swap retake completed in 578.450 seconds versus the old clean
1,920.079-second path and reduced MLX peak from 88.59 to 41.14 GiB. The output
was byte-identical to the same-seed post-alignment screen. This is the
production result: 69.87% lower cold-host wall time and 53.56% lower MLX peak
without skipping an evaluation or selecting an approximation policy.

## Fixed sources

- mere.run base: Ref2VA 8-bit commit
  `184932ae5e2788812376285a39c606c6a568ebd3`, stacked on `origin/main`.
- external h3.c oracle repository: `https://github.com/antirez/h3.c.git`.
- h3.c revision: `03cb1339825feb19bcafcc60685680cb9ec6e2fe`.
- h3.c source license: MIT. Its dynamic symmetric int8 design is included in
  the BSD-3-Clause notice identified in upstream `THIRD_PARTY_NOTICES.md`;
  mere.run reproduces both notices for the selected source-derived behavior and
  activation-quantization design. No Morton decoder or TensorOps scheduler was
  ported.
- MiniMax-H3 weights remain governed by the MiniMax-H3 Community License.

Run the source oracle without adding it to the repository:

```bash
scripts/h3c-oracle.sh test
scripts/h3c-oracle.sh pin
scripts/h3c-oracle.sh run --help
```

No h3.c source code, binary, or library is vendored, linked, or shipped by
mere.run. The optional runner fetches an ignored checkout under
`.build/h3c-oracle` by default and refuses revision drift by checking out the
immutable h3.c commit listed in this section on every build. It does not download
checkpoint weights.
h3.c expects the original upstream `FL2VA/` and `Ref2VA/` directory trees;
mere.run consumes its own flat managed MLX artifacts, so output comparisons
must record which weight representation each arm used.

For a local BF16 parity run, `MiniMaxH3Resources` also accepts ignored flat
overlays with the released shard directories at `transformer-bf16/` and
`text-encoder-bf16/`. The remaining flat config, tokenizer, VideoVAE, and
AudioVAE files may be symlinked from the matching managed package. This permits
both engines to read the same verified official BF16 transformer and
conditioner without converting or duplicating them. Validate such an overlay
without loading weights:

```bash
MERERUN_H3_BF16_OVERLAY_ROOT=/absolute/path/to/overlay \
  swift test --filter \
  MiniMaxH3Tests/testInstalledShardedBF16Ref2VAOverlayWhenAvailable
```

## Two independent lanes

### Kernel lane

Each kernel candidate must first reproduce an explicit decomposed oracle. If a
candidate deliberately changes arithmetic, such as dynamic INT8 activation
quantization, it also needs a same-seed checkpoint quality comparison and may not be
described as trajectory-exact.

| ID | Candidate | First proving shape | Required fallback |
| --- | --- | --- | --- |
| K1 | Attention residual gate + following MLP AdaLN, then optional activation quantization | BF16 `[1, rows, 5376]` | Existing decomposed MLX graph |
| K2 | QKV projection directly to `[1, 56, rows, 128]`, with fused Q/K RMSNorm and RoPE | H3 QKV `5376 -> 21504` | Existing linear, split, transpose, norm, and RoPE path |
| K3 | Head-major SDPA output directly into the INT8 output projection | H3 attention output `7168 -> 5376` | Existing transpose, reshape, and linear path |
| K4 | FC1, SwiGLU, and FC2 with H3-specialized `5376 -> 28672 -> 14336 -> 5376` dimensions | Complete production MLP | Existing MLX compiled MLP |
| K5 | Activation lifetime aliases for QKV, attention, and MLP arenas | Complete block | Independent MLX arrays |

K1 has BF16 and mixed BF16/Float32 custom-Metal canaries in
`MiniMaxH3FusedKernels.swift`. Each keeps the gated residual in threadgroup
memory while computing the next RMSNorm and AdaLN. The mixed path reproduces
MLX's BF16 rounding of `1 + scale`; omitting that single boundary caused a
large 50-block drift even though the residual itself matched. The INT8
candidate continues directly
through h3.c-compatible per-row symmetric activation quantization:
`scale = max(abs(row)) / 127`, with a finite `1 / 127` scale for an all-zero
row. A standalone quantizer provides the two-kernel oracle and release timing
arm. Both paths are byte-exact at the INT8 boundary in the deterministic GPU
canary. The floating K1 path participates in the opt-in installed-checkpoint
dispatch described in the installed-checkpoint section. The activation-INT8
continuation remains lab-only.

The managed H3 artifacts use affine Q8/group-64 *weights*. Stock MLX
`QuantizedLinear` still accepts floating activations and does not expose a
prequantized-activation input. Consuming K1's INT8 rows without re-expanding
them therefore requires a matching custom projection kernel; producing the
bytes alone is not described as an end-to-end acceleration.

Run the isolated K1 release arms with no other ML workload active:

```bash
scripts/h3-kernel-lab.sh gate-adaln
scripts/h3-kernel-lab.sh gate-adaln-int8
```

Every kernel-lab mode applies the same clean-host boundary as the generation
harness: no matching ML process, `mere.run` process, Swift compiler, or Xcode
build, and no more than 1024 MiB of starting swap. Override the ceiling only as
an explicit evidence-policy change with
`MERERUN_H3_LAB_MAX_STARTING_SWAP_MIB`; the selected ceiling and matched
processes are retained in `.build/h3-kernel-lab/start-gate.txt` before any
release build or benchmark begins.

Run every exact candidate in evidence order with the resumable suite wrapper:

```bash
scripts/h3-kernel-suite.sh --output .build/h3-kernel-suite/ref2va
scripts/h3-kernel-suite.sh --output .build/h3-kernel-suite/ref2va --resume
```

The default queue covers K1 floating and activation-INT8 boundaries, K2a/K2b,
K3, K4, K5, and the installed Ref2VA full-forward gate. Each attempt gets its
own stdout, stderr, and start-gate receipt, so resuming does not overwrite failed
evidence. A clean-host rejection stops the suite immediately with exit 75;
other failures are recorded while remaining modes continue. Successful modes
leave commit-bound pass markers, so `--resume` skips only proven passes. The
suite requires a clean worktree and binds resume to its original commit and
ordered mode set. An atomic repository-local lock prevents two suites from
producing overlapping timing evidence; an exited lock owner is recovered only
after its recorded PID no longer exists.

K2 through K4 need two implementations where hardware requires it:

- A portable MLX and Metal fallback for supported M-series Macs.
- An M5-gated Metal 4 or TensorOps implementation that is not selected by device-name
  assumptions alone if capability probing is available.

K2 is split into two proof stages so layout correctness is not confounded with
projection arithmetic:

- K2a consumes the existing BF16 global-slab projection
  `[1, rows, 3 * 56 * 128]` and performs the split, per-head Q/K RMSNorm,
  96-dimension RoPE, and direct write to MLX SDPA's
  `[1, 56, rows, 128]` head-major contract in one Metal kernel.
- K2b applies the managed affine Q8/group-64 QKV weights in a custom projection
  that writes three raw head-major tensors, then runs one fused Q/K RMSNorm and
  RoPE kernel. It eliminates the global-slab projection tensor and lets V flow
  directly into SDPA. Because standalone MLXFast outputs cannot donate storage,
  raw head-major Q/K remain explicit until the normalization kernel finishes;
  the path is projection-direct but not allocation-free.

The K2a deterministic GPU canary compares all three outputs with the existing
decomposed graph; V is byte-exact and Q/K remain inside the declared BF16
tolerance. Its isolated release arm is:

```bash
scripts/h3-kernel-lab.sh qkv-layout
scripts/h3-kernel-lab.sh qkv-direct
```

The `qkv-direct` receipt compares the complete portable affine-Q8 projection,
split, norm, RoPE, transpose, and contiguous path against K2b. It reports the
global projection bytes avoided along with Q/K/V maximum absolute differences.
The installed-checkpoint path uses direct head-major projection for every
block. Its first BF16 block retains MLXFast's exact RMSNorm/RoPE graph after
that projection because the experimental BF16 fused reduction missed the
quality envelope; the subsequent Float32 blocks use the fused norm/RoPE
kernel. This hybrid preserves the quality receipt without pretending the BF16
reduction has been solved. Production-row release timing remains outstanding.

K3 now has a first exact-artifact-contract candidate. It reads MLX SDPA's
contiguous `[1, 56, rows, 128]` BF16 or Float32 output directly and applies the managed
checkpoint's existing affine Q8/group-64 output weight (`uint32` packed codes,
BF16 scale and bias per 64 input columns). The kernel writes
`[1, rows, 5376]`, eliminating the head-major transpose/reshape materialization.
This is weight INT8 with floating activations, matching the managed artifact;
it is not h3.c's separate symmetric activation-INT8 arithmetic. The deterministic GPU
canary compares it with MLX `quantizedMM` using the same packed arrays. Its
isolated release arm is:

```bash
scripts/h3-kernel-lab.sh affine-oproj
```

K4 also has an exact-artifact-contract pair for the managed Q8/group-64
transformer. K4a performs the `5376 -> 28672` FC1 projection and folds the
immediate SwiGLU into the same dispatch, writing only the compact
`[1, rows, 14336]` activation. It therefore removes the 28,672-wide FC1 tensor.
K4b applies the exact `14336 -> 5376` FC2 projection to that compact result.
Both candidates retain the graph's BF16 or Float32 activation boundary and
compare independently with MLX `quantizedMM`; they do not yet consume K1's symmetric activation-INT8
rows. The whole-path isolated release arm alternates the decomposed and fused
orders and reports the FC1 materialization bytes avoided:

```bash
scripts/h3-kernel-lab.sh affine-ffn
```

K5 distinguishes fusion from true buffer aliasing. Standalone MLXFast custom
Metal kernels always allocate their declared outputs, so K2 through K4 remove
large transpose and projection intermediates but do not donate an input buffer.
MLX compiled primitives can instead donate an input allocation when its output
has the same shape and dtype, the input is contiguous, and no caller retains
it. The K5 canary proves both sides of that contract at H3's BF16 residual
boundary: an unretained `[1, rows, 5376]` input is reused, while deliberately
retaining the same input forces another tensor-sized allocation. Ineligible
inputs automatically preserve the ordinary allocating path; there is no unsafe
manual alias. The production-row memory arm reports both peak increments:

```bash
scripts/h3-kernel-lab.sh buffer-alias
```

### Installed-checkpoint exact-kernel dispatch

`MERERUN_H3_EXACT_KERNELS=boundary-layout` enables only the exact candidates
that won their clean production-shape microbenchmarks: K1 gated AdaLN and K2a
head-major QKV layout with fused Q/K normalization and RoPE. It retains MLX's
quantized projections instead of selecting the slower custom affine-Q8 GEMMs.
The mode remains explicit, requires `--h3-acceleration quality`, and falls back
per call when the exact H3 shape or dtype contract is unavailable. Sequences at
or below 12,000 rows retain whole-step compilation. Larger sequences use eager
layer evaluation because the clean Ref2VA comparison measured a larger gain and
lower peak footprint than embedding these custom boundaries in independently
compiled blocks.

`MERERUN_H3_EXACT_KERNELS=affine-q8` enables the exact kernel candidates inside
the real transformer loop for controlled checkpoint comparisons. Admission is typed
and fail-closed: the request must use `--h3-acceleration quality`, every main
transformer block must retain its unadapted affine Q8/group-64 QKV, output, FC1,
and FC2 linears, and resident-BF16 materialization must be off. The mode forces
eager block execution so custom-kernel behavior and memory remain attributable.

Within an admitted block, K2b handles QKV projection plus Q/K norm/RoPE, K3
handles the head-major attention output projection, K1 fuses the attention
residual into the following feed-forward AdaLN, and K4 handles both feed-forward
projections. Every custom call still validates its complete shape, dtype, and
quantization contract and falls back to the decomposed graph if that individual
contract is unavailable. K1's activation-INT8 output remains a lab boundary:
the installed path uses its BF16 sibling until a projection consumes the
dynamic INT8 rows directly.

`MERERUN_H3_EXACT_KERNELS=affine-q8-mlp` selects only the matrix-tiled K4a and
K4b kernels. It does not select K1, K2, or K3. The managed FastH3 VSA Q8 recipe
uses this mode automatically when the environment variable is absent; setting
the variable to `disabled` retains the portable MLX MLP for diagnostics.

The gated real-weight test can also enable one stage at a time before the
combined run. On the installed Ref2VA artifact, all five stages selected in all
50 main blocks with no fallbacks. For the seven-row deterministic forward, the
combined candidate measured video relative L2 `0.000435404` and audio relative
L2 `0.000339896` against the decomposed graph. The individual receipts were:

| Stage | Video relative L2 | Audio relative L2 |
| --- | ---: | ---: |
| K1 gated AdaLN | 0.0000786547 | 0.0000577654 |
| K2b head-major QKV | 0.000313553 | 0.000195879 |
| K3 attention output | 0.000236497 | 0.000203095 |
| K4a FC1/SwiGLU | 0.000161476 | 0.000128909 |
| K4b FC2 | 0.000097037 | 0.0000835104 |

Reproduce the arithmetic gate with the installed artifact path:

```bash
MERERUN_TEST_MLX_DEVICE=gpu \
MERERUN_H3_EXACT_FULL_FORWARD=1 \
MERERUN_H3_EXACT_STAGE_DIAGNOSTICS=1 \
MERERUN_H3_EXACT_KERNEL_MODEL_ROOT="$HOME/Library/Application Support/MereRun/models/video-minimax-h3-ref2va-mlx" \
swift test --filter MiniMaxH3Tests/testInstalledRef2VAExactKernelFullForwardWhenEnabled
```

The full-generation harness always writes the selected exact-kernel mode into
`receipts.tsv`. Its normal quality and algorithm arms explicitly disable the
mode so an inherited shell environment cannot contaminate their baseline. Add
the exact arm only for the managed Q8 Ref2VA artifact:

```bash
MERERUN_H3_BAKEOFF_ARMS=quality,exact-affine-q8 \
  scripts/h3-algorithm-bakeoff.sh \
  video-minimax-h3-ref2va-mlx \
  .build/h3-bakeoff/ref2va-exact-kernels \
  "preserve the subject and motion" \
  --reference image:./subject.png \
  --reference video:./motion.mp4
```

This is an experimental evidence surface, not a production default. The
50-block real-weight arithmetic pass and fixed-seed generated-media review are
complete. The generated-media results justify the shape-aware execution policy,
but not ordinary dispatch: FL2VA has no measured speed win and Ref2VA has only
one fixed prompt/reference fixture so far.

### Clean M4 Max kernel result

The 2026-08-11 clean-host run used commit `62a2c123`, zero starting swap, no
competing build or ML process, and no thermal or performance warning. It
established a sharp split between useful fusion boundaries and custom
quantized GEMMs:

| Candidate | Portable/unfused | Custom/fused | Result |
| --- | ---: | ---: | ---: |
| K1 gated AdaLN | 7.098 ms | 2.245 ms | 3.161x |
| K1 AdaLN to activation INT8 | 2.764 ms | 1.577 ms | 1.753x |
| K2a QKV layout/norm/RoPE | 18.426 ms | 3.561 ms | 5.174x |
| K2b direct affine-Q8 QKV | 311.872 ms | 1,423.968 ms | 0.219x |
| K3 affine-Q8 output projection | 109.946 ms | 536.511 ms | 0.205x |
| K4 affine-Q8 FFN | 806.792 ms | 3,191.661 ms | 0.253x |

K5 buffer donation avoided a 160,841,728-byte retained peak increment. The
direct QKV and FFN arms also exceeded their synthetic absolute-error envelopes;
all affine stages nevertheless remained below `0.00044` combined relative L2
in the installed 50-block Ref2VA arithmetic gate. The speed regressions are
decisive: K2b/K3/K4 remain research prototypes until their GEMM core uses a
competitive MLX/Metal matrix path. `boundary-layout` is the only exact mode
advanced to generated-media evaluation from this run.

### Clean generated-media result

The fixed railway-platform fixture used a 512 x 256 canvas, 124 frames at 24 frames per second,
seed `20260810`, one pinned 110,364-byte image with SHA-256
`34ea0fee383e3b5d353f6a9556af12b5e7d3a7846c6899e768791b5354818ebd`,
and, for Ref2VA, one pinned 663,630-byte audio reference with SHA-256
`444afb780a0b1a8fe5b1bb90ac744669ef9086c721439c4a0d569389c2e1df80`.
FL2VA used the same image as its first-frame condition and the installed
`minimax-h3-lightx2v-4step` adapter. Every listed arm began with zero swap, a
clean worktree, no matched build/ML process, and no thermal or performance
warning; all arms also ended at zero swap.

| Model and boundary execution | Commit | Baseline | Boundary | Wall delta | Peak-footprint delta | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Ref2VA / eager | `e93bf2ab` | 1,920.079 s | 1,860.912 s | -59.167 s (-3.08%) | -1.344 GiB | Retain above 12,000 rows |
| FL2VA / eager | `e93bf2ab` | 174.096 s | 194.674 s | +20.578 s (+11.82%) | effectively unchanged | Reject as a universal policy |
| FL2VA / compiled | `dc14efb3` | 215.976 s | 217.275 s | +1.299 s (+0.60%) | +0.002 GiB | No demonstrated speed win |
| Ref2VA / blockwise compiled | `dc14efb3` | 1,911.015 s | 1,905.189 s | -5.826 s (-0.30%) | +1.891 GiB | Inferior to eager Ref2VA policy |

The compiled FL2VA session had unusually variable baseline steps (30.885 to
49.190 seconds) compared with the earlier stable baseline (30.400 to 33.721
seconds), despite the clean host receipts. It is evidence that compilation
removes the large eager penalty, not evidence of a 0.60% regression. Likewise,
the blockwise-compiled Ref2VA arm reduced steady MLX cache from 16.11 to 12.61
GiB but did not reduce the 88.59 GiB MLX peak and increased process peak
footprint, so the small wall-time delta is not a promotion candidate.

All four selected outputs passed the structural gate: H.264 video at 512 x 256,
124 frames, 24 frames per second, 5.167 seconds, plus stereo AAC at 32 kHz. Matched-seed
quality for the selected shape-aware paths was:

| Model and candidate | Video SSIM | PSNR | VMAF | Audio correlation | Audio relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ref2VA / eager boundary | 0.989612 | 45.954653 dB | 91.309876 | 0.999354733 | 0.0359229473 |
| FL2VA / compiled boundary | 0.988137 | 46.453891 dB | 91.366412 | 0.998332539 | 0.0577492307 |

The contact sheets preserve the two people, rain-soaked platform, train motion,
and first-frame composition. As a narrow reference-retention check, the first
decoded frame was compared with the same Lanczos-resized source image. Ref2VA
SSIM moved from 0.792608 baseline to 0.792859 boundary; FL2VA moved from
0.969759 to 0.969820. These pixel metrics and contact sheets do not replace a
larger blinded semantic/reference-retention review.

## Quality-sensitive algorithm lane

These modes are always explicit. They may compose only after their individual
quality envelopes have been measured.

| ID | Candidate | Initial comparison |
| --- | --- | --- |
| A1 | Same-aspect reduced internal render canvas followed by high-quality upscale | Native output, 75%, and 62.5% internal dimensions |
| A2 | AdaLN-gate-ranked layer thinning with protected first/final blocks | 50, 45, and 40 active blocks |
| A3 | Whole-velocity or transformer-core reuse | Adaptive tail reuse, interval-2 velocity reuse, and interval-4 core reuse |
| A4 | Target-video token pairing with full-resolution bypass and delta restoration | Blocks 4-40 early, then 4-30, versus full tokens |

The adaptive block-tail cache is not equivalent to h3.c whole-velocity
or core reuse. It remains a separate bake-off arm rather than being silently
relabelled.

A3 now has its first explicit runtime arm: `--h3-acceleration
velocity-reuse-2`. It runs the first and final denoise evaluations in full,
linearly extrapolates both complete video and audio velocity outputs from the
two most recent full evaluations on intervening odd steps, and keeps the quality
schedule point count. Selecting it disables the adaptive tail and dynamic-sparse
policies so its quality and timing deltas remain attributable. It is
experimental and non-default. Fixed FL2VA and three-seed Ref2VA comparisons found a
repeatable speed benefit but failed the visual-trajectory promotion gate.

A2 has two equally isolated arms: `--h3-acceleration layers-45` and
`--h3-acceleration layers-40`. They reproduce the pinned oracle's ranking:
mean absolute attention and MLP AdaLN gate values over every cached schedule
point and modality, with blocks 0, 1, and 49 always protected. The lowest
remaining scores are skipped while original block indices and weights remain
unchanged. The prototype reduces executed transformer work but retains
all loaded weights; it must not claim h3.c's additional residency reduction
until loading itself prunes the inactive blocks.

#### Ref2VA AdaLN cache gate

A2 requires the cache shipped with the model, not a machine-local preparation
step. The first Ref2VA cache candidate exposed two otherwise silent contract
violations. Projecting all 90 schedule/modality rows in one quantized matrix
multiplication was not numerically equivalent to live Ref2VA's three-row
projection, and the fallback cache identity described the local model-store
symlink rather than the immutable transformer.

The generator now evaluates one three-modality batch per released schedule
point. Managed Ref2VA binds the cache to transformer SHA-256
`234f22f69f8d40d6ed81cceed8259fa287f3c9417d40fba5274e3a7aa84e18a2`.
The corrected cache is 873,820,740 bytes with SHA-256
`2cbe9e3324ef2cc5108a3ba7f1219d84079ff00a017f604fd86300005cc64fcd`
and is published in the pinned artifact revision
`61dc387ef1a7166425cdacd63c2340598dcc364f`. At schedule step 10, the installed
file, a fresh in-memory cache, and direct live AdaLN produced zero maximum video
and audio output error. The 45- and 40-block rankings both retained protected
blocks 0, 1, and 49.

Reproduce the installed-artifact gate without writing the cache:

```bash
MERERUN_TEST_MLX_DEVICE=gpu \
MERERUN_H3_ADALN_CACHE_PARITY=1 \
MERERUN_H3_EXACT_KERNEL_MODEL_ROOT="$HOME/Library/Application Support/MereRun/models/video-minimax-h3-ref2va-mlx" \
swift test --filter MiniMaxH3Tests/testInstalledRef2VAAdaLNCacheMatchesLiveBranchWhenEnabled
```

A1 is exposed as paired `--h3-render-width` and `--h3-render-height` controls.
The output canvas remains `--width` by `--height`; only target conditioning,
DiT rows, and VAE decode use the smaller same-aspect grid. Decoded uint8 RGB is
upscaled with the pinned oracle's `vImageScale_ARGB8888` flags
`kvImageHighQualityResampling | kvImageEdgeExtend`. Both dimensions are
required, must remain on the 32px grid, and must not exceed output dimensions.
Continuation/sliding windows fail closed until their full-resolution history
is explicitly resampled into the internal conditioning grid.

A4 is available as `--h3-acceleration token-reduction`. It reproduces the
pinned default block schedule (`4..<40` for the first ten evaluations, then
`4..<30`) and pairs only adjacent horizontal target-video rows. Prefix and odd
trailing rows are exact. Reduced RoPE uses the pair's mean position. At restore,
each source token is its saved full-grid value plus the processed reduced token
minus its pooled baseline, with update scale 1. The mode retains the quality
schedule and disables the other approximation policies.

Run the fixed-seed algorithm matrix with one installed FL2VA or Ref2VA model.
The harness refuses another active ML workload, builds release once, emits one
preflight JSON and stderr log per arm, and records wall time plus output SHA-256:

```bash
scripts/h3-algorithm-bakeoff.sh \
  video-minimax-h3-fl2va-bf16-mlx \
  .build/h3-bakeoff/fl2va \
  "a fixed evaluation prompt" \
  --image ./subject.png \
  --h3-adapter minimax-h3-lightx2v-4step

scripts/h3-algorithm-bakeoff.sh \
  video-minimax-h3-ref2va-mlx \
  .build/h3-bakeoff/ref2va \
  "preserve the subject and motion" \
  --reference image:./subject.png \
  --reference video:./motion.mp4
```

Pin conditioning-input identity for a reportable run with a tab-separated
manifest. The harness hashes FL2VA `--image` and `--end-image` inputs as well as
every ordered Ref2VA `--reference`. Rows are ordered and contain `order`,
`kind`, `bytes`, and `sha256`; the header is optional and additional columns
are ignored. A FL2VA manifest contains its image rows only, while a Ref2VA
manifest contains its ordered reference rows:

```text
order	kind	bytes	sha256
1	image	110364	34ea0fee383e3b5d353f6a9556af12b5e7d3a7846c6899e768791b5354818ebd
2	audio	663630	444afb780a0b1a8fe5b1bb90ac744669ef9086c721439c4a0d569389c2e1df80
```

```bash
MERERUN_H3_BAKEOFF_REFERENCE_MANIFEST=./references.expected.tsv \
  scripts/h3-algorithm-bakeoff.sh \
  video-minimax-h3-ref2va-mlx \
  .build/h3-bakeoff/ref2va \
  "preserve the subject and motion" \
  --reference image:./subject.png \
  --reference audio:./voice.wav
```

The default 512 x 512 matrix includes native resolution, 384 x 384 (75%), 320 x 320
(62.5%), 45 and 40 layers, interval-2 velocity reuse, and token reduction.
Override geometry, seed, frame count, executable, or the comma-separated arm
list with the `MERERUN_H3_BAKEOFF_*` variables documented by the script's
defaults. A render arm is marked skipped when an exact same-aspect scale cannot
remain on the 32-pixel grid; the harness does not round it into a different aspect.

The runner fails closed when a matching ML process, `mere.run` process, Swift
compiler, or Xcode build is active. It also rejects a host starting above
`MERERUN_H3_BAKEOFF_MAX_STARTING_SWAP_MIB`, which defaults to 1024 MiB. Raising
that ceiling is an explicit evidence-policy change and must be reported with
the results; it is not a way to describe swap-heavy timing as clean.

The process and swap gates are checked again immediately before every arm and
the arm's starting swap is written directly to `receipts.tsv`. A long matrix
therefore cannot silently cross the evidence policy after its initial gate; a
additional competing process is captured as sanitized process ID, executable,
and working-directory evidence. To
score a narrow follow-up without regenerating a byte-identical expensive
baseline, omit the `quality` arm and set `MERERUN_H3_BAKEOFF_BASELINE` to the
existing MP4. The runner rejects a missing baseline, records its resolved path,
byte count, and SHA-256, and does not allow an external baseline together with
another `quality` arm.

`receipts.tsv` measures generation only; preflight is completed before its
clock starts. Every passing arm records wall time, `/usr/bin/time` maximum RSS
and peak footprint, the maximum MLX `peak_gib` reported by per-step profiling,
the output SHA-256, and all raw logs. The timed command enables both step and
phase profiling so denoising, conditioner/text preparation, transformer
preparation, video VAE, audio VAE, and generation-total values are preserved.
`environment.txt` preserves hardware, OS, thermal warnings, swap, VM
statistics, disk headroom, the process-deny pattern, the exact executable
SHA-256, the prompt SHA-256, the reference-manifest SHA-256, and the clean/dirty
source state. `prompt.txt`, `arguments.tsv`, and `references.tsv` retain the
exact request and resolved reference identities; `start-gate.txt` preserves
rejected process and swap evidence even when no arm runs. Per-arm before/after
snapshots capture thermal, swap, and VM state around the timed region.

After generation, `scripts/h3-bakeoff-score.py` verifies that every MP4 matches
the requested width, height, frame count, 24-frames-per-second video, 32 kHz stereo audio,
and duration. It compares each arm with the same-seed dense-quality output and
writes a matched eight-frame baseline/candidate contact sheet, one JSON report,
and a `quality.tsv` summary containing video SSIM, PSNR, VMAF, decoded-audio
zero-lag correlation, relative L2, RMS, peak, and clipping fractions. A
structural or metric failure makes the harness fail but does not delete the
expensive artifacts or raw evidence.

The numeric media scores are diagnostics, not an automatic acceptance rule.
They expose trajectory drift and broken media contracts; blinded visual review,
reference retention, motion/coherence, dialogue intelligibility, and A/V sync
remain required. Score an existing pair independently with:

```bash
scripts/h3-bakeoff-score.py \
  ./quality.mp4 \
  ./candidate.mp4 \
  --json ./candidate.quality.json \
  --contact-sheet ./candidate.contact.png \
  --expected-width 512 \
  --expected-height 512 \
  --expected-frames 124 \
  --expected-fps 24 \
  --expected-sample-rate 32000 \
  --expected-channels 2
```

### Fixed-seed algorithm result

The post-reboot algorithm screen used the same pinned railway fixture as the
exact-kernel bake-off: 512 x 256, 124 frames at 24 frames per second, seed `20260810`, image
SHA-256 `34ea0fee383e3b5d353f6a9556af12b5e7d3a7846c6899e768791b5354818ebd`,
and, for Ref2VA, audio SHA-256
`444afb780a0b1a8fe5b1bb90ac744669ef9086c721439c4a0d569389c2e1df80`.
All arms used release binary SHA-256
`bb79a419bcf508d5b732daee58214e8cf9b490bf104904625e09d9891a5e1c56`
from commit `ee64b21c`.

This first screen predates the h3.c-aligned reference-serving contract. Its
Ref2VA latency and 88.59 GiB peak describe the former 2,048-pixel-short-edge
reference preprocessing path, not the dense runtime.

The full FL2VA matrix began with zero swap. Every arm through A3 also began at
zero swap; A3 caused macOS to retain 200.62 MiB, so A4 began below, but not at,
the 1024 MiB ceiling. No arm recorded a thermal or performance warning.

| FL2VA arm | Wall time | Delta vs quality | Video SSIM | VMAF | Audio correlation | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| quality / 50 blocks | 211.995 s | baseline | 1.000000 | 99.543328 | 1.000000 | dense reference |
| A1 / 75% internal canvas | 129.447 s | -38.94% | 0.808859 | 2.761354 | 0.398702743 | reject tested scale |
| A1 / 62.5% internal canvas | 96.570 s | -54.45% | 0.797838 | 1.490763 | 0.408785695 | reject tested scale |
| A2 / 45 active blocks | 211.230 s | -0.36% | 0.820338 | 3.437342 | 0.095347761 | no useful speed win |
| A2 / 40 active blocks | 181.841 s | -14.22% | 0.792344 | 2.987691 | 0.082693270 | reject quality |
| A3 / interval-2 velocity reuse | 177.169 s | -16.43% | 0.870424 | 18.213738 | 0.376719974 | reject complete reuse |
| A4 / token reduction | 152.245 s | -28.19% | 0.844319 | 4.141657 | 0.419743523 | reject visible artifacts |

Every FL2VA arm retained the same 61.78 GiB MLX peak and an effectively
unchanged process peak footprint. The metrics are matched-seed diagnostics, so
they penalize any trajectory change rather than directly measuring semantic
quality. Contact-sheet review nevertheless confirmed the decision: A1 changed
train timing and color, A2 changed carriage motion and lost audio fidelity, A3
was the closest visual arm but changed train motion, and A4 produced visible
facial ghosting.

Only the two evidence-selected candidates were carried into Ref2VA. They were
scored against the clean dense baseline MP4 whose SHA-256 is
`e2a2af06e1c3b3ac4f8cfc25f46de0d8edcab2a729e40afc003a2eac572aeb9f`.
That artifact was generated twice byte-identically in zero-swap sessions at
1,920.079 and 1,911.015 seconds; their 1,915.547-second mean is used only as the
timing reference for this comparison. The selected follow-up began with 200.62 MiB of swap,
dropped to 192.62 MiB, and recorded no thermal or performance warning, so these
are screening timings rather than additional zero-swap baselines.

| Ref2VA arm | Wall time | Delta vs baseline mean | Process-peak delta | Video SSIM | VMAF | Audio correlation | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A1 / 75% internal canvas | 1,588.565 s | -17.07% | +3.362 GiB | 0.759957 | 1.667707 | 0.995790045 | retain as research only |
| A3 / interval-2 velocity reuse | 1,245.342 s | -34.99% | -0.505 GiB | 0.779995 | 6.565895 | 0.995529730 | retain for blind multi-seed QA |

Both candidates retained the 88.59 GiB MLX peak. A3 performed three genuine
zero-block cache hits, executing 250 blocks instead of the dense 400. It also
preserved the generated waveform closely, and motion continues across the full
124-frame output. Relative to the same-seed dense baseline, mean frame luma
falls from 29.073 to 23.074 (-20.6%) and the blue/rain structure changes
substantially. The dense baseline also lacks a clearly identifiable passing
train, so train omission is a fixture-level prompt/seed compliance limitation,
not evidence against velocity reuse. The demonstrated result is
photometric and structural divergence, not frozen motion. Because SSIM and VMAF
against another generative output measure divergence rather than independent
perceptual quality, this fixture does not establish that A3 is visually worse.
A1 kept the subjects, dialogue waveform, and moving-train concept, but changed
train trajectory/color and softened faces. Neither mode advances to ordinary
dispatch from this single fixture. This result selected A3 for the multi-seed
multi-seed follow-up; it did not itself authorize promotion.

### Post-alignment Ref2VA multi-seed result

The follow-up reran dense quality and interval-2 velocity reuse for seeds
`20260810`, `20260811`, and `20260812` after reference preprocessing was
aligned with the pinned h3.c serving contract. It used commit `2fe645f7`,
release executable SHA-256
`34854543dfacf58383e1feebaaeeced65aed3b532676e1e768979235420f9125`,
the same prompt and pinned image/audio references, and explicitly disabled the
exact-kernel mode. Every output passed the 512 x 256, 124-frame,
24-frames-per-second, stereo
32 kHz structural gate.

These runs began with 12,694.5 to 12,710.5 MiB of swap after earlier model
work. No thermal or performance warning was recorded. They establish
cross-seed behavior and the direction of the serving-contract improvement,
but their wall times remain screening evidence until repeated from zero swap.

| Seed | Dense wall | Velocity wall | Wall reduction | Dense / velocity MLX peak | Video SSIM | PSNR | VMAF | Audio correlation | Audio relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260810` | 404.458 s | 258.673 s | 36.04% | 41.18 / 40.39 GiB | 0.857812 | 28.137487 dB | 16.863799 | 0.998590921 | 0.0534063275 |
| `20260811` | 380.310 s | 257.649 s | 32.25% | 41.14 / 40.21 GiB | 0.784123 | 23.751878 dB | 7.906392 | 0.998063194 | 0.0629630560 |
| `20260812` | 357.576 s | 244.846 s | 31.53% | 41.14 / 40.21 GiB | 0.849296 | 26.587013 dB | 7.901656 | 0.998182345 | 0.0602687853 |
| Mean | 380.781 s | 253.723 s | 33.37% | 41.15 / 40.27 GiB | 0.830410 | 26.158793 dB | 10.890616 | 0.998278820 | 0.0588793900 |

The dense arm now packed 5,919 rows and peaked around 41.15 GiB, compared with
the earlier mis-sized reference path's 88.59 GiB peak. Its 380.781-second mean
is 80.12% below the old 1,915.547-second mean. That is an exact dense-path
serving improvement: no denoise evaluation was skipped and no approximation
mode was selected.

Velocity reuse then saved a repeatable 31.53% to 36.04% over each matched dense
run and kept audio correlation above 0.998. It nevertheless changed train
appearance, lighting, or trajectory in all three contact sheets; seed
`20260811` also developed a conspicuous magenta train-light artifact late in
the clip. The result rejects velocity reuse as an ordinary default. It stays
available only as a named experimental policy for later research.

### Zero-swap finalist retake

The final non-LTX retake used commit `bc2532a7`, release executable SHA-256
`34854543dfacf58383e1feebaaeeced65aed3b532676e1e768979235420f9125`,
an Apple M4 Max with 128 GB of unified memory, and macOS 26.5.2. Both arms
started with zero swap, a clean worktree, no matching build or ML process, and
no thermal or performance warning. Ref2VA also ended at zero swap; the larger
LightX2V arm ended with 15.94 MiB.

| Finalist | Geometry and evaluations | Wall time | Denoising | Non-denoise remainder | MLX peak | Process peak footprint | Output SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Dense Ref2VA | 512 x 256, 8 | 578.450 s | 528.492 s | 49.958 s | 41.14 GiB | 47,255,358,296 bytes | `7b716380a857c3203e8434db11493fbf8e9b2b26166da41bf5f84e18f95c0866` |
| LightX2V v1.0 8-step | 960 x 544, 8 | 2,457.841 s | 2,277.428 s | 180.413 s | 61.90 GiB | 68,143,584,184 bytes | `cebc0d88c80145057cd7e6daeaf45568a59122ac72173d91c94c048681995822` |

The Ref2VA output is byte-identical to seed `20260810` from the post-alignment
multi-seed screen. Compared with the old clean 1,920.079-second,
88.59-GiB path, the corrected serving contract reduces cold-host wall time by
69.87% and MLX peak by 53.56%. The screening runs were faster because they
followed substantial earlier model work and began with about 12.7 GiB of swap;
they are retained for cross-seed behavior, not averaged into this cold-host
timing.

The selected LightX2V finalist uses the released v1.0 8-step recipe at its
960 x 544 training geometry: shifts 12 and 3, LoRA alpha 8, eight evaluations, and
19,317 packed rows. Its output is byte-identical to the earlier screening
artifact, including 124 H.264 frames and stereo 32 kHz AAC with mean/max levels
of -23.2/-5.6 dB. The cold-host wall was 3.59% slower and denoising 4.54% slower
than that swap-heavy screen. The result confirms reproducibility and the
expected cost; it is not evidence that swap improves performance.

The retake exposed a harness receipt omission: the wrapper enabled step
profiling but not the runtime's phase profiler. Therefore the table reports the
exact clean non-denoise remainder rather than inventing a clean load/VAE split.
The byte-identical earlier LightX2V screen separately recorded 0.821 seconds of
conditioner preparation, 0.105 seconds of conditioner load, 4.921 seconds of
text encoding, 14.869 seconds of transformer preparation, 169.078 seconds of
video VAE decode, and 1.534 seconds of audio VAE decode. Those phase values are
useful attribution evidence but remain labeled as screening timings. The
harness now enables both phase and step profiling for every subsequent arm.

The portable direction is therefore narrower, not more aggressive: retain the
exact reference-serving alignment, evaluate a 448 x 224 (87.5%) internal canvas,
at most one reused middle evaluation, less aggressive token pairing/block
spans, and true load-time pruning for A2 before another promotion attempt.
Exact `boundary-layout` remains the only h3.c kernel lane with a supported
experimental execution policy in this PR.

## Acceptance gates

Every result records the commit, executable, model artifact identity, hardware,
OS, thermal state, competing ML processes, prompt, references, seed, output
geometry, internal geometry, frame count, schedule, and acceleration flags.

Kernel gates:

1. A deterministic small-shape tensor test against decomposed MLX operations.
2. A production-shape release benchmark with alternating arm order and warmup.
3. Peak Metal, physical-memory, and swap deltas.
4. One complete 50-block forward using real weights.
5. Same-seed FL2VA and Ref2VA generations with the fallback forced available.
6. `./scripts/check.sh` on the final production change.

Algorithm gates:

1. Fixed FL2VA and Ref2VA prompt/reference fixtures.
2. Dense-quality output retained as the source-of-truth arm.
3. Video and audio latent relative L2, frame-level similarity, audio integrity,
   synchronization, and blinded visual review.
4. Separate isolated release runs; no timing from debug binaries or concurrent
   ML workloads.
5. No automatic default until the accepted quality envelope is documented.

## Initial execution order

1. Qualify K1 on the local M4 Max and measure whether it actually removes time
   or memory from a production H3 block.
2. Add K2 as a BF16 head-major parity kernel before introducing INT8 projection
   arithmetic. This isolates layout correctness from quantization quality.
3. Qualify the resident-BF16 MPP projection candidate on M3/M4 at all four H3
   projection shapes and then through one exact 50-block forward. Keep it out
   of model dispatch until those clean-host gates pass; M5 work is out of scope.
4. Qualify K3, K4, and K5 in that order because each consumes the preceding
   layout and activation contract.
5. Run A1-A4 only after the exact kernel baseline is stable, beginning with A1
   and the already-related A3 cache machinery.
