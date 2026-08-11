# MiniMax-H3 h3.c transfer program

This document defines the evidence contract for transferring useful h3.c
techniques into mere.run. It is not a performance receipt. Production behavior
does not change until a candidate passes its tensor, checkpoint, quality,
memory, and fallback gates.

## Fixed sources

- mere.run base: Ref2VA 8-bit commit
  `184932ae5e2788812376285a39c606c6a568ebd3`, stacked on `origin/main`.
- h3.c repository: `https://github.com/antirez/h3.c.git`.
- h3.c revision: `f0dbe7699250c4943ec148ed7c2c16031fee8d05`.
- h3.c source license: MIT, with the BSD-3-Clause components identified in its
  `THIRD_PARTY_NOTICES.md`.
- MiniMax-H3 weights remain governed by the MiniMax-H3 Community License.

Run the source oracle without adding it to the repository:

```bash
scripts/h3c-oracle.sh test
scripts/h3c-oracle.sh pin
scripts/h3c-oracle.sh run --help
```

The ignored checkout lives under `.build/h3c-oracle` by default. The runner
refuses revision drift by fetching and checking out the immutable commit above
on every build. It does not download checkpoint weights. h3.c expects the
original upstream `FL2VA/` and `Ref2VA/` directory trees; mere.run consumes its
own flat managed MLX artifacts, so output comparisons must record which weight
representation each arm used.

## Two independent lanes

### Kernel lane

Each kernel candidate must first reproduce an explicit decomposed oracle. If a
candidate deliberately changes arithmetic, such as dynamic INT8 activation
quantization, it also needs a same-seed checkpoint quality A/B and may not be
described as trajectory-exact.

| ID | Candidate | First proving shape | Required fallback |
| --- | --- | --- | --- |
| K1 | Attention residual gate + following MLP AdaLN, then optional activation quantization | BF16 `[1, rows, 5376]` | Existing decomposed MLX graph |
| K2 | QKV projection directly to `[1, 56, rows, 128]`, with fused Q/K RMSNorm and RoPE | H3 QKV `5376 -> 21504` | Current linear, split, transpose, norm, and RoPE path |
| K3 | Head-major SDPA output directly into the INT8 output projection | H3 attention output `7168 -> 5376` | Current transpose, reshape, and linear path |
| K4 | FC1, SwiGLU, and FC2 with H3-specialized `5376 -> 28672 -> 14336 -> 5376` dimensions | Complete production MLP | Current MLX compiled MLP |
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
dispatch below. The activation-INT8 continuation remains lab-only.

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
own stdout, stderr, and start-gate receipt, so resuming never overwrites failed
evidence. A clean-host rejection stops the suite immediately with exit 75;
other failures are recorded while remaining modes continue. Successful modes
leave commit-bound pass markers, so `--resume` skips only proven passes. The
suite requires a clean worktree and binds resume to its original commit and
ordered mode set. An atomic repository-local lock prevents two suites from
producing overlapping timing evidence; an exited lock owner is recovered only
after its recorded PID no longer exists.

K2 through K4 need two implementations where hardware requires it:

- a portable MLX/Metal fallback for current M-series Macs;
- an M5-gated Metal 4/TensorOps implementation, never selected by device-name
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

The K2a deterministic GPU canary compares all three outputs with the current
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
This is weight INT8 with floating activations, matching the current artifact;
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
per call when the exact H3 shape or dtype contract is unavailable.

`MERERUN_H3_EXACT_KERNELS=affine-q8` enables the exact kernel candidates inside
the real transformer loop for controlled checkpoint A/Bs. Admission is typed
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
50-block real-weight arithmetic pass is complete. Full generated-video quality
review, peak-memory receipts, and uncontaminated release timing are still
required before ordinary dispatch changes.

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

## Quality-sensitive algorithm lane

These modes are always explicit. They may compose only after their individual
quality envelopes have been measured.

| ID | Candidate | Initial comparison |
| --- | --- | --- |
| A1 | Same-aspect reduced internal render canvas followed by high-quality upscale | Native output, 75%, and 62.5% internal dimensions |
| A2 | AdaLN-gate-ranked layer thinning with protected first/final blocks | 50, 45, and 40 active blocks |
| A3 | Whole-velocity or transformer-core reuse | Current adaptive tail reuse, interval-2 velocity reuse, and interval-4 core reuse |
| A4 | Target-video token pairing with full-resolution bypass and delta restoration | Blocks 4-40 early, then 4-30, versus full tokens |

The current adaptive block-tail cache is not equivalent to h3.c whole-velocity
or core reuse. It remains a separate bake-off arm rather than being silently
relabelled.

A3 now has its first explicit runtime arm: `--h3-acceleration
velocity-reuse-2`. It runs the first and final denoise evaluations in full,
reuses both complete video and audio velocity outputs on intervening odd steps,
and keeps the quality schedule point count. Selecting it disables the adaptive
tail and dynamic-sparse policies so its quality and timing deltas remain
attributable. It is experimental and non-default until the fixed FL2VA and
Ref2VA same-seed A/Bs pass.

A2 has two equally isolated arms: `--h3-acceleration layers-45` and
`--h3-acceleration layers-40`. They reproduce the pinned oracle's ranking:
mean absolute attention and MLP AdaLN gate values over every cached schedule
point and modality, with blocks 0, 1, and 49 always protected. The lowest
remaining scores are skipped while original block indices and weights remain
unchanged. The current prototype reduces executed transformer work but retains
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
preflight JSON and stderr log per arm, and records wall time plus output SHA256:

```bash
scripts/h3-algorithm-bakeoff.sh \
  video-minimax-h3-fl2va-bf16-mlx \
  .build/h3-bakeoff/fl2va \
  "a fixed evaluation prompt"

scripts/h3-algorithm-bakeoff.sh \
  video-minimax-h3-ref2va-mlx \
  .build/h3-bakeoff/ref2va \
  "preserve the subject and motion" \
  --reference image:./subject.png \
  --reference video:./motion.mp4
```

Pin reference identity for a reportable Ref2VA run with a tab-separated
manifest. Rows are ordered and contain `order`, `kind`, `bytes`, and `sha256`;
the header is optional and additional columns are ignored:

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

The default 512x512 matrix includes native resolution, 384x384 (75%), 320x320
(62.5%), 45 and 40 layers, interval-2 velocity reuse, and token reduction.
Override geometry, seed, frame count, executable, or the comma-separated arm
list with the `MERERUN_H3_BAKEOFF_*` variables documented by the script's
defaults. A render arm is marked skipped when an exact same-aspect scale cannot
remain on the 32px grid; the harness never rounds it into a different aspect.

The runner fails closed when a matching ML process, `mere.run` process, Swift
compiler, or Xcode build is active. It also rejects a host starting above
`MERERUN_H3_BAKEOFF_MAX_STARTING_SWAP_MIB`, which defaults to 1024 MiB. Raising
that ceiling is an explicit evidence-policy change and must be reported with
the results; it is not a way to describe swap-heavy timing as clean.

`receipts.tsv` measures generation only; preflight is completed before its
clock starts. Every passing arm records wall time, `/usr/bin/time` maximum RSS
and peak footprint, the maximum MLX `peak_gib` reported by per-step profiling,
the output SHA-256, and all raw logs. `environment.txt` preserves hardware, OS,
thermal warnings, swap, VM statistics, disk headroom, the process-deny pattern,
the exact executable SHA-256, the prompt SHA-256, the reference-manifest
SHA-256, and the clean/dirty source state. `prompt.txt`, `arguments.tsv`, and
`references.tsv` retain the exact request and resolved reference identities;
`start-gate.txt` preserves rejected process and swap evidence even when no arm
runs. Per-arm before/after snapshots capture thermal, swap, and VM state around
the timed region.

After generation, `scripts/h3-bakeoff-score.py` verifies that every MP4 matches
the requested width, height, frame count, 24 fps video, 32 kHz stereo audio,
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
3. Prototype M5 INT8/TensorOps work in the pinned mlx-swift/MLXFast fork; retain
   K1/K2 portable fallbacks in mere.run.
4. Qualify K3, K4, and K5 in that order because each consumes the preceding
   layout and activation contract.
5. Run A1-A4 only after the exact kernel baseline is stable, beginning with A1
   and the already-related A3 cache machinery.
