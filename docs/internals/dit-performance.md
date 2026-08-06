# DiT Forward-Pass Performance

Findings from the image-stack deep dive (July 2026, M4 Max 128 GB), asking why
the diffusion-transformer denoise step — 91-96% of image generation wall time —
appears to run at low hardware utilization, and which optimizations are worth
carrying. Short version: the step is roofline-bound on GEMM/SDPA kernels;
almost everything else was measured and rejected.

The measurement harness lives in `Tests/MereRunCoreTests/DiTShapeBenchTests.swift`
(env-gated, in-process, interleaved arms so ambient load hits both sides):

```bash
MERERUN_DIT_BENCH=1 swift test -c release -Xswiftc -DDEBUG --filter DiTShapeBenchTests
```

`-DDEBUG` works around test helpers gated behind `#if DEBUG` that otherwise
fail the release test build; if the runner dies with "Failed to load the
default metallib", colocate one next to the test binary:

```bash
cp .build/arm64-apple-macosx/release/Resources/default.metallib \
   .build/arm64-apple-macosx/release/MereRunPackageTests.xctest/Contents/MacOS/mlx.metallib
```

## Where a klein-nano step goes (1024², 4-bit, ~3.8 s/step)

Cumulative-prefix timing of one single-block attention and block-count scaling
of the full transformer close the ledger:

| component | share of step |
| --- | --- |
| fused projection + output GEMMs (4-bit qmm) | ~65% |
| SDPA (`MLXFast.scaledDotProductAttention`) | ~15% |
| MLP elementwise, QK norms, RoPE apply, splits/transposes | ~5% |
| double blocks' extra text-stream GEMMs, embedders, final norm | remainder |

The 20 single blocks dominate (~80% of the forward). GEMMs run at 10-14.5
TFLOPS bf16 and SDPA at ~12.5 TFLOPS — both at MLX's practical ceiling for
these shapes on this hardware. Effective whole-forward utilization is roughly
38-40% of nominal peak, which is normal for MLX GEMM workloads; earlier "15%
MFU" estimates were calibrated against an unrealistic peak. **There is no
hidden overhead layer to recover.** Faster images come from fewer steps,
lower resolution, or smaller models — not kernel work.

## Measured and rejected

- **`MLX.compile` over the transformer** (`MERERUN_FLUX2_COMPILE=1`): 15%
  *slower* than eager on the full forward (3815 → 4395 ms). The flag also
  skips the per-step `MLX.eval`, which makes `MERERUN_FLUX2_TIMING` report
  graph-build time (~5 ms/step) instead of real work — do not read step
  timings under compile.
- **Dequantize + dense GEMM instead of quantized matmul.** In an isolated
  microbench, 4-bit qmm costs 1.24-1.45x bf16 at DiT shapes and in-graph
  dequantize+GEMM looked ~20% faster. In the full model the effect inverts:
  transient dequant is +10% and cached dense weights +13% versus qmm
  (paired in-process A/B, `testKleinNanoQmmVsDensePaired`). The microbench
  win was a hot-cache artifact — an isolated op re-reads the same weights
  every iteration, while the real forward reads each layer's weights once,
  cold, where 4-bit's 4x-smaller weight traffic wins. Native qmm is the
  right call at every batch size; outputs of the two paths matched to
  `maxAbsDiff = 0.0` in bf16.
- **Krea2's dense additive attention mask**: an all-zeros `[B,1,L,L]` mask
  costs ~1 ms on a ~39 ms SDPA (~0.1% of the step). Not worth the plumbing
  to elide.
- **Krea2's manual KV-head repeat** (12 kv-heads → 48 q-heads before SDPA):
  native GQA saves ~0.7 ms per SDPA. Same verdict.

## Carried

- **Half-precision RoPE apply** (Flux2/Klein): `applyRotaryEmb` used to
  round-trip q/k through float32 around a handful of multiply-adds. The
  sin/cos tables are still computed in float32 upstream; applying the
  rotation in the tensor's own dtype is 2.5x faster at klein shapes
  (2.4 → 0.95 ms per apply, ~50 applies per forward, ≈2% of the step).
  The `image-klein-seed7` gate check passes byte-identical against the
  fp32-rotation baseline, so no baseline re-record was needed.

## Klein-nano facts worth remembering

- The managed `image-klein-nano` transformer ships 4-bit groupwise quantized
  (109 U32 weight tensors + scales/biases in a 2.0 GB single-file
  checkpoint); `image-klein-9b` ships float16. Quantized single-file loads
  go through `applyQuantizedWeightsFromArrays` → `PortableQuantizedLinear`.
- Geometry: hidden 3072 (24 heads × 128), 5 double + 20 single blocks,
  mlp_ratio 3, 4096 image tokens at 1024² plus ~512 text tokens.
- Single blocks fuse QKV+MLP-in into one 3072→18432 projection and
  attn+MLP-out into one 12288→3072 projection — already the right shape for
  the hardware; splitting or refusing them was not worth measuring further.

## MiniMax-H3 kernel lab

H3 performance experiments must start with bounded, non-generative loops. The
default lab loads no checkpoint, runs no conditioner or VAE, and writes no
media. It searches exact schedules for the dominant dense-attention kernel at
the measured 14,958-row shape:

```bash
scripts/h3-kernel-lab.sh quick
```

The script reuses the release XCTest bundle while it is newer than `Package.swift`,
`Sources/`, and `Tests/`; set `MERERUN_H3_LAB_REBUILD=1` to force a cold rebuild.

The quick loop walks query chunk size, attention-head chunk size, and the number
of Metal kernels queued before synchronization using coordinate descent. That
keeps the default inner loop to about a dozen arms instead of a Cartesian sweep.
It reports wall time, effective TFLOP/s, peak memory, and speedup relative to the
current production schedule. Every arm is compared with that schedule using
maximum absolute and relative-L2 error; a numerically divergent arm fails the
test instead of becoming a runtime candidate.

Each arm is timed next to the production schedule, with their order reversed on
alternate rounds. This paired design cancels most thermal drift and sweep-order
bias; do not promote a raw first-arm timing from an unpaired loop.

The search surface is configurable without editing code:

```bash
MERERUN_H3_BENCH_ROWS=37966 \
MERERUN_H3_BENCH_CHUNKS=1024,1536,2048,2560,3072,4096 \
MERERUN_H3_BENCH_HEAD_CHUNKS=14,28,56 \
MERERUN_H3_BENCH_EVAL_BATCHES=1,2,4,8 \
scripts/h3-kernel-lab.sh attention
```

`attention` defaults to the exhaustive grid. Set
`MERERUN_H3_BENCH_SEARCH=coordinate` when narrowing a larger shape before the
full confirmation sweep.

Projection-only loops isolate Q4 matrix multiplication from the resident-BF16
path without running a transformer step:

```bash
MERERUN_H3_BENCH_ROWS=14958,37966 scripts/h3-kernel-lab.sh projections
```

The modulation loop compares per-token AdaLN gathers with exact contiguous-run
modulation, without loading a checkpoint:

```bash
scripts/h3-kernel-lab.sh modulation
```

At 29,018 rows, the run formulation matched exactly and measured 5.7 ms versus
10.4 ms for gathers (1.835x). That isolated saving is too small relative to a
full H3 block to justify production graph complexity, so the experiment remains
in the lab.

The full-block loop instantiates one production-width H3 block with resident
BF16 weights. It measures the real compiled graph boundaries, chunked SDPA,
and MLP together instead of extrapolating from isolated kernels:

```bash
MERERUN_H3_BENCH_ROWS=37966 scripts/h3-kernel-lab.sh block
```

At the 37,966-row true-768 shape, exact fresh-process controls measured the
1,024-query schedule at 9.287 seconds for a split full block and 6.511 seconds
for its attention phase. The previous 2,048-query schedule measured 10.350 and
7.120 seconds respectively. The 1,024-query schedule therefore carries into
production; its output matched the previous schedule exactly in the attention
gate (`max_abs=0`, `rel_l2=0`).

A finer coordinate pass found a smaller but repeatable improvement at that
large-row tier: 768 query rows with one attention kernel evaluated at a time.
The order-balanced whole-block acceptance gate measured 9.733 versus 9.592
seconds (1.015x) in one fresh process and 10.475 versus 10.278 seconds (1.019x)
in a second, with bit-identical BF16 output (`rel_l2=0`). Production therefore
uses 768-by-1 for 32,768+ packed rows while retaining 1,024-by-4 below that
measured tier. Reproduce the paired gate directly with:

```bash
scripts/h3-kernel-lab.sh attention-block
```

Several tempting graph changes did not carry. Explicitly materializing
contiguous Q/K/V copies regressed the full block to 15.148 seconds. Narrowing
the implicit module state passed to each compiled closure also regressed, and
larger post-attention fusion did not repeatably beat the existing split graph.
Those arms remain rejected rather than becoming hidden runtime switches.

The lower-level MLX Metal tile is not an untapped switch either. A temporary
JIT-controlled sweep ran one BF16 attention chunk with 56 heads, 1,024 query
rows, 37,966 key/value rows, and head dimension 128. Every arm was checked
against the default output before the dependency checkout was restored:

| Steel tile (BQ x BK) | Median | Effective throughput |
| --- | ---: | ---: |
| **32 x 16 (MLX default)** | **102.4 ms** | **10.89 TFLOP/s** |
| 16 x 16 | 157.3 ms | 7.09 TFLOP/s |
| 32 x 8 | 103.7 ms | 10.75 TFLOP/s |
| 32 x 32 | 123.4 ms | 9.04 TFLOP/s |
| 64 x 16 | 107.6 ms | 10.36 TFLOP/s |
| 64 x 32 | 115.2 ms | 9.67 TFLOP/s |

The default was both fastest and bit-identical to its repeated control. The
different-key-tile arms remained well inside the numerical gate but did not
improve time. H3 is already using MLX's fused Steel full-attention path on M4;
the NAX path is unavailable on this generation, and replacing Steel with a new
approximate attention algorithm would be a model-math change rather than a
runtime scheduling optimization.

Dense projection dispatch did expose one exact M4 win. MLX's large-device BF16
heuristic used a 64 x 64 x 16 Steel GEMM tile with two total warps and no
threadgroup swizzle. At the true-768 row count, four total warps plus a two-bit
swizzle improved every production projection family; selecting the alternate
32-row tile for the two shapes where it won preserved the same whole-block gain:

| H3 projection | MLX default | Tuned | Speedup |
| --- | ---: | ---: | ---: |
| QKV, 5,376 -> 21,504 | 745.1 ms | 621.2 ms | 1.199x |
| attention out, 7,168 -> 5,376 | 270.0 ms | 237.7 ms | 1.136x |
| feed-forward in, 5,376 -> 28,672 | 1,099.1 ms | 994.4 ms | 1.105x |
| feed-forward out, 14,336 -> 5,376 | 669.0 ms | 583.9 ms | 1.146x |

Every projection arm was bit-identical in BF16. A three-arm, order-balanced
whole-block comparison then measured 7.892 seconds for MLX's default, 7.625
seconds for the generic four-warp schedule, and 7.619 seconds for the
shape-aware schedule. The shape-aware result is a 1.036x exact full-block
speedup (`rel_l2=0`), and is deliberately restricted to H3's four exact
projection dimensions above 32,768 packed rows. Reproduce the projection and
whole-block checks with `scripts/h3-kernel-lab.sh gemm` and
`scripts/h3-kernel-lab.sh gemm-block`.

The release-mode one-evaluation model probe packed 37,794 rows. After the
shape-aware GEMM and large-row attention schedules landed, its 50 blocks fell
from 953.192 to 775.062 seconds (1.230x). Counting the dominant projection and
dense-attention arithmetic gives about 3.504 PFLOP per evaluation, improving
the sustained pass from 3.68 to 4.52 effective TFLOP/s. This is lower than both
the 7.6 TFLOP/s hot
single-block result and the 10-14 TFLOP/s isolated-kernel roof because the real
pass turns over 50 independent parameter sets and includes every dependent
normalization, modulation, transpose, synchronization, and thermal effect.
Peak Metal memory was only 48.77 GiB; additional disk or unified memory does
not close that utilization gap.

The first valid ten-second geometry is 243 frames, which produces 72 video
latent frames. At 1344x768, the locked benchmark prompt therefore packs 73,470
rows: 72,576 video rows, 810 audio rows, and 84 text rows. True-shape searches
did not justify another transformer policy. A sampled 384-query/four-kernel
attention candidate lost its full-block gate to the shipped 768-query/single-
kernel schedule (31.845 s versus 31.438 s, `rel_l2=0`). A longer-row GEMM tier
also lost its order-balanced whole-block gate (29.151 s versus 29.088 s,
`rel_l2=0`) and was removed. These results put one exact 50-block evaluation
near 24-26 minutes before VAE decode; a 20-point exact schedule still needs 19
such evaluations.

The video VAE did retain a separate full-precision runtime win. Fresh-process released-
checkpoint sweeps at 832x480 and 124 frames measured 96.091 s with 256-pixel
tiles, 76.421 s with 320-pixel tiles, and 77.577 s with 480-pixel tiles. The
320-pixel arm was 1.257x faster than the old default and reduced reported peak
Metal memory from 9.85 to 8.91 GiB. At 1344x768 it decoded the same 124 frames
in 213.005 s at 15.12 GiB peak; the 480-pixel arm crossed 252 seconds without
finishing and was stopped. The production default is therefore 320 pixels.
The model, precision, causal tile algorithm, and overlap blend are unchanged;
the larger tile reduces redundant overlap and tile-boundary context loss.
The true-spatial 22-frame screen also rejected tile-count breakpoints that a
pixel-work estimate alone made attractive: in a hot-state pair, 416-pixel tiles
took 33.756 s versus 32.321 s at 320 pixels, while 432- and 496-pixel arms were
materially slower. Batching two independent temporal chunks into one decoder
submission likewise produced no repeatable wall-time win at 39 frames and
raised peak Metal memory from 13.49 to 21.40 GiB, so temporal chunk execution
remains serial.
Reproduce a candidate in a fresh process with:

```bash
MERERUN_H3_MODEL_ROOT=/path/to/h3 \
MERERUN_H3_VAE_TILE_SIZE=320 \
MERERUN_H3_VAE_WIDTH=1344 \
MERERUN_H3_VAE_HEIGHT=768 \
MERERUN_H3_VAE_FRAMES=124 \
scripts/h3-kernel-lab.sh vae
```

Only improvements that repeat across fresh processes, preserve the numerical
gate, and improve a matched full-block or denoise-step probe should move into
the runtime. Full video generation remains a final quality gate, not the inner
optimization loop.

### Bounded tail-block reuse

Exact attention scheduling and resident BF16 improve each native call, but H3
still executes 50 sequential transformer blocks. The explicit `balanced` and
`maximum` acceleration modes reduce the number of blocks on safe adjacent
schedule steps. A full step stores `finalHidden - warmHidden`; a cache step
recomputes 25 leading blocks in `balanced` or nine in `maximum`, then adds that
stored tail residual. Both video and audio sigma deltas must be below 0.12, the
schedule position must be inside 10%...90%, and at least two complete
evaluations must precede reuse. Balanced refreshes after two cache steps;
maximum refreshes after four.

Automatic long-geometry schedules use 21 points (20 model evaluations),
matching the current practical CUDA default instead of the previous 31 points.
Maximum acceleration caps automatic schedules at 12 points. With an explicit
20-point schedule, its bounded reuse policy runs six full evaluations and
recomputes nine of 50 blocks on 13 cache evaluations, for 417 block calls
versus 950 in exact quality (56.1% less transformer-block work).

On the M4 Max 128 GB test machine, a matched warm-cache 512x320, 22-frame,
16-point run measured 112.413 s of denoise in exact `quality` mode and 74.754 s
in `maximum`: 1.504x faster, or 33.5% less denoise time. Eight of fifteen calls
used 13 blocks. The same-seed MP4 remained coherent but changed portal framing.
For the accepted automatic-speed envelope, a matched 512x320, 22-frame,
12-point same-seed pair measured 62.847 s of denoise and 73.07 s end to end in
`quality`, versus 41.928 s and 51.45 s in `maximum` (1.499x denoise). The
accelerated artifact remained free of the spatial lattice seen when reuse began
after only one full evaluation. This is an approximate trajectory accelerator
rather than an exact kernel optimization; exact `quality` remains the default
and video/audio artifacts are the acceptance boundary for speed-mode changes.

The full acceptance receipt, including per-step timings, artifact geometry,
hash, and the important 124-frame duration boundary, is recorded in
[MiniMax-H3 BF16 on M4 Max](../benchmarks/minimax-h3-bf16-m4-max.md).

A first-order tail-residual extrapolation was also tested at the accepted
four-cache-step block count. It added no meaningful runtime cost, but the
same-seed 416x256 proxy fell from 0.688 SSIM against the exact trajectory to
0.583 and visibly shifted exposure and composition. The predictor was removed;
maximum mode keeps the more faithful bounded stale residual.
