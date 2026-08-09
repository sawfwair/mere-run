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
Fusing only the feed-forward half was exact but lost at both practical production
shapes: 1.879 versus 1.830 seconds at 14,958 rows and 10.517 versus 10.021
seconds at 37,966 rows. Its 30.581 versus 30.800 second result at the 73,470-row
ten-second shape was only a 1.007x lead, too small and too shape-specific to
justify a second production schedule.

Fusing that feed-forward tail across the next block's attention-projection
boundary was also exact, but did not survive the true-shape gate. The paired
two-block work improved from 2.388 to 2.370 seconds at 14,958 rows and from
13.219 to 12.159 seconds at 37,966 rows. At the 73,470-row ten-second shape,
however, it improved only from 29.557 to 29.295 seconds (1.009x), while the
boundary itself regressed from 5.684 to 5.770 seconds. Production therefore
keeps the simpler split boundary. Reproduce the order-balanced release-mode
probe with `scripts/h3-kernel-lab.sh boundary`.

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

The `turnover` lab checks whether that gap comes from reusing one compiled
runner for all 50 parameter sets. At 14,958 rows, two dedicated compiled runners
beat alternating weight rebinding by 1.077x, but the result reversed at the
37,966-row true-768 shape: dedicated runners measured 14.325 seconds versus
13.909 seconds for the shared runner, or 0.971x. Rebound output matched the
corresponding dedicated runner exactly (`rel_l2=0`). Production therefore keeps
one runner rather than multiplying compiled state for a small-shape-only win.
Reproduce the release-mode comparison with `scripts/h3-kernel-lab.sh turnover`.

The first valid ten-second geometry is 243 frames, which produces 72 video
latent frames. At 1344x768, the locked benchmark prompt therefore packs 73,470
rows: 72,576 video rows, 810 audio rows, and 84 text rows. A phase ledger at
that shape measured 32.645 seconds for one split block: 24.351 seconds (74.6%)
in exact attention, 2.079 seconds in its input projection, and 4.424 seconds in
the complete post-attention and feed-forward tail.

Splitting the 56 independent attention heads across smaller Steel submissions
exposed a large exact scheduling win that query-only searches missed. A full
73,470-query attention pass with 640-query chunks, eight heads per kernel, and
one kernel evaluated per submission measured 16.383 seconds versus 22.088
seconds for the paired legacy control (`max_abs=0`, `rel_l2=0`). The decisive
order-balanced full-block gate then measured 32.249 seconds for the shipped
768-query/56-head schedule and 25.173 seconds for 640-query/eight-head
execution: a 1.281x exact speedup with `rel_l2=0`. Production uses this schedule
only at 65,536 or more packed rows; the separately measured 37,966-row schedule
remains unchanged. Reproduce the acceptance gate with:

```bash
MERERUN_H3_BENCH_ROWS=73470 \
MERERUN_H3_BENCH_REFERENCE_QUERY_TOKENS=768 \
MERERUN_H3_BENCH_REFERENCE_HEADS=56 \
MERERUN_H3_BENCH_REFERENCE_EVAL_BATCH=1 \
MERERUN_H3_BENCH_CANDIDATE_QUERY_TOKENS=640 \
MERERUN_H3_BENCH_CANDIDATE_HEADS=8 \
MERERUN_H3_BENCH_CANDIDATE_EVAL_BATCH=1 \
scripts/h3-kernel-lab.sh attention-block
```

A sampled 384-query/four-kernel attention candidate still loses, and a
longer-row GEMM tier also lost its order-balanced whole-block gate (29.151 s
versus 29.088 s, `rel_l2=0`) and was removed. The accepted head schedule puts
one exact 50-block evaluation near 21 minutes before VAE decode. A 20-point
exact schedule still needs 19 such evaluations.

The accepted eight-head schedule was also paired with the transformer's
existing fused post-attention graph. At 73,470 rows, an order-balanced two-round
gate measured 24.801 seconds for the split graph and 24.570 seconds for the
fused graph, a bit-identical but only 1.009x improvement (`max_abs=0`,
`rel_l2=0`). That falls below the 1.02 production threshold, so the runtime
keeps the simpler split graph. Re-run this rejection gate with
`scripts/h3-kernel-lab.sh post`.

A subsequent 40-candidate local grid searched 512...768 query chunks, four,
seven, eight, or fourteen heads per submission, and one or two submissions per
evaluation. Each candidate used 4,096 sampled query rows against the complete
73,470-row key/value shape. The existing 640-query/eight-head/single-submission
schedule remained the winner, projecting 16.532 seconds for the complete
attention pass; the earlier exact full-pass gate measured 16.383 seconds.

Full-block FP16 execution was also screened rather than inferred from nominal
GPU peak rates. With identical seeded weights and inputs, FP16 stayed inside a
one-block `rel_l2 <= 0.01` gate but was slower at every measured packed shape:
1.772 versus 1.652 seconds at 14,958 rows, 8.778 versus 8.644 seconds at 37,966
rows, and 30.204 versus 29.679 seconds at the true ten-second 73,470-row shape.
Production therefore remains resident BF16. Re-run the deterministic comparison
after an MLX or Metal update with `scripts/h3-kernel-lab.sh dtype`.

The fused non-NAX Steel attention microtile was then swept locally at 14,958
rows with exact output checks. Apple's existing 32-query x 16-key tile with
four SIMD groups measured 511 ms in the initial cool-state run. Query tiles of
16, 24, 40, 48, and 64 rows and key tiles of 8, 24, 32, and 64 rows did not
beat it; the valid candidates ranged from 535 to 678 ms. A 48-key candidate
also failed exactness (`rel_l2=0.278`) and was rejected. The temporary MLX
controls were removed, leaving the upstream Steel attention dispatch intact.

The production 73,470-key call was then swept separately with eight heads and
640 query rows per exact Steel submission. Across 20 BQ/BK/warp
specializations, the existing 32-query x 16-key tile with four SIMD groups
remained the local winner: 17.129 ms versus its paired 17.181 ms control, only
measurement noise. Every numerically valid alternative was slower, ranging
from 0.602x to 0.928x; several unsupported tile shapes also failed the output
gate. The temporary selection controls were removed rather than adding a
non-winning H3-specific Steel branch.

A focused 42-candidate schedule grid then searched 592...688 query rows and
five, six, eight, nine, ten, or twelve heads per submission. A short 2,048-row
screen suggested 608/eight, but the complete 73,470-query gate reversed it:
608/eight measured 18.567 seconds versus 17.926 seconds for 640/eight, with
exact output. Queuing every 640/eight submission through `asyncEval` also lost
its direct four-round gate, 18.650 seconds versus 17.182 seconds for synchronous
evaluation (`rel_l2=0`). Both laboratory controls were removed.

The true-ten-second arithmetic explains the remaining scale. One transformer
block contains approximately 211.390 TFLOP: 154.767 TFLOP (73.2%) in dense
attention and 56.624 TFLOP (26.8%) in the four linear projections. The accepted
25.173-second block therefore sustains about 8.40 TFLOP/s end to end. Even an
unattainable 36-TFLOP/s-perfect execution would require 5.872 seconds per block
and 293.6 seconds per complete 50-block evaluation. Nineteen exact evaluations
have a compute-only floor of about 93 minutes; the 417-block `maximum` schedule
has a floor of about 40.8 minutes before VAE decode. This is an arithmetic
lower bound, not a performance forecast, and shows why the current checkpoint
cannot reach a 30-minute ten-second render through exact dispatch tuning alone.

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

The audio VAE has its own checkpoint-backed parity loop:

```bash
MERERUN_H3_MODEL_ROOT=/path/to/h3 scripts/h3-kernel-lab.sh audio-parity
```

This fixture caught a checkpoint-mapping defect that discarded the released
`mean_proj`, `logs_proj`, and `dec_in_proj` biases. With identical deterministic
latents and FP32 weights, the old waveform differed from ComfyUI's
MiniMaxH3AudioVAE at `16e3f3034f2bba1fff6c70cbd759339778555cd6` by
`rel_l2=0.463`. Restoring the biases reduced the complete decode error to
`rel_l2=0.0000194` with effectively unit cosine similarity. The gate locks
reference waveform samples and RMS across the full decoder, while the
production path remains the released 32 kHz stereo model with no denoiser or
post-processing added.

Only improvements that repeat across fresh processes, preserve the numerical
gate, and improve a matched full-block or denoise-step probe should move into
the runtime. Full video generation remains a final quality gate, not the inner
optimization loop.

### Dynamic sparse H3 attention on Apple GPUs

H3's packed sequence is not a homogeneous video grid. It contains dense text,
conditioning video, generated-audio, and target-video regions, so sparsifying
the whole sequence would place dialogue and reference fidelity at unnecessary
risk. The production path instead applies an independently written Metal
implementation of the dynamic-routing ideas published by
[Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/) only to target-video
queries. It uses 64-token blocks and a 128-wide BF16 SIMDgroup-matrix kernel.
Every prefix query stays on MLX's fused dense SDPA path; every prefix key and
the neighboring target-video blocks remain exact for sparse queries.
Query-centroid and key-centroid scores route additional exact blocks at
runtime. Skipped blocks still contribute through key-centroid logits and
summed values in the same online-softmax accumulator.

The first two transformer layers, the first 20% of schedule evaluations, and
the final evaluation remain dense. Production eligibility starts at 12,000
packed rows. Before the first sparse layer at a new shape, an all-routes-dense
sample compares the custom Metal result with fused SDPA; unsupported devices,
dtypes, or a failed error envelope fall back to dense attention. On the real
13,085-row checkpoint tensors, FP32 projection outputs staged through BF16 MMA
passed at `max_abs=0.10026`, `mean_abs=0.00219996`, and
`rel_l2=0.0027471`.

The bounded release benchmarks separate kernel throughput from artifact
quality:

| Input | Rows / prefix / heads | Dense | Sparse | Speedup | Routed blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16 | 12,930 / 951 / 56 | 389 ms | 293 ms | 1.327x | 14.44% |
| FP32 staged to BF16 MMA | 13,085 / 653 / 56 | 717 ms | 371 ms | 1.931x | 15.84% |

Those random-tensor timings establish the execution win, not approximation
quality. The acceptance boundary was a real fixed-seed LightX2V generation at
768x448, 124 frames, 24 fps, and four transformer evaluations. Dense denoising
took 631.826 seconds; dynamic sparse attention took 444.216 seconds, a measured
29.7% reduction. The clean sparse step measured 101.294 seconds versus
131.430 and 134.311 seconds for the comparable late dense steps. The final
protected dense step in the sparse process measured 127.645 seconds, ruling
out a simple late-run thermal advantage. Complete native generation fell from
736.285 to 535.386 seconds. The dense run paid a larger first-step compile cost,
so the end-to-end figure is an observed run receipt rather than an isolated
kernel attribution.

Both MP4s contain exactly 124 coherent H.264 frames plus synchronized 32 kHz
stereo AAC. Eight-point matched contact sheets retained the two actors,
recorder handoff, train motion, faces, and final composition without sparse
block corruption. Native Parakeet transcribed both soundtracks as
`You kept the recording? Every second.` The full hashes and phase ledger are in
[MiniMax-H3 BF16 on M4 Max](../benchmarks/minimax-h3-bf16-m4-max.md).

### Adaptive first-block cache

Exact attention scheduling and resident BF16 improve each native call, but H3
still executes 50 sequential transformer blocks. The explicit `balanced` and
`maximum` acceleration modes now execute block 1 on every schedule point and
use its measured change to decide whether blocks 2 through 50 can be reused.
A full refresh stores the target-only tail residual. The next candidate stacks
four statistics into one host boundary: global and worst-time-slice relative
change for video and audio. Reuse requires every finite statistic to fit its
mode's envelope; otherwise the runtime executes all 50 blocks and refreshes.

`balanced` uses 0.08 global and 0.12 temporal thresholds, admits at most two
adjacent hits, and reserves the final two evaluations for complete refreshes.
`maximum` uses the measured 0.30/0.40 envelope, admits at most four adjacent
hits, and reserves the final evaluation. Both require two complete evaluations
before reuse and apply the cached residual only to target rows, never the text
or media-condition prefix.

The former fixed-depth scheduled-tail policy remains available solely for
matched comparisons through `MERERUN_H3_CACHE_STRATEGY=scheduled-tail`. On the
locked 416x256, 107-frame, 20-point resident-BF16 proxy, scheduled-tail ran 417
blocks in 222.220 seconds of denoise. Adaptive `maximum` ran 362 blocks in
163.953 seconds: 26.2% less denoise time and 24.7% less end-to-end time. A
conservative 0.12/0.18 calibration was explicitly rejected after it admitted
only seven hits and took 328.364 seconds.

This remains an approximate trajectory accelerator rather than an exact kernel
optimization. Exact `quality` remains the default, and synchronized video and
audio artifacts are the acceptance boundary for speed-mode changes. Automatic
long-geometry schedules use 21 points; maximum acceleration caps automatic
schedules at 12 points.

The full acceptance receipt, including per-step timings, artifact geometry,
hash, and the important 124-frame duration boundary, is recorded in
[MiniMax-H3 BF16 on M4 Max](../benchmarks/minimax-h3-bf16-m4-max.md).

A first-order tail-residual extrapolation was also tested at the accepted
four-cache-step block count. It added no meaningful runtime cost, but the
same-seed 416x256 proxy fell from 0.688 SSIM against the exact trajectory to
0.583 and visibly shifted exposure and composition. The predictor was removed;
maximum mode keeps the more faithful bounded stale residual.

Lower-evaluation ODE math was tested on that same 416x256, 107-frame prompt and
seed before any long render. A 12-point variable-step Adams-Bashforth candidate
completed denoise in 314.742 seconds but reached only 0.667 SSIM and 19.22 dB
PSNR against the 20-point Euler reference. Plain 12-point Euler was both faster
on this small compiled-step geometry (253.063 seconds) and closer to the
reference at 0.694 SSIM and 20.57 dB, so the multistep candidate was removed.

The highest-leverage undistilled screen used one Heun interval: an Euler
predictor plus endpoint correction, or two complete model evaluations. It
finished denoise in 45.548 seconds and the complete proxy in 74.344 seconds,
but decoded into tiled chromatic noise rather than a coherent scene. Its 0.327
SSIM and 13.22 dB PSNR confirm that the released H3 vector field is not straight
enough for a two-evaluation solve. The candidate and its endpoint AdaLN path
were removed. On this checkpoint, reducing the quality schedule to the physical
one- or two-evaluation boundary therefore requires learned distillation rather
than an inference-only integrator swap.

Spatial token reduction and depth pruning were also screened behind temporary
laboratory controls, then removed. A KV-only candidate kept conditioning tokens
exact, retained full-resolution keys and values for nearby video frames, and
2x2-pooled only distant video keys and values. On the 416x256, 107-frame,
12-point proxy, a one-frame-radius arm measured 0.591 SSIM and 16.18 dB PSNR
against dense 12-point Euler. A three-frame-radius arm improved that to 0.617
SSIM and 16.73 dB and remained visually coherent, but still changed composition
and motion. At the true 73,470-row call shape, the corresponding full-block
times were 14.635 and 17.908 seconds versus 25.173 seconds for exact dense
attention.

That speedup is real but insufficient. The automatic 12-point `maximum`
schedule executes 304 transformer blocks under its accepted reuse policy. The
measured sparse-block times therefore project to 74.1 minutes of denoise at the
one-frame radius or 90.7 minutes at the better-quality three-frame radius,
before roughly seven minutes of video/audio decode. Running every fifth block
with dense attention scored only 0.602 SSIM and 16.38 dB, below the simpler
three-frame arm. Combining the one-frame arm with an intentionally more
aggressive 235-block reuse schedule fell to 0.488 SSIM and 15.21 dB and exposed
a visible spatial lattice. None of those variants moved into production.

Pooling 2x2 video queries as well as distant keys and values reduced the proxy
one-evaluation denoise from 24.556 to 21.012 seconds, but the decoded result
collapsed into large spatial blocks (0.444 SSIM, 10.39 dB). Executing alternate
transformer layers was faster still, but both unscaled and 2x-residual-scaled
variants collapsed into a finer latent lattice at one evaluation (0.184 and
0.179 SSIM). These failures bound the remaining inference-only frontier:
near-30-minute output at this geometry needs a checkpoint trained for fewer
evaluations and/or fewer layers, not an untrained token- or depth-pruning
shortcut.
