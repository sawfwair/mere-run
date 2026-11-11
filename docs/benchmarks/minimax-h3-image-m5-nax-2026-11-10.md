# MiniMax-H3-Image M5 affine Q8 acceleration

This follow-up qualifies native packed Q8 computation on an Apple M5 Max
with 40 GPU cores and 128 GiB of unified memory. The fastest observed warm
4K request took 229.53 seconds, at a 26.11 GiB session peak. The
[earlier Q8 report](minimax-h3-image-q8-2026-11-10.md) records the fallback
baseline. Both use the same 52.04 GB Q8 export, camera prompt, seed 42,
4096 × 4096 canvas, six PDD evaluations, and CFG 1. The original checkpoint
remains separate.

## Dispatch regression and repair

The pinned MLX fork restricts quantized NAX matrix acceleration to GPU
architecture generation 18 or later. This M5 Max reports `applegpu_g17s`.
Floating-point matmul and attention already use NAX, while packed Q8 falls
back to slower kernels.

The historical wide-output failure does not reproduce with the current
kernel sources on this machine when acceleration is enabled. The repair
therefore restores dispatch for **affine 8-bit weights only** on generation
17 when MLX reports NAX availability. It leaves other quantization modes,
alignment checks, and TF32 policy unchanged. No shader bodies or vendored
artifacts change. This qualifies a scoped dispatch restoration, rather than
a new matrix kernel or a claim that every historical quantization bug is fixed.

The patch binds to MLX Swift `1e077ba968b291e0237d1ab522b271d433fa1713`
and embedded MLX `e3214809fa5ebd5aac5d6783e8101513b3d9a774`.
[Source preparation](../internals/mlx-host-patches.md) checks both revisions
and exact original, patch, and resulting host-file hashes. The metallib
verifier accepts this exact host change and continues to reject extra
tracked dependency edits, including shader edits.

## Additional native changes

A synthetic BF16 attention sweep at a representative 4K sequence length of
16,528 rows favored 28-head batches with 1024 queries. Its warm median was
275 ms, compared with 299 ms for 56 heads and 640 queries and 427 ms for
eight heads and 640 queries. These are single-layer probes. Every query still attends to every key; this
changes scheduling without introducing sparse attention. The native Q8
runtime selects this default on `applegpu_g17s`. Other architecture and
precision defaults retain the previous schedule. Benchmark flags can override
both limits.

The VAE decoder now batches up to 32 existing overlapping tiles. It preserves
the released FP32 weights, tile geometry, overlap, and assembly order. The
encoder keeps sequential tiling. Independent decoder fixtures compare batches
of 1, 2, 3, and 32, including two input samples and a partial final batch.
Floating-point reordering can change pixels; visual qualification remains
necessary.

## Measurement boundaries

Release timings include all six denoising evaluations, decoding, and PNG
encoding and writing. The first request also loads components and computes
conditioning. Later requests retain the transformer and VAE and reuse the
identical text conditioning. A different prompt needs conditioner execution
and is not timed here. OS peak process footprint covers the whole session;
it is not a per-request MLX counter or a minimum RAM requirement.

The host runs macOS 27.2 on AC power, with other desktop applications open.
Image jobs run sequentially. File caches are not purged. The initial NAX-only
pair overlapped repository compilation; the final configuration runs after
the validation gate finishes. This is one prompt and seed, not a throughput
distribution or general quality benchmark.

## Results

| Native 4096 × 4096 configuration | First request (s) | Warm request (s) | Peak process footprint (GiB) |
| --- | ---: | ---: | ---: |
| Original BF16 baseline | 498.93 | — | 72.12 |
| Packed Q8, generation-17 fallback | 510.76 | — | 26.16 |
| Q8 storage, resident BF16 compute | 289.17 | 246.51 | 43.47 |
| Restored Q8 NAX, 56 heads, 640 queries, sequential decode | 298.52 | **229.53** | 26.11 |
| Restored Q8 NAX, 28 heads, 1024 queries, 32 decode tiles per batch | **272.55** | 241.91 | 26.31 |

The first two historical rows are single CLI runs that include admission.
The resident BF16 and NAX-only rows each have one warm repeat. The last row
reports the median of two warm repeats, **239.60 and 244.23 seconds**. Its first
request took 47% less time than the old packed fallback run, at similar peak
memory. The fastest observed warm request is 229.53 seconds, about 3.28 times
the 70-second target.

The restored Q8 kernel measured 23.18 ms for a synthetic BF16-input projection
with shape `M=4096, K=5376, N=28672`, down from 91.34 ms with the fallback.
Dense BF16 took 20.33 ms with the restored build. These medians exclude the
first evaluation and isolate one projection.

| Warm phase (s) | Restored NAX, sequential decode | Restored NAX, tuned attention and batched decode |
| --- | ---: | ---: |
| QKV preparation | 43.00 | 47.36 |
| Attention | 82.89 | 88.64 |
| Feed-forward and residual work | 91.32 | 97.63 |
| Entire denoising loop | 217.36 | 233.90 |
| All other request work | 12.18 | 8.01 |

The batched configuration spent about four seconds less outside denoising,
where VAE decoding and PNG output run. It did not improve warm end-to-end
latency in this comparison. Several denoising phases varied together between
requests, so the separate sessions do not isolate the effect of attention
scheduling. The synthetic attention improvement did not translate into a
measured warm gain over the 56-head control.

At a representative 16,528 rows, the four dense block projections and full
attention across 50 layers and six steps account for about 6.17 quadrillion
arithmetic operations when a multiply-add counts as two. A 70-second budget
would require roughly 88 trillion operations per second for that work alone,
before the rest of the request. This is a workload estimate. The phase timing
points to feed-forward work and full attention as the largest remaining costs.
Fused normalization, rotary embedding, and projection epilogues merit separate
real-shape tests before another default change.

## Image quality

The three PNGs from the 28-head, batched-decode session are byte-identical.
Compared with the sequential-decode NAX control, the maximum decoded RGB
channel difference is one level out of 255; mean absolute differences are
0.0034, 0.0034, and 0.0034 levels for red, green, and blue. This compares the
combined settings on one prompt and seed.

Whole-frame review and unscaled 512-pixel lens, covering, and tabletop crops
show coherent geometry and clear material detail, with no obvious tile seams
or mosaic corruption. The cup is larger than requested and the camera still
has malformed fine markings. Editing, other prompts, and other devices remain
outside this qualification.

![Native 4096 × 4096 packed Q8 output, downsampled for this page.](/qualification/minimax-h3-image-m5-nax-2026-11-10/optimized.jpg)

![Unscaled lens, covering, and tabletop crops from the native 4K output.](/qualification/minimax-h3-image-m5-nax-2026-11-10/optimized-details.jpg)

## Upstream comparison

The [SGLang recipe](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3-Image.mdx)
currently reports **84.96 seconds** at 4736 × 3552 on one DGX Spark GB10.
It uses native precision, resident components, exact AdaLN precomputation,
and warm conditioning. Its median covers 10 warm requests across two fresh
servers and excludes startup, HTTP, PNG encoding, and saving. Hardware,
aspect ratio, precision, and timing boundaries differ from this native MLX
measurement. This report uses 70 seconds as a stretch goal for the M5 Max.

## Reproduction

```bash
python3 scripts/prepare_mlx.py
swift build -c release --product mere.run
release_dir="$(swift build -c release --show-bin-path)"
"$release_dir/mere.run" model benchmark h3-image --model-root ./image-minimax-h3-q8 \
  --prompt "$PROMPT" --size 4096 --runs 3 \
  --attention-heads 28 --attention-queries 1024 --vae-tile-batch 32 \
  --output ./h3-m5-q8-benchmark --json
```

Use the same prompt recorded in the JSON receipt. Set `MERERUN_H3_IMAGE_PROFILE=1` for phase timings on stderr and wrap the
process with `/usr/bin/time -l` for whole-session peak memory. The contributor
benchmark emits `benchmark.json` and each PNG's SHA-256. It uses packed Q8 unless `--resident-bf16` is passed.
For the NAX control settings, use `--attention-heads 56 --attention-queries 640
--vae-tile-batch 1`.

## Validation

The repository gate passed. Nine affine Q8 regression tests passed on Metal
with TF32 disabled and again with TF32 enabled. Seven image computation tests
and five Q8 packing and loading tests passed on Metal with TF32 disabled.
The shader verifier confirmed that the release used the unchanged vendored
metallib. The [JSON receipt](/qualification/minimax-h3-image-m5-nax-2026-11-10.json)
retains model, binary, source, host patch, shader, and output identities and
per-request timings.
