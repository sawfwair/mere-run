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
