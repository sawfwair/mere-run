# Unlimited-OCR-2 numerical parity follow-up

This follow-up fixes native BF16 arithmetic differences found after the
[initial Metal qualification](../README). The source and original checkpoint
remain pinned to `c14071b549f4fcab10909f3c4acc6a2edb366947`. The original
document outputs are preserved in the parent directory; corrected outputs are
recorded separately here.

The later [native JPEG follow-up](../jpeg/README) replaces the JPEG input
decoder and reruns receipt comparisons. The measurements below describe the
earlier ImageIO input path and remain preserved as its baseline.
The [execution follow-up](../execution/README) also corrects the historical
JPEG/PNG price attribution: the harness JPEG and canonical PNG CLI runs loaded
different Metal binaries, so that contrast didn't isolate decoding.

## Price, box, and layout results

The original JPEG still produces `9.00` for the item's printed `9.000` price.
The identical-RGB PNG produces `9.000` with the same corrected runtime.
The JPEG misspelling `MODELING` also remains shared with upstream.

| Comparison with upstream | Initial native | Corrected native |
| --- | ---: | ---: |
| Receipt exact block body/type matches | 24 | 24 |
| Receipt mean absolute coordinate delta | 0.63 | 0.75 |
| Receipt maximum coordinate delta | 6 | 6 |
| W-9 exact block body/type matches | 24 | 25 |
| W-9 mean absolute coordinate delta | 0.33 | 0.21 |
| W-9 maximum coordinate delta | 4 | 3 |

Coordinates use the model's 0–1,000 scale. Geometry comparisons include only
unique exact body/type pairs; repeated price strings are excluded because their
correspondence is ambiguous when one price differs. The matched sets can change
between runs, so these numbers describe agreement with upstream, not accuracy
against annotated boxes. See [layout-comparison.json](./layout-comparison.json)
and the adjacent before/after raw diffs.

The corrected W-9 signature area uses text blocks, as upstream does, rather
than the initial native table. It still merges two signature labels that
upstream separates. The main table's row spans also remain different. Receipt
coordinates don't improve overall. Full price, coordinate, and structural
parity with the upstream MPS run isn't established.

The W-9 comparison uses the same 35-token repetition constraint on both sides.
The saved receipt reference used size zero. Checking its generated token trace
shows that enabling size 35 with a 128-token window would never ban the chosen
greedy token; that constraint doesn't explain the receipt differences.

## Repeated Metal document suite

The corrected runtime passes six of the same eight fixed cases. Across ten
input pages, 70 of 72 anchors are recovered. The two failures remain the shared
receipt spelling error. Both multi-page orders pass all anchors and separation
checks, every request stops before its token cap, and all emitted box
coordinates are ordered and within 0–1,000.

| Case | Content checks | Output tokens | Wall seconds | Peak MLX GiB |
| --- | --- | ---: | ---: | ---: |
| document-control | Pass | 424 | 31.4 | 9.02 |
| receipt-base | Fail: spelling | 495 | 32.0 | 6.60 |
| receipt-auto | Fail: spelling | 495 | 30.0 | 6.60 |
| w9-form-base | Pass | 1,878 | 142.5 | 6.60 |
| w9-form-auto | Pass | 1,958 | 171.4 | 6.66 |
| w9-instructions-base | Pass | 2,609 | 146.2 | 6.60 |
| w9-two-page-base | Pass | 4,641 | 326.3 | 6.60 |
| w9-reversed-base | Pass | 4,654 | 323.9 | 6.60 |

These are Debug-build measurements on the same M5 Max with 128 GiB and TF32
disabled. Repository checks added CPU load during parts of the run; timings
aren't an isolated throughput benchmark. The peak is MLX allocator memory,
not total process memory. See [summary.json](./summary.json) for stage timings,
per-page checks, and the unchanged acceptance criteria. Raw records and text
are in `results/` with the case names above.

The asset-dependent XCTest and independent verifier exit with failure for the
two receipt assertions. The standard `./scripts/check.sh` gate passes, including
the new odd-padding regression. All 17 owned numerical tests pass on GPU.
The BF16 prefill fixture uses a stated input-scale error bound against its
independent FP32 oracle; cached-decode and representable activation goldens
retain exact checks. The strict identical-PNG preprocessing probe also passes.

## Native corrections

- Evaluate SAM GELU and decoder SiLU in FP32 before returning BF16, matching
  PyTorch's intrinsic operations. Direct BF16 expression evaluation rounded
  intermediate values too early.
- Keep QuickGELU's `1.702` scalar in FP32, round its product to BF16, evaluate
  sigmoid in FP32, then round before the final BF16 product. The old scalar
  conversion alone changed about 12.6% of the captured CLIP activation values.
  SAM neck epsilon addition follows the same scalar-promotion rule.
- Construct RoPE inverse frequencies on CPU with FP32 reciprocal powers, as
  upstream does. The previous `exp(log(...))` expression changed rotary
  rounding. Isolated Q/K rotations now match the captured upstream values
  exactly for the receipt prompt.
- Use BF16 Q/K/V and router bias for prefill, and FP32 operands and bias for
  incremental decode. The original Metal mask workaround promoted both paths,
  losing upstream's prefill dtype boundary. An independent upstream fixture
  tests prefill and cached decode separately.
- Apply positional-grid antialiased bicubic resampling horizontally, then
  vertically, with the upstream coefficient operation order. PyTorch's
  [separable interpolation implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/cpu/UpSampleKernel.cpp)
  specifies that pass order.
- Center odd letterbox gaps with Pillow's ties-to-even rounding, rather than
  integer floor division. A new narrow-image fixture catches the one-pixel
  shift. The frozen document inputs have unchanged offsets under this fix.
  Those offsets are recorded in [letterbox-centering.json](./letterbox-centering.json).

The 17 numerical regression tests pass on Metal. Their small fixtures run
without Python; `generate_pytorch_precision.py` records how the independent
PyTorch precision fixture was generated. Production still runs Swift and MLX.

## Tensor comparison

The reference capture uses the original upstream model on MPS with the same two
device-transfer adaptations as the initial qualification. Diagnostic hooks
observe tensors without changing arithmetic. Each isolated probe receives its
exact upstream input, so earlier vision drift cannot hide an operation error.

| Isolated operation | Native versus upstream result |
| --- | --- |
| SAM GELU | RMSE approximately 3.6 × 10⁻⁹, down from 9.2 × 10⁻⁴ |
| CLIP QuickGELU | Exact match |
| Decoder gated SiLU product | Exact match |
| Decoder first-block input and post-attention RMS norms | Exact match |
| Decoder first-block Q/K/V projections | Exact match |
| Decoder first-block Q/K rotations | Exact match |

With identical upstream image tensors, projected visual-feature RMSE decreases
from 0.00783 before the corrections to approximately 0.00422. This doesn't
establish full-model numerical identity. The original BF16 vision-attention
path remains in use. See [tensor-metrics.json](./tensor-metrics.json) for full
and isolated per-block measurements, including positional-grid probes.

## Differences outside those native operations

ImageIO and Pillow decode the receipt JPEG differently: the largest RGB sample
delta is 41 on the 0–255 scale, with a mean absolute delta of about 0.692.
After resizing and BF16 normalization, the largest delta is 0.28125. Saving
Pillow's decoded RGB samples as a PNG removes that variable. Native PNG decoding,
resizing, padding, and normalization then match upstream exactly, with strict
zero-difference assertions. See the [JPEG](./receipt.jpeg.pixels.json) and
[identical-RGB PNG](./receipt-canonical.png.pixels.json) measurements.

The difference is already present in ImageIO's raw decoded provider bytes.
This image takes the direct 32-bit RGB byte-copy path; it doesn't pass through
a CGContext color conversion. See [jpeg-decoder-origin.json](./jpeg-decoder-origin.json).
With the final native runtime, the identical-RGB PNG preserves the printed
`9.000` price and stops normally. Its [raw output](./canonical.stdout.txt) and
[run record](./canonical.json) are retained separately from the original JPEG.

Attention kernels also differ independently of the port. A fixed 64-token,
10-head, 128-dimension causal probe uses the same captured Q/K/V in every run.
Native FP32 attention has RMSE about 5.7 × 10⁻⁷ against PyTorch CPU FP32 and
1.2 × 10⁻⁶ against PyTorch MPS FP32, after BF16 output rounding. Against the
PyTorch MPS BF16 kernel, the RMSE is about 2.7 × 10⁻⁴. PyTorch CPU BF16 and MPS
BF16 also disagree with each other on those same inputs. These are measured
kernel differences, not an assumption that every residual output difference
is harmless. See [backend-comparison.json](./backend-comparison.json).

The floating-point differences can change greedy coordinate tokens or layout
grouping. Box coordinates and HTML are kept as generated; these checks don't
measure geometric accuracy against annotated boxes or establish exact
transcription of the complete documents.

## Reproduce the tensor diagnosis

Prepare the checkpoint and upstream package using the parent report's scripts.
Capture the receipt reference tensors with the temporary PyTorch environment:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 docs/qualification/unlimited-ocr-2-metal-20261030/trace-upstream.py \
  --source-root /tmp/uocr/model \
  --image .build/qualification/unlimited-ocr-2/inputs/receipt.jpeg \
  --output /tmp/uocr/receipt-upstream.safetensors
```

Set `MERERUN_TEST_MLX_DEVICE=gpu`, `MLX_ENABLE_TF32=0`, the checkpoint root in
`MERERUN_TEST_UNLIMITED_OCR_ROOT`, and the capture path in
`MERERUN_TEST_UNLIMITED_OCR_TRACE`. Then run
`swift test --filter UnlimitedOCRParityTests`. Set
`MERERUN_TEST_UNLIMITED_OCR_PARITY_OUTPUT` to choose the output metrics JSON.
The capture contains checkpoint-derived tensors and remains outside Git.

For pixel diagnosis, set `MERERUN_TEST_UNLIMITED_OCR_IMAGE` and run
`swift test --filter UnlimitedOCRImageParityTests`. Use a lossless PNG made
from Pillow's decoded RGB samples and set
`MERERUN_TEST_UNLIMITED_OCR_EXPECT_IDENTICAL_PIXELS=1` to enforce exact equality.
The ordinary repository gate skips these asset-dependent diagnostics.
