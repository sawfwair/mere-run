# Unlimited-OCR-2 first layout-token investigation

The first receipt output split occurs at zero-based token 20, in the first
text box's left coordinate: native emits `104`, and upstream emits `106`.
This follow-up traces that low-margin decision with identical token histories,
upstream activation substitutions, and original BF16 checkpoint weights.
It doesn't resolve full receipt parity or establish another native equation
or dtype error that warrants a production correction.

## First layout decision

Both runs share the preceding text:

```text
<PAGE><BLOCK>image[131, 0, 862, 99]<FORK>
<BLOCK>text[
```

The upstream model and checkpoint remain pinned to
`c14071b549f4fcab10909f3c4acc6a2edb366947`. The reference uses PyTorch 2.13.0
on MPS, BF16 weights, greedy selection, and the Pillow receipt input. Native
uses MLX on Metal with TF32 disabled. The JPEG samples and normalized BF16
input remain identical. These are agreement checks against upstream, without
annotated ground-truth boxes.

| Decoder and image features | `104` score | `105` score | `106` score | Choice |
| --- | ---: | ---: | ---: | --- |
| Upstream, upstream features | 31.0 | 31.0 | 31.125 | `106` |
| Native, upstream features | 31.125 | 31.125 | 31.125 | `105` |
| Native, native features | 31.0 | 30.875 | 31.0 | `104` |

Token IDs are 7163 (`104`), 6992 (`105`), and 7336 (`106`). Ordinary argmax
selects the lowest ID at an exact tie. The shared history is replayed through
the target decision; the target token remains an unforced prediction.

The diagnostic substitutes each upstream decoder block output at every
replayed step, including prefill. With either image-feature source, replacing
block 0 yields `105`, while replacing block 1 first restores `106`. Subsequent
boundaries are nonmonotonic: some restore the reference margin, and others
produce ties. This doesn't isolate a unique block-1 implementation defect.
Replacing the last block or final normalized state recovers all three
reference candidate scores exactly. The native final normalization and output
head therefore reproduce this decision on the upstream state.

[`boundaries.json`](./boundaries.json) records every substitution, candidate
score, and earliest token disagreement. The corresponding upstream capture
metadata is in [`upstream.json`](./upstream.json).

## Identical-input numerical probes

Each decoder block is tested on the upstream hidden input, upstream prior KV
and router cache, and identical positions. At token 20, layer 0's two norms
and attention output match exactly. Its feed-forward output differs in 26
of 1,280 cells; the residual output differs in six. Isolating its operations
finds one differing cell in each gate/up projection. SiLU and the down
projection match exactly on identical inputs.

The gate projection's differing value is approximately 0.002693178394 before
BF16 rounding. CPU FP64 accumulation rounds to native's 0.002700805664;
upstream instead returns 0.002685546875. The up projection's value is
approximately 0.096435545085. CPU FP64 rounds to upstream's 0.0966796875;
native returns 0.09619140625. Their proximity to BF16 rounding midpoints makes
FP32 accumulation order significant. Neither backend consistently supplies
the correctly rounded result on these probes.

Prefill has an additional decoder difference. On upstream embeddings, the
first input norm and isolated Q/K/V projections match exactly; isolated gate
and up projections on upstream's second norm also match. The complete first
attention path differs in 258,094 of 357,120 output cells, with RMSE 0.0003884.
Native uses a gathered KV set for each query, while upstream uses dense masked
prefill, so the attention kernels receive different shapes and reduction
orders. This probe doesn't isolate a further equation or dtype defect within
that path. Restoring the first coordinate with the upstream patch tensor
doesn't eliminate these additional decoder differences.

A separate sweep checks every finite BF16 input in [−32, 32], totaling 33,794
values. SiLU and sigmoid match upstream exactly. GELU differs at −5.5 and
−5.53125 by at most 1.65e−7, reflecting the backends' error-function tails.
This doesn't identify a SiLU precision-boundary error behind the decoder split.

The first numerical difference in the vision path is SAM's patch convolution:
94 of 3,145,728 BF16 outputs differ, with maximum difference 0.0078125.
At the first differing cell, `[0, 0, 33, 751]`, CPU FP64 accumulation favors
upstream. Other sampled cells favor native. The convolution preserves FP32
accumulation and bias addition before BF16 output rounding; its remaining
differences arise near cancellation and rounding boundaries.

That first convolution drift changes the layout prefix causally. Replacing
only its output with upstream's patch tensor restores `106` at token 20;
every later vision and decoder operation remains native. No earlier token
changes. This restores the first coordinate, without establishing full-run
parity.

| Patch calculation supplied to native vision | Differences from upstream patch | First coordinate |
| --- | ---: | --- |
| Original native convolution | 94 | `104` |
| Upstream patch output | 0 | `106` |
| Native FP32 channel-major matrix projection | 483 | `105` |
| Native FP32 spatial-major matrix projection | 94 | `104` |

Both matrix probes add bias before BF16 output rounding. Merely matching the
upstream tensor's channel-major storage order when flattening patches makes
agreement worse. The
spatial-major projection reproduces the native convolution. Neither probe
provides an evidence-supported production fix, so the production equations
and precision remain unchanged from the prior DASH scalar correction.

[`isolated-blocks.json`](./isolated-blocks.json) contains same-input metrics.
[`rounding.json`](./rounding.json) preserves the actual FP64 dot products and
sampled backend results. These probes distinguish accumulation differences
from equation and dtype mistakes; they don't justify changing model precision
or token tie-breaking to favor this receipt.

## Repeated receipt comparison

JPEG base, JPEG auto, and canonical PNG base were repeated through both CLI
and XCTest with greedy decoding, repetition size 35/window 128, and a
1,024-token cap. All six outputs are byte-identical, contain 495 generated
tokens, and stop before the cap. They also match the prior price-investigation
outputs byte for byte. Native still emits `9.00`.
The fresh independent upstream run emits `9.000` and reproduces all 495 token
IDs from the prior reference exactly. Its original-source hash and full output
are recorded in [`upstream-rerun/receipt-base-bfloat16-ngram35.json`](./upstream-rerun/receipt-base-bfloat16-ngram35.json).

Every run uses verified Metal library SHA-256
`64431ed485178f63a8dc870c8231ba021090c3356218c607a726821a7b6abb82`.
The kernel verification script passes against the pinned source and binary
digest. The same verified artifact is loaded beside the XCTest image.
[`results/summary.json`](./results/summary.json) records hashes, comparisons,
and raw-output locations. Timings include model loading and aren't controlled
throughput measurements.

Agreement with upstream remains 24/25 exact block body/type matches. The same
17 unique body/type pairs have mean absolute coordinate difference 1.176 and
maximum difference 4. Repeated bodies are excluded from geometry comparisons.
The shared printed `MODELLING CLAY` spelling failure remains. The unrelated
eight-case document suite wasn't repeated.

The standard gate passes: 5,675 XCTest cases, including 477 skips,
plus 197 Swift Testing cases, with zero failures. All 18 owned numerical tests
also pass explicitly on GPU, as do the external-asset boundary and prefill
diagnostics. The documentation build passes. Details and source/capture hashes
are in [`validation.json`](./validation.json) and [`method.json`](./method.json).

## Reproduce

Use the upstream environment and pinned source described in the parent report.
Capture all block states through the first layout decision:

```bash
python docs/qualification/unlimited-ocr-2-metal-20261030/price/trace-price.py \
  --source-root /path/to/upstream/model --image /path/to/receipt.jpeg \
  --reference /path/to/upstream-receipt.json --step 20 \
  --capture-all-steps --capture-leaves --output /tmp/unlimited-ocr-layout

python docs/qualification/unlimited-ocr-2-metal-20261030/layout/probe-rounding.py \
  --source-root /path/to/upstream/model --capture /tmp/unlimited-ocr-layout --prepare
```

Set `MERERUN_TEST_UNLIMITED_OCR_LAYOUT` to the capture directory and
`MERERUN_TEST_UNLIMITED_OCR_ROOT` to the original checkpoint. Run
`UnlimitedOCRLayoutDiagnostics.testFirstLayoutSplit` with
`MERERUN_TEST_MLX_DEVICE=gpu` and `MLX_ENABLE_TF32=0`.
`MERERUN_TEST_UNLIMITED_OCR_PRICE_NATIVE_FEATURES` optionally supplies native
page features. `MERERUN_TEST_UNLIMITED_OCR_LAYOUT_PREFILL` supplies the earlier
prefill capture, including pixels and SAM/decoder intermediates.
`MERERUN_TEST_UNLIMITED_OCR_LAYOUT_PATCH_SUBSTITUTIONS=1` enables vision-patch
substitutions; `MERERUN_TEST_UNLIMITED_OCR_LAYOUT_PROBES_ONLY=1` skips decoder
boundary replay. Run `probe-rounding.py` again without `--prepare` to produce
the FP64 comparison, optionally passing `--prefill` with that earlier capture.

Large activation and checkpoint files remain external. Production inference
requires neither Python nor diagnostic captures. Activation substitutions are
optional package-scoped test hooks and default to no substitution.
