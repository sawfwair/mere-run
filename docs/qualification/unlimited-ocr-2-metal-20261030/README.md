# Unlimited-OCR-2 Metal qualification, October 30, 2026

For the subsequent arithmetic corrections and repeated document tests, see the
[numerical parity follow-up](./parity/README). Results below preserve the initial
implementation's measurements.

This report tests the original BF16 checkpoint on an Apple M5 Max with 40 GPU
cores and 128 GiB of unified memory. The checkpoint and upstream Python source
are pinned to `c14071b549f4fcab10909f3c4acc6a2edb366947`.
The [pinned upstream tree](https://huggingface.co/baidu/Unlimited-OCR-2/tree/c14071b549f4fcab10909f3c4acc6a2edb366947)
contains the original weights and inference implementation used for comparison.
Both downloaded shard SHA-256 hashes match the pinned Hugging Face LFS objects.
The installed assets total 6,698,712,614 bytes. See [the receipt](./checkpoint-receipt.json).

## Results

Six of eight native cases pass the fixed content checks. The two failures are
receipt base and automatic mode: both omit an `L` in `MODELLING CLAY`, as the
pinned upstream implementation does. Both two-page orders pass. Across ten
input pages, 70 of 72 anchors are recovered, no forbidden anchors appear on the
wrong page, and every request reaches EOS before its output budget. All emitted
block boxes have ordered coordinates within 0–1,000.

| Case | Content checks | Output tokens | Wall seconds | Decode tokens/s | Peak MLX GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| document-control | Pass | 424 | 35.8 | 15.3 | 9.02 |
| receipt-base | Fail: spelling | 495 | 35.3 | 15.8 | 6.52 |
| receipt-auto | Fail: spelling | 495 | 31.5 | 17.9 | 6.52 |
| w9-form-base | Pass | 1,889 | 115.3 | 17.3 | 6.52 |
| w9-form-auto | Pass | 1,942 | 152.1 | 15.3 | 6.58 |
| w9-instructions-base | Pass | 2,623 | 152.5 | 17.9 | 6.52 |
| w9-two-page-base | Pass | 4,661 | 291.3 | 16.7 | 6.59 |
| w9-reversed-base | Pass | 4,654 | 324.1 | 14.9 | 6.59 |

See [summary.json](./summary.json) for per-page checks and stage timings. Raw
records and text are saved under `results/`, named after each case. The opt-in
checkpoint XCTest and independent verifier exit with failure because the two
receipt spelling assertions remain failures. No anchors were relaxed after
seeing the results.

The standard `./scripts/check.sh` gate passes. It skips checkpoint tests unless
explicitly enabled. Thirteen owned numerical tests also pass on Metal, and
three no-repeat tests cover window boundaries, allowlists, and the dotted-line
loop. The documentation build passes.

The public CLI also passes routing and stop checks for a single receipt and a
mixed two-page receipt/invoice document. The original default-math CLI run has
the same receipt body text but slightly different boxes from the library suite.
Setting `MLX_ENABLE_TF32=0` makes the CLI receipt output identical to the library
output after removing the terminal newline. The default-math outputs remain
under `cli-default-math/`; controlled outputs and checks are in
[cli/summary.json](./cli/summary.json). This CLI smoke checks routing anchors;
the separate exact-spelling failure remains in the main suite.

## Inputs and method

The fixed manifest includes a known-text invoice control, a photographed SROIE
receipt, and the first two pages of the IRS March 2024 W-9. The W-9 pages were
rasterized at 144 DPI to 1,224 × 1,584 pixels. The original IRS PDF is frozen in
`sources/fw9.pdf`; the receipt URL is pinned to a dataset revision. Source URLs,
source hashes, raster hashes, expected anchors, and page-order exclusions are
recorded in [inputs-manifest.json](./inputs-manifest.json).

Each request uses temperature zero. Native measurements use the Debug build on
MLX `Device.gpu`, with TF32 disabled. The model remains loaded between requests;
the first request includes loading. Before each request the harness clears the
allocator pool and resets its peak counter. Wall time includes preprocessing,
vision, prefill, and decoding. MLX peak memory measures the allocator, not total
process RSS. These measurements don't establish release-build peak throughput.
Repository and documentation checks overlapped some native cases, adding CPU load;
the upstream and native OCR processes ran sequentially.
See [environment.json](./environment.json) for toolchain and dependency versions.

The eight native cases cover single-page base and automatic crop modes, the
two-page W-9, and the same pages in reversed order. The long single page uses
twelve local tiles in automatic mode. Multi-page automatic crop defaults to
base mode, so the two-page cases explicitly use base. Receipt base and automatic
mode use the same global view because that source image is smaller than 640px.

Content checks use fixed per-page anchors with case, punctuation, and diacritics
normalized. Spelling stays significant: `MODELLING` and `MODELING` differ.
Page-specific forbidden anchors detect swapped or leaked page content. Checks
also require one `<PAGE>` per input, no text before the first marker, and an EOS
stop before the output cap. Box checks only assess coordinate bounds and order;
they don't measure localization accuracy. This is a small smoke suite, not a
character-error-rate benchmark or an exhaustive transcription review.

## Upstream comparison

The reference uses the same original weights, source revision, raster files,
crop mode, greedy decoding, and output budgets where stated. It runs PyTorch
2.13.0 and Transformers 4.52.1 in BF16 on MPS. Two hardcoded `.cuda()` image-mask
transfers are changed to use the input embedding device. Architecture math is
unchanged. Upstream uses its dense fallback on MPS; this doesn't establish
numerical equivalence to CUDA/Triton or between MLX and PyTorch.

With repetition constraints disabled, both implementations transcribe the
receipt's printed `MODELLING CLAY` as `MODELING CLAY`. Both produce 495 tokens.
Their outputs differ in several box coordinates and one printed price:
upstream preserves `9.000`, while native emits `9.00`. The spelling error is
shared upstream; the numeric and coordinate differences remain port/backend
differences and aren't dismissed as parity.

Both unconstrained implementations loop on the W-9 form's dotted LLC line.
Native reaches its 4,096-token cap. A separate upstream diagnostic reaches a
1,024-token cap while repeating that same line. Those caps differ, so this
comparison establishes reproduction of the loop, not token-for-token parity.
See the [native baseline](./baseline-without-ngram/w9-form-base.json) and
[upstream baseline](./upstream/w9-form-base-bfloat16-ngram0.json).

The final native suite uses the upstream serving recipe's 35-token no-repeat
constraint, examining the last 128 generated tokens for one page and 1,024 for
multiple pages. The constraint runs before the existing page-count guard.
No lexical corrections or output cleanup are applied.

With that constraint, upstream completes the base-mode W-9 form at EOS after
1,894 tokens and recovers all seven form anchors. This run is saved as
`upstream/w9-form-base-bfloat16-ngram35.json`.
Native completes at 1,889 tokens with the same anchor outcome. The raw form
outputs also differ in layout grouping: native emits the signature area as a
table, while upstream emits separate text blocks. Upstream merges one lower
column paragraph into the preceding block; native assigns it a separate box.
See [comparison.json](./upstream/comparison.json) and the adjacent raw diffs.
Both base-mode form outputs contain errors outside the anchor checks, including
`Apologies to accounts maintained` where the source says `Applies to accounts
maintained`. Passing the form's anchor checks doesn't mean exact transcription.
The native tiled form and the forward two-page run recover `Applies` correctly
in that phrase.

## Implementation corrections found on Metal

Full-asset runs found three native issues before collecting the baseline:

- SAM spatial row indices used true division. Integer floor division keeps
  relative-position gathers integral on Metal.
- Biased BF16 projections rounded the matrix product before adding bias.
  Computing the biased operation in FP32 and casting once preserves the expected
  rounding. Biased convolutions use the same accumulation rule.
- Sparse attention combined BF16 values with an FP32 router bias, which Metal
  rejected. Attention now evaluates in FP32 and casts its result back to BF16.

Targeted numerical regressions cover these cases on GPU. Scalar fixtures also
cover sparse routing, page coordinates, cached decoder steps, router RMS norms,
and FP32 expert accumulation. These tests catch the observed implementation
errors; they don't prove full-checkpoint numerical identity.

## Reproduce

Pull the original checkpoint with `mere.run model pull vision-ocr-unlimited-2`.
With Python and Poppler installed, prepare the inputs from the repository root:

```bash
python3 docs/qualification/unlimited-ocr-2-metal-20261030/prepare-inputs.py \
  --output .build/qualification/unlimited-ocr-2/inputs
```

Set these environment variables and run
`swift test --filter UnlimitedOCRCheckpointTests`:

```bash
export MERERUN_TEST_UNLIMITED_OCR=1 MERERUN_TEST_MLX_DEVICE=gpu MLX_ENABLE_TF32=0
export MERERUN_TEST_UNLIMITED_OCR_ROOT="$HOME/Library/Application Support/MereRun/models/vision-ocr-unlimited-2"
export MERERUN_TEST_UNLIMITED_OCR_INPUTS="$PWD/.build/qualification/unlimited-ocr-2/inputs"
export MERERUN_TEST_UNLIMITED_OCR_OUTPUT=/tmp/unlimited-ocr-results
```

Use `MERERUN_TEST_UNLIMITED_OCR_CASES` for a comma-separated case subset.
The test intentionally fails when an anchor or stop check fails and saves
every completed raw result before asserting. Independently verify saved records:

```bash
python3 docs/qualification/unlimited-ocr-2-metal-20261030/verify-results.py \
  --manifest .build/qualification/unlimited-ocr-2/inputs/manifest.json \
  --results /tmp/unlimited-ocr-results --output /tmp/unlimited-ocr-summary.json
```

For the reference, create a temporary virtual environment with `torch==2.13.0`,
`torchvision==0.28.0`, `transformers==4.52.1`, `tokenizers==0.21.4`, Pillow,
accelerate, addict, easydict, einops, requests, and safetensors. Run
`prepare-upstream.py --checkpoint "$MERERUN_TEST_UNLIMITED_OCR_ROOT" --output /tmp/uocr/model`,
then `compare-upstream.py --source-root /tmp/uocr/model --inputs "$MERERUN_TEST_UNLIMITED_OCR_INPUTS" --output /tmp/uocr/reference`.
Use `PYTORCH_ENABLE_MPS_FALLBACK=1`, `HF_HUB_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1` for inference. The comparison script defaults to
unconstrained receipt and form diagnostics; add `--ngram-size 35` for the
serving constraint or `--max-tokens 1024` for the short form diagnostic.

## Scope

The native runtime retains document KV rows and router keys. Memory still grows
with document length. Two-page tests don't qualify hundred-page documents,
bounded-memory streaming, the 131,072-token limit, or the SGLang page-window
scheduler. BF16 numerical parity and CUDA reference runs remain separate work.
