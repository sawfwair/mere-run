# LightOnOCR 3 native checkpoint qualification — 2026-10-08

The released 1B, 0.8B, and 4B checkpoints ran through native Swift/MLX on Metal.
Twenty-eight of thirty bounded output cases passed. **1B remains the default**:
it passed every checked case with lower memory use and latency than 4B here.
The 0.8B variant remains available, with a demonstrated handwriting limitation.

## Scope and results

Five inputs ran in both plain and grounding modes for each checkpoint: a public
receipt scan, a public handwritten-word crop, a generated English/French
two-column document with an invoice table, a generated labeled bar chart, and
a blank page. Temperature was zero; the output cap was 1,024 tokens, or 64 for
blank pages. All thirty runs completed below their token caps. Blank pages
returned empty text in both modes for every model.

| Variant | Passing output cases | Peak MLX allocations | Receipt plain / grounding | Document plain / grounding |
| --- | --- | --- | --- | --- |
| 1B | 10 / 10 | 3.57 GiB | 11.39 / 12.58 s | 13.58 / 18.47 s |
| 0.8B | 8 / 10 | 2.58 GiB | 14.86 / 14.44 s | 10.50 / 18.69 s |
| 4B | 10 / 10 | 9.41 GiB | 37.48 / 37.55 s | 25.29 / 24.96 s |

Timings are one sequential native debug-build run.
The first receipt/plain run includes model initialization; later runs reuse the
loaded model. Disk and Metal caches were already warm. These are bounded smoke
measurements, not a comparative throughput benchmark. MLX peak memory resets
before each call and measures MLX allocations, not total system memory or RSS.

The independent [output verifier source](https://github.com/sawfwair/mere-run/blob/main/docs/benchmarks/lightonocr3-2026-10-08/verify-results.py) checks
selected text anchors, exact invoice/chart cell rows, empty blank-page output,
termination, normalized box bounds, and known table/chart region overlap.
The full [summary](lightonocr3-2026-10-08/summary.json) and all thirty
[raw output records](lightonocr3-2026-10-08/results/1B-document-grounding.json)
are saved alongside this report.

| Grounded region | 1B IoU | 0.8B IoU | 4B IoU |
| --- | --- | --- | --- |
| Invoice table | 0.947 | 0.982 | 0.973 |
| Chart | 0.960 | 0.974 | 0.971 |

All checked invoice rows retain their item, quantity, and amount. All chart rows
retain Jan=10, Feb=25, and Mar=40. The reference bounds come from fixture drawing
coordinates; the pass threshold is intersection-over-union of at least 0.75.
This checks two region types, not every predicted text box.

## Observed limitations

The 0.8B model transcribes the handwritten `industrie` as `indus frie` in both
modes and emits no grounding box for that crop. The opt-in checkpoint test and
independent verifier therefore **exit with failure** for the full matrix; this
quality failure is preserved rather than exempted. The 1B and 4B models return
`industrie`, with a full-crop text box in grounding mode.

Passing selected receipt anchors does not mean exact full transcription.
The Qwen variants emit `RF` for the visible `KF` item prefix and split some
receipt labels; 4B plain also changes the last barcode digit. Receipt HTML
header/cell alignment is imperfect. Raw results retain these errors. No
structured receipt correctness claim follows from the anchor checks.

A bounded [upstream Transformers spot check](lightonocr3-2026-10-08/upstream-handwriting.json)
on the same pinned 0.8B weights and crop also failed both modes: it returned
`indus trie`, with no grounding box. That diagnostic used Torch 2.14.1,
Transformers 5.5.4, CPU FP32, eager attention, greedy decoding, and KV caching.
The miss is therefore not unique to native inference. The different text and
precision/backend mean this does not establish numerical parity.

No upstream numerical parity, benchmark-scale accuracy, difficult handwriting
corpus, long-document throughput, maximum-resolution stress test, or old-versus-new
model comparison was performed. PDF rendering and multi-page document processing
are outside this image-input matrix.

## Checkpoints and inputs

Original BF16 safetensors were loaded directly, without conversion or weight
quantization. Each downloaded weight file's SHA-256 matches its published LFS
hash. Complete file receipts are saved for
[1B](lightonocr3-2026-10-08/1B-download-receipt.json),
[0.8B](lightonocr3-2026-10-08/0.8B-download-receipt.json), and
[4B](lightonocr3-2026-10-08/4B-download-receipt.json).

| Upstream checkpoint | Immutable revision |
| --- | --- |
| `lightonai/LightOnOCR-3-1B` | `b9a2b4c17f1eee9f29058d716b66b5f8e7d8db86` |
| `lightonai/LightOnOCR-3-0.8B` | `be8cee5d200b80218cb2865a5deeec1fe6e25f52` |
| `lightonai/LightOnOCR-3-4B` | `22a24f41d5bf21e307a427c311dc90e30616e572` |

The [input manifest](lightonocr3-2026-10-08/inputs/manifest.json) records hashes,
selected anchors, expected rows, and region bounds. Public scans come from
[`hf-internal-testing/fixtures_ocr`](https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/tree/28fe12cdf7816b5dde94e22051b2ec8dc74267b7)
at the pinned revision. Their bytes are fetched by the reproduction helper;
the generated document, chart, and blank images are committed.

## Reproduce

Use the original checkpoints at the revisions above in external directories
named `1B`, `0.8B`, and `4B`. The fixture helper requires Python with Pillow and
the macOS Arial fonts; it verifies the hashes of the two downloaded public scans.

```bash
python3 docs/benchmarks/lightonocr3-2026-10-08/make-fixtures.py /tmp/lightonocr3-inputs

MERERUN_TEST_MLX_DEVICE=gpu \
MERERUN_TEST_LIGHTONOCR3=1 \
MERERUN_TEST_LIGHTONOCR3_ROOT=/path/to/checkpoints \
MERERUN_TEST_LIGHTONOCR3_INPUTS=/tmp/lightonocr3-inputs \
MERERUN_TEST_LIGHTONOCR3_OUTPUT=/tmp/lightonocr3-results \
swift test --filter LightOnOCR3CheckpointTests

python3 docs/benchmarks/lightonocr3-2026-10-08/verify-results.py \
  /tmp/lightonocr3-inputs/manifest.json /tmp/lightonocr3-results \
  --summary /tmp/lightonocr3-summary.json
```

`MERERUN_TEST_LIGHTONOCR3_SIZES=1B,4B` or
`MERERUN_TEST_LIGHTONOCR3_CASES=receipt,document` can narrow a checkpoint run.
Normal repository tests skip this external-assets qualification.

The first CLI attempt exposed a generic 16 GiB admission floor for every OCR
run. The managed 1B and 0.8B single-backend plans now use the existing small
6 GiB admission class, backed by the bounded measurements above. Comparisons,
4B, other backends, and unknown local directories retain their existing floors.
All six public CLI document runs (three sizes × two modes) passed, and their
output matched the saved native records exactly. The temporary override roots
lacked managed source metadata, so the resolver fetched pinned Hub snapshots;
this also exercised auto-download. The original installed OCR model was preserved.
[CLI records](lightonocr3-2026-10-08/cli-checks.json) include output hashes and
available process measurements. These CLI timings include resolution, and first
runs include downloads; they are separate from the native timings above.
The 4B CLI peaked at 10.47 GiB RSS and 18.96 GiB process footprint.
Minimum-memory configurations were not qualified.

## Local repository gate

`./scripts/check.sh` passed after the runtime and admission changes: 5,343 XCTest
cases (441 skipped, zero failures), 192 Swift Testing cases, strict Swift lint
with zero violations across 2,465 files, package/architecture policy checks,
CLI help sweeps, and hygiene scans. Eight focused admission regression tests
also passed. The source gate skips the opt-in external-checkpoint test; its
separate full-matrix result remains failed because of the two 0.8B quality cases.
All five inputs and their annotated manifests reproduced exactly with the helper.

To repeat the separate upstream handwriting diagnostic:

```bash
uv run --no-project --with transformers==5.5.4 --with torch==2.14.1 \
  --with torchvision --with Pillow --with accelerate \
  docs/benchmarks/lightonocr3-2026-10-08/upstream-handwriting.py \
  /path/to/checkpoints/0.8B /tmp/lightonocr3-inputs/handwriting.jpeg \
  /tmp/lightonocr3-upstream-word.json
```
