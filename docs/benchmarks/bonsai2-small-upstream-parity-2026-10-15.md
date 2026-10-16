# Bonsai 2 Small upstream parity

Follow-up: the [October 16 precise-intrinsic correction](./bonsai2-small-upstream-parity-2026-10-16.md) resolves the strict 2B/4B vision failures. This report preserves the earlier measurements.

This check compares the released 2B, 4B, and 9B checkpoints with the native Swift/MLX implementation. The independent reference uses each pack's unmodified `vision_artifact.py` loader and its pinned Python dependencies. Python is a validation reference; product inference uses Swift/MLX.

Verification is complete. The corrected implementation matches all 930 teacher-forced greedy decisions across 18 cases. Text passes for all three checkpoints. The 9B pack passes every frozen gate; 2B and 4B fail strict vision parity. A 2B chart trace isolates its remaining drift to rotary-frequency rounding; the other failed cases remain unqualified. These failures remained open in that measurement; tolerances were not changed.

## Reference and inputs

The checkpoints and fixture bytes are the same as the [initial quality and throughput report](./bonsai2-small-2026-10-15.md). That report measures the earlier implementation and is retained as historical evidence.

| Pack | Upstream revision |
| --- | --- |
| [2B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-mlx-2bit/tree/1c6cbae21f8c38830ce337972b46fdae7b18eaaa) | `1c6cbae21f8c38830ce337972b46fdae7b18eaaa` |
| [4B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-mlx-2bit/tree/dff2caaa23fa8706164c44c10f1b58feb405f76a) | `dff2caaa23fa8706164c44c10f1b58feb405f76a` |
| [9B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-mlx-2bit/tree/f4e28a59531731c41b6dfac88cf271bd58f0a287) | `f4e28a59531731c41b6dfac88cf271bd58f0a287` |

Reference dependencies follow the published `runtime/requirements.txt`: MLX 0.32.0, mlx-lm 0.31.3, mlx-vlm 0.6.3, and Transformers 5.5.0. The resolved environment uses Pillow 12.3.0 and NumPy 2.5.4. Native MLX core is 0.32.1. Results therefore measure agreement across these pinned implementations, without requiring bitwise identity between the MLX core versions.

All model storage, the isolated Python environment, raw tensors, logs, source receipts, and comparison reports are on T7 under `/Volumes/T7/mere-run-qualification/bonsai2-small-2026-10-15/parity`. Reference exports verify checkpoint SHA-256 identities and the runtime files' upstream Git blobs before execution. Inference uses the normal shared machine admission coordinator, serially.

## Method

Each model receives two text prompts (thinking arithmetic and non-thinking JSON) and four image prompts (shapes, receipt, chart, and photograph). System text, message order, thinking flags, image bytes, and token budgets are frozen before comparison.

The upstream reference selects up to 64 greedy continuation tokens. Native execution independently renders and tokenizes the prompt, preprocesses the image, runs the vision tower, inserts image features, builds mRoPE positions, prefills, and decodes. It then follows the exact upstream token prefix at every compared position. Full-vocabulary logits are saved; matching generated prose is insufficient.

Both implementations prefill in 512-token chunks within a 16,384-token context. Image chunks receive explicit slices of the upstream model's own full-prompt mRoPE positions. Automatic scalar cached-decode positions cannot represent spatial image tokens inside later prefill chunks. This supplies the model's position API without modifying its neural implementation.

| Gate | Frozen requirement |
| --- | --- |
| Prompt token IDs | Exact |
| Image grid and mRoPE positions | Exact |
| Normalized image patches | Maximum absolute delta at most `1e-6` |
| Image embeddings | Relative L2 delta at most `0.01` |
| Conditional token probabilities | Maximum absolute delta at most `0.01` |
| Greedy argmax under the common prefix | Exact at every exported position |

Every gate is reported separately. Any failure prevents an overall parity pass. This bounded teacher-forced comparison does not establish sampled-output equivalence, broad quality, maximum-context accuracy, or performance on other hardware.

## Measured comparisons

The [machine-readable summary](./bonsai2-small-upstream-parity-2026-10-15-summary.json) records every case and gate, tensor hashes, source identities, and exact native executable receipts.

The original implementation had large vision errors. Bicubic preprocessing, ordinary RMS scales, checkpoint-dtype position interpolation, Conv3d projection, and the upstream reciprocal/power frequency formula substantially reduce them. All intermediate variants remain in the summary and T7 evidence.

| Model | Original max image relative L2 | Final max image relative L2 | Final max probability delta | Final argmax | Overall |
| --- | ---: | ---: | ---: | ---: | --- |
| 2B | 1.902417 | 0.023889 | 0.012618 | 331/331 | Fail |
| 4B | 1.861957 | 0.018809 | 0.006331 | 293/293 | Fail |
| 9B | 1.945880 | 0.005231 | 0.003248 | 306/306 | Pass |

2B fails the image-embedding gate for shapes, receipt, and chart, plus the probability gate for shapes (0.012618 versus the 0.01 limit). Its photograph passes. 4B fails only the shapes image-embedding gate. Prompt IDs, grids, mRoPE positions, normalized patches, and greedy decisions pass for every case. After conversion to checkpoint FP16, all 12 normalized image inputs are bitwise identical to upstream.

### Remaining rotary-frequency drift

A chart stage trace for 2B has bitwise-equal patch embeddings, learned-position embeddings, and summed vision inputs. The raw rotary frequencies differ by at most `4.76837158203125e-7`. That small difference grows through the FP16 vision blocks to a merged embedding relative L2 of 0.022253. The manual native stage replay exactly reproduces its production vision tower.

In a separate controlled ablation, the unmodified upstream vision blocks receive only the native rotary frequencies. Their traced blocks and merged image embeddings then match native bitwise. This isolates the remaining chart discrepancy to the frequency input; this altered-input control is not counted as an upstream parity pass.

The native formula now follows upstream's FP32 positive power followed by reciprocal, rather than CPU `powf(theta, -exponent)`. CPU synthetic frequency fixtures match exactly. Metal still differs slightly. A separately installed Python MLX 0.32.1 control produces exactly the same upstream frequencies, first block, and merged embeddings as the pinned MLX 0.32.0 reference. Thus the MLX version number alone does not explain the difference. The native bundled Metal library uses Metal 3.2 with a macOS 15 target; the Python wheel uses a different build. Compiler/kernel rounding is the remaining suspect, not a confirmed compiler diagnosis. No vendor artifact was changed.

### Corrected release smoke

The corrected release SHA-256 is `7a9ffafb6ae212d660aa2f68eb744f234bfd73d469287d3c6f59dd02c407cf2f`. One fresh-process chart request per model uses greedy sampling, thinking disabled, a 256-token budget, the same system/image/prompt, and no output grammar. All three return the correct values: PINE 9, CEDAR 4, difference 5. All three wrap the JSON in Markdown fences, as the upstream reference does; semantic scoring passes and strict JSON scoring fails.

This smoke does not rerun the original 16-case sampled quality corpus or resident throughput benchmark. Those measurements remain evidence for the earlier binary only. The corrected release, executable receipts, chart stdout/stderr, and full numerical comparisons are preserved on T7.

## Corrections examined

- The original Small-model processor resized with nearest-neighbor sampling. The published processor uses Pillow bicubic RGB resampling. The unchanged receipt and chart fixtures require resizing; the shapes and photograph do not. A CPU source probe measured normalized component differences up to 1.2784 and 0.9882 for the receipt and chart. Small-model preprocessing now uses the existing Pillow-compatible bicubic implementation and explicit FP32 rescaling. An independently exported synthetic image fixture checks patch values and ordering.
- Prism checkpoints contain ordinary FP16 RMSNorm scales. Subtracting one and later adding it in FP16 changes 17,871 of 103,424 inspected norm values for 2B, 36,670 of 170,496 for 4B, and 72,470 of 270,336 for 9B. The largest change is `0.00048828125`. Prism norms now use ordinary scales directly. Regression tests compare loaded bits and forward output with the original scales.
- Upstream vision learned-position interpolation uses the embedding dtype. FP32 interpolation rounds only after summing its four terms; upstream rounds each product and addition in FP16. Prism now selects checkpoint-dtype interpolation; independent FP16 position fixtures cover the arithmetic.

- Native patch embedding flattened the Conv3d kernel into a matrix multiply. The operations differ in FP16 accumulation order. Bounded real-weight CPU probes showed maximum differences of 0.00390625 (2B) and 0.0078125 (4B and 9B); native Conv3d matched the independent reference exactly for all three. Prism now uses the published Conv3d operation.

- Vision rotary frequencies now use the upstream FP32 power/reciprocal expression. Independent CPU fixtures check the formula; GPU fixtures allow at most `1e-6` for this isolated operation. This component tolerance does not change the frozen full-model gates above.

The preserved baseline test executable and baseline source hashes identify the original implementation. Baseline exports and corrected exports are retained separately. Earlier quality scores and throughput samples must not be attributed to a corrected executable.

## Local validation

`./scripts/check.sh` passed: 5,472 XCTest cases, 454 skips, zero failures, plus 193 Swift Testing tests. The release build, independent Metal rotary component fixture, documentation example checks, docs build, and whitespace checks passed. The opt-in real-checkpoint jobs and controlled ablations ran separately under standard shared admission. All 50 original checkpoint source files still match their source SHA-256 receipts after execution. The tested release and its preserved T7 copy match the recorded binary hash. No hosted CI or PR publication was run.

## Reproduce

Prepare the exact pinned snapshots and their source receipts on T7, copy the corresponding published runtime sources into `metadata/SIZE/runtime`, and install the published requirements into the parity environment. Keep the frozen `cases.json` and `protocol.json` with the evidence.

```bash
export MERERUN_TEST_MLX_DEVICE=gpu
export MERERUN_TEST_BONSAI_PARITY_ROOT=/Volumes/T7/mere-run-qualification/bonsai2-small-2026-10-15/parity
export MERERUN_Q35_PREFILL_CHUNK_TOKENS=512
export MERERUN_TEST_BONSAI_PARITY_SIZE=2
MERERUN_TEST_BONSAI_PARITY_MODE=reference swift test --filter Bonsai2SmallReferenceTests
MERERUN_TEST_BONSAI_PARITY_MODE=native swift test --skip-build --filter Bonsai2SmallReferenceTests
"$MERERUN_TEST_BONSAI_PARITY_ROOT/venv/bin/python" scripts/reference-parity/check_bonsai2_small.py \
  --root "$MERERUN_TEST_BONSAI_PARITY_ROOT" --size 2 \
  --output "$MERERUN_TEST_BONSAI_PARITY_ROOT/comparison-2.json"
```

Repeat for 4B and 9B. Completed export directories are evidence; preserve them before another run.
