# Unlimited-OCR-2 verified-kernel price investigation

The full native receipt still emits `9.00`; upstream's independent greedy run
emits `9.000`. This investigation fixes a native DASH scalar-promotion mismatch
and traces the price choice through matched token histories and hidden states.
The remaining full-run transcription difference isn't resolved.

## Native correction

Upstream computes its router coefficients as Python scalars before multiplying
FP32 tensors. For the released 128-dimensional indexer and scale 128, it uses
`scores * (128 / sqrt(128))`. Native code previously evaluated tensor
multiplication and division separately, using a rounded FP32 square root.
That adds a rounding boundary and can differ by one FP32 ULP.

`UnlimitedOCRMath.scaleRouterScores` combines the constants in CPU double
precision, converts the coefficient to FP32 once, and performs one tensor
multiplication. Router head summaries, block scores, and per-layer scaling use
that helper. The independent PyTorch fixture in
`Tests/UnlimitedOCRRuntimeTests/Fixtures/router-scaling.json` exposes the old
rounding difference. Its generator records the expected and separately rounded
results. The new exact checks pass on GPU with the other 17 owned numerical
tests.

This correction changes some generated coordinates. It doesn't recover the
missing price zero. Agreement with upstream geometry doesn't improve uniformly;
the correction follows upstream arithmetic rather than selecting outputs by
their agreement with this receipt.

## Controlled price decision

The differing suffix is one generated token: ID 504 is `00`, and ID 1320 is
`000`. It occurs at zero-based output step 341, after `<FORK>9.`. Prefix replay
forces tokens only before that decision; the price suffix remains a model
prediction. Both paths retain the original BF16 checkpoint and output-head
rounding boundary.

Before the scalar correction, the controlled results were:

| Decoder | Vision features | Earlier text/layout prefix | `00` score | `000` score | Choice |
| --- | --- | --- | ---: | ---: | --- |
| Upstream MPS | Upstream | Upstream | 20.0 | 20.25 | `000` |
| Native MLX | Upstream | Upstream | 19.875 | 20.125 | `000` |
| Native MLX | Native | Upstream | 19.875 | 20.0 | `000` |
| Upstream MPS | Upstream | Native | 19.75 | 19.875 | `000` |
| Native MLX | Upstream | Native | 19.875 | 20.0 | `000` |
| Native MLX | Native | Native | 20.0 | 20.0 | `00` |

The last row reaches an exact BF16 logit tie. Ordinary argmax selects ID 504,
the lower token ID. Replacing either vision features or preceding tokens with
the upstream values restores a positive `000` margin. Small state differences
and the model's earlier geometry choices jointly affect the final price token.

The scalar correction changes the preceding coordinates, so the corrected run
was replayed again:

| Decoder/input | Corrected native prefix | `00` score | `000` score | Choice |
| --- | --- | ---: | ---: | --- |
| Native with native vision | Yes | 20.25 | 20.0 | `00` |
| Native with upstream vision | Yes | 19.75 | 19.75 | `00` |
| Upstream with upstream vision | Yes | 19.875 | 19.75 | `00` |

Upstream itself selects `00` when conditioned on the corrected native layout
prefix. The native replay follows its greedy choices exactly through step 341,
which verifies the reconstructed prefix against the model's decisions.
The complete native result therefore contains a different suffix token; text
decoding isn't dropping a zero.

The raw replay records are in `before-reference-prefix/`,
`before-native-prefix/`, and `after-native-prefix/`. Captures use the same source
revision, image, and checkpoint described in the parent qualification. Large
tensor files remain external.

## Projection and attention checks

PyTorch's BF16 output head, applied to the corrected native hidden state,
produces the same price scores as native: 20.25 for `00` and 20.0 for `000`.
Promoting its matrix multiplication to FP32 before BF16 output rounding doesn't
change that choice. Native's unrounded two-row FP32 projection also favors
`00`: approximately 20.2303 versus 20.0611. See
[`output-projection.json`](./output-projection.json). A projection precision
change wouldn't recover `000` from this hidden state.

Isolated native attention on upstream's actual query, key, value, and router
tensors matches the BF16-rounded reference exactly in nine of twelve layers
at the original price step. The other layers have one or two differing output
elements, with maximum absolute difference 0.0004883. A dense explicit FP32
matrix/softmax probe reduces some differences but doesn't establish complete
backend identity. The isolated output projection also matches the upstream
price candidates on an upstream hidden state; its remaining vocabulary errors
are recorded separately.

The vision probe uses real first-window SAM tensors. Upstream MPS BF16 SDPA
differs from its own explicit FP32 attention calculation after BF16 rounding;
adding BF16 score/probability rounding doesn't reproduce the fused backend
exactly either. See [`vision-probe.json`](./vision-probe.json). These probes
document finite-precision differences; they don't identify another native
equation or dtype error that warrants a production change.

## Repeated receipt comparison

The final matrix repeats JPEG base, JPEG auto, and canonical PNG base through
both CLI and XCTest, with temperature zero, TF32 disabled, a 1,024-token cap,
and no-repeat size 35/window 128. Kernel provenance validation remains enabled.
Raw outputs, checkpoint routing, selected kernel hashes, timings, frozen anchor
checks, and upstream comparisons are in
[`results/summary.json`](./results/summary.json).

All six outputs are byte-identical, contain 495 generated tokens, and stop
before the cap. They use verified kernel SHA-256
`64431ed485178f63a8dc870c8231ba021090c3356218c607a726821a7b6abb82`.
The independent upstream greedy rerun with repetition size 35 also contains
495 tokens and exactly matches the earlier size-zero reference, including
`9.000`. Its raw record is in `upstream-rerun/`.

| Agreement with upstream | Before scalar correction | After scalar correction |
| --- | ---: | ---: |
| Exact block body/type matches | 24/25 | 24/25 |
| Mean absolute coordinate difference | 0.441 | 1.176 |
| Maximum coordinate difference | 6 | 4 |

Geometry uses the same 17 unique exact body/type pairs and excludes repeated
bodies. These are agreement measurements against upstream output, without
annotated ground-truth boxes. Wall times include model loading and weren't
measured as an isolated throughput benchmark.

The price gap and the shared `MODELING` versus printed `MODELLING CLAY` spelling
failure remain. These checks don't establish exact transcription or full BF16
numerical parity. The unrelated eight-case document suite wasn't repeated.
No checkpoint tensors, decoder tie-breaking rule, or text formatting changed.
The standard repository gate, documentation build, and all 18 GPU numerical
tests pass. The asset-dependent prefix and projection diagnostics also pass.

## Reproduce

Use the upstream environment and pinned local source prepared by the parent
qualification. Capture the price state under the reference prefix:

```bash
python docs/qualification/unlimited-ocr-2-metal-20261030/price/trace-price.py \
  --source-root /path/to/upstream/model \
  --image /path/to/receipt.jpeg --reference /path/to/upstream-receipt.json \
  --output /tmp/unlimited-ocr-price
```

Set `MERERUN_TEST_UNLIMITED_OCR_PRICE` to that capture directory and
`MERERUN_TEST_UNLIMITED_OCR_ROOT` to the installed original checkpoint. Run
`UnlimitedOCRPriceDiagnostics.testSamePrefixPriceDecision` with
`MERERUN_TEST_MLX_DEVICE=gpu` and `MLX_ENABLE_TF32=0`.
`MERERUN_TEST_UNLIMITED_OCR_PRICE_NATIVE_FEATURES` optionally supplies native
page features from an external diagnostic capture for the vision substitution.
The existing execution report's diagnostic patch produces that file.

`probe-head.py` checks same-input output projection; `--native-state` accepts
the native state emitted by the diagnostic. `probe-vision.py` checks actual
SAM attention inputs. Production inference requires neither Python nor these
diagnostic captures. `./scripts/check.sh` is the standard local gate.
