# SOV Core ML research follow-up

The opt-in 8-bit ANE configuration reduces resident encoder/projector latency
from about 329 ms to 223 ms per 30-second window and halves the compiled
artifact size. It passes the existing tensor tolerances and exact DFlash token
checks on both English fixtures. It still **does not beat MLX** on the tested
M5 Max. The default speech provider remains MLX, and the default converter
preset remains the original-weight FP16 `tiled` configuration.

The [raw receipt](./sov-transcribe-coreml-research-2026-11-13.json) retains
individual measurements, tokens, source revisions, artifact hashes, numerical
errors, component profiles, and clustering results. This report follows the
[FP16 optimization qualification](./sov-transcribe-ane-optimization-2026-11-13.md).

## Changes and rejected alternatives

The `palettized` converter preset combines per-head 4D attention expressed with
`einsum`, 500-frame query tiles, 128-frame feed-forward tiles, scaled channel
LayerNorm, and learned 8-bit convolution weight palettes. Every query retains
all 1500 keys. Scaling the normalization input by one eighth and epsilon by
one sixty-fourth preserves the FP32 formula and reduces the range of FP16
intermediates. The projector still emits 375 tokens per window.

The attention equations follow the public `SplitHeadsQ` implementation in
[Argmax's inference toolkit](https://pypi.org/project/argmaxtools/0.1.23/).
The [MASHBOX encoder](https://huggingface.co/mashbox-labs/whisper-large-v3-turbo-coreml-ane)
also uses input scaling before channel LayerNorm. Apple's
[palettization guidance](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html)
explains the compressed weight format and its dependence on memory traffic.

Serial full-graph Core ML measurements isolate the contributions. Each uses
two warmups and nine predictions with the same real mel input and CPU_AND_NE.
These measurements exclude Swift feature extraction, MLX handoff, and decoding.

| Full encoder/projector graph | Median prediction | Compiled size |
| --- | ---: | ---: |
| Original qualified FP16 tiling | 326.0 ms | 1,324 MB |
| Per-head attention and scaled normalization, FP16 | 222.8 ms | 1,326 MB |
| Original attention with 8-bit palettes | 320.8 ms | 667 MB |
| Per-head attention and scaled normalization, 8-bit palettes | 216.4 ms | 669 MB |

Most latency improvement comes from the graph layout. Palettization primarily
reduces storage here. The uncompressed per-head candidate changes **received**
to **receive** on the long fixture and is rejected despite passing tensor
tolerances. The selected 8-bit candidate matches every token on both fixtures;
this agreement does not establish accuracy on other recordings.

Isolated original-weight block probes also tested 6-bit palettes and grouped
palettes. Six-bit palettes increased error without improving on the selected
8-bit tiled block's latency. Keeping LayerNorm and softmax in FP32 instead
increased block latency to 272 ms; the compute plan assigned many operations
to CPU. That precision variant is rejected. Compute-plan placements describe
anticipated execution, rather than measured operator times.

## Published encoder controls on this machine

The downloaded controls use pinned revisions recorded in the receipt. Each
measurement uses CPU_AND_NE, two warmups, nine predictions, and the same SOV
fixture's mel features. These are different checkpoints and output interfaces,
so they are hardware and implementation controls rather than SOV parity tests.

| Published encoder | Input window | Median prediction |
| --- | ---: | ---: |
| WhisperKit large-v3, FP16 | 30 s | 479.7 ms |
| MASHBOX large-v3-turbo, palettized | 30 s | 340.1 ms |
| MASHBOX large-v3-turbo, palettized | 10 s | 62.5 ms |

WhisperKit emits encoder features. MASHBOX additionally projects decoder
cross-attention keys and values. Their model downloads contain no SOV projector
or Gemma decoder. The shorter MASHBOX window demonstrates the cost of padding;
it does not establish a parity-preserving shorter SOV encoder. Native SOV
retains its full window and prompt interface.

## e-means comparison

[e-means](https://e-means.github.io/) accelerates codebook training. This
experiment measures offline palette construction separately from inference.
It uses the official Faiss 1.16.0 NumPy implementation, eight CPU threads,
three seeds, and one million sampled weights from each of SOV layers 0 and 16.
For each tensor, 800,000 values train the palette and 200,000 score held-out
error. Palettes are rounded to their actual FP16 storage precision before
scoring. Training and final assignment are timed together.

Three-epoch e-means improves on 25-iteration Faiss Lloyd clustering in speed
and held-out error on these scalar weights. Core ML's specialized weighted
one-dimensional solver is faster and more accurate than either in this test.
It clusters unique BF16 values with their occurrence counts, rather than
repeatedly assigning every sampled scalar. The converter retains that solver.
The receipt includes all 48 measurements and implementation hashes. This
result does not evaluate vector palettization or e-means' GPU kernels.

## End-to-end parity and performance

Both providers use the original pinned BF16 Gemma target, FP32 router, the same
DFlash companion, and a 448-token budget. Each process performs one warmup and
three measured runs. Model loading, file decoding, and first inference are
reported separately. Compilation, repository checks, tracing, and clustering
do not overlap these resident measurements. All table values are milliseconds.

| Audio | Provider | Encoder/projector | Prepared pipeline |
| --- | --- | ---: | ---: |
| 3.518 s | MLX + DFlash | 57.2 | 413.4 |
| 3.518 s | 8-bit ANE + MLX DFlash | 222.9 | 588.1 |
| 46.570 s | MLX + DFlash | 109.4 | 2114.7 |
| 46.570 s | 8-bit ANE + MLX DFlash | 438.4 | 2503.8 |

All three repetitions match 13 tokens for the short clip and 138 tokens for
the long clip. Projected tensors are finite. Relative RMSE is 3.827% and 2.644%,
and cosine similarity is 0.999288 and 0.999651, respectively. The limits remain
5% and 0.999. These two English fixtures are not a dataset-wide or multilingual
word-error-rate evaluation.

A separate native Swift CLI Instruments recording confirms actual Neural
Engine predictions at 223.10 ms and 223.14 ms. The
[exported hardware events](./sov-transcribe-coreml-research-2026-11-13-ane-events.xml)
retain those intervals. These are hardware predictions, rather than compute-plan
cost estimates. The decoder and DFlash remain on MLX GPU. The requested
end-to-end performance target remains unmet.

## Reproduce

```bash
uv run scripts/model-conversion/test_sov_coreml_graphs.py
uv run scripts/model-conversion/convert_sov_coreml.py /path/to/sov /path/to/sov-p8 --optimization palettized
uv run scripts/model-conversion/profile_sov_coreml.py /path/to/sov /path/to/mel.safetensors /path/to/profiles --variants block-baseline block-p8 block-p6 attention-einsum500 block-scaled-einsum
uv run scripts/model-conversion/profile_sov_palettes.py /path/to/sov /path/to/palette-training.json
```

Use the [runtime guide](../runtime/sov-transcribe.md#core-ml-encoder-and-projector)
for native paired benchmarks, tensor comparisons, and Instruments recording.
The installed-checkpoint regression covers both recordings. Canonical PyTorch
tests compare the layout variants against independent scaled-dot-product
attention and LayerNorm, including incomplete query tiles and nonzero affine
biases. The raw receipt records repository-gate and checkpoint-test status.

Validation passed: the full `./scripts/check.sh` gate, all three canonical
conversion tests, and the expanded real-checkpoint GPU parity regression using
the selected 8-bit artifact. The checkpoint test compares both recordings
across MLX and Core ML in one test run.
