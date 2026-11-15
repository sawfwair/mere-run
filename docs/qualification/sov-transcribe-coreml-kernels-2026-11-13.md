# SOV ANE kernel follow-up

The native pipeline reduces resident MLX inference from 403 to 296 ms on the
short recording and from 2.110 to 1.403 seconds on the long recording. The
qualified palettized ANE path improves from 588 to 465 ms and from 2.504 to
1.724 seconds. All 13 and 138 reference tokens match on five release runs.
The ANE encoder itself remains slower than MLX on this M5 Max.

This experiment tests delayed attention normalization, feed-forward tiling,
calibrated activation quantization, and direct ANE execution on the same M5 Max
and original SOV checkpoint as the
[previous research round](./sov-transcribe-coreml-research-2026-11-13.md).
The [raw receipt](./sov-transcribe-coreml-kernels-2026-11-13.json) retains
component timings, calibration provenance, artifact hashes, token comparisons,
resident measurements, and validation results.

## Delayed normalization

The previous per-head attention computes `softmax(QKᵀ)V`. The new graph computes
`exp(S − max(S))V / sum(exp(S − max(S)))`, where `S = QKᵀ / sqrt(64)`.
Moving the division after the value reduction preserves the FP32 function and
avoids rounding every normalized probability to FP16 before that reduction.
Each 500-query tile still attends to all 1500 keys. The projector still produces
375 tokens per window.

The experimental `streaming` layout also supports online softmax over smaller
key tiles, with running maxima and denominator corrections. Smaller key tiles
reduce score-buffer size but were slower on this host. Independent tests cover
query and key tails, multiple batches, and head ordering against PyTorch SDPA.

## Disjoint calibration and held-out probes

Calibration uses 12 English utterances from
[`hf-internal-testing/librispeech_asr_dummy`](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/tree/5be91486e11a2d616f4ec5db8d3fd248585ac07a).
Eight different utterances supply held-out activations. Neither qualification
WAV is used for calibration. The receipt records the Parquet hash, audio hashes,
utterance IDs, speaker IDs, and split membership. This small dataset uses one
speaker and does not establish multilingual transcription accuracy.

The calibration frontend is checked against the independent NumPy fixture,
including normalization across two audio windows. The graph collects actual
intermediate activations from the original FP32 encoder. Probe errors compare
each independently converted sublayer or block with its FP32 computation;
these are not accumulated full-encoder errors.

| Layer-0 probe | Median | Worst held-out relative RMSE |
| --- | ---: | ---: |
| Previous per-head attention sublayer | 4.652 ms | 0.794% |
| Delayed-normalization attention sublayer | 4.315 ms | 0.077% |
| Online softmax, 500-key tiles | 5.090 ms | 0.100% |
| Original FP16 block | 7.344 ms | 1.455% |
| Delayed-normalization block | 6.909 ms | 1.281% |
| Delayed block, 1024-channel FFN tiles | 6.801 ms | 1.281% |
| Delayed block, 750-query tiles | 10.505 ms | 1.281% |
| Delayed block, calibrated input-A8 MLP, full token width | 6.578 ms | 1.814% |

Channel tiling splits the FFN hidden dimension and sums all second-projection
contributions, adding the output bias once. Its independent test includes
nondivisible token and hidden-channel tails. Packing four attention heads into
the spatial dimension instead takes 8.07 ms; packing all 20 causes CPU fallback
and takes 94.21 ms. Those layouts are rejected.

## Activation quantization

Apple documents higher int8 throughput on newer Neural Engines in its
[quantization guidance](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html).
This experiment requests per-channel int8 convolution weights and calibrated
activation Q/DQ boundaries. Those graph boundaries do not prove the hardware's
MAC precision.

The channel-balancing transformation follows the algebra described in
[SmoothQuant](https://arxiv.org/abs/2211.10438): divide a linear input by positive
channel scales and multiply the corresponding weight columns by those scales.
For FC1, both LayerNorm gamma and beta absorb the input division. FC2 retains an
explicit division after GELU. The scales use training activations and weight
maxima with alpha 0.5. This is a local implementation of the rescaling method,
not the published model's calibration recipe. FP32 block identity checks retain
relative errors around 2e-7.

Input-only A8 on the MLP reduces block error from 4.014% to 1.963% after
balancing, but its 7.115 ms latency barely improves on 7.344 ms FP16. Adding
globally quantized convolution outputs before GELU increases block error to
29.475%. Per-channel output quantization brings that error down to 2.058% but
still takes 7.162 ms. Those output-quantized variants are rejected.

## Direct ANE control

The optional research script uses
[ANEForge 0.4.0](https://github.com/sbryngelson/ANEForge) outside the shipping
runtime. Its direct e5rt graph runs the complete original-weight attention
sublayer in 4.271 ms, including channel normalization, Q/K/V projections, full
attention, and output projection. It has essentially the same latency and error
as the 4.315 ms Core ML graph. Removing the Core ML API layer does not expose a
large improvement in this control.

The native fused-SDPA bridge is a separate path. It rejects the tested
500-query/1500-key and 250-query/1500-key shapes, and a 250-by-250 control, with
ANE error 10, `verifyBundleAtPath: invalid model`. This is a host/runtime
compatibility failure; it establishes no fused-SDPA performance limit for SOV.
The bridge's default persistent worker assumes equal Q/K lengths. The research
script applies its subprocess bridge's asymmetric-query netplist adjustment
before testing the full key context.

## Full-model qualification

The final serial release measurements below use two warmups and five measured
runs per provider and recording. Every run must match the frozen original
greedy tokens. Instrumented measurements are separate from this timing series.

| Recording | Provider | Current encoder/projector | Before resident total | Current resident total |
| --- | --- | ---: | ---: | ---: |
| 3.518 s | MLX + DFlash | 58.3 ms | 402.5 ms | 296.3 ms |
| 46.570 s | MLX + DFlash | 108.2 ms | 2109.8 ms | 1403.1 ms |
| 3.518 s | Palettized ANE + MLX DFlash | 221.2 ms | 588.1 ms | 464.9 ms |
| 46.570 s | Palettized ANE + MLX DFlash | 447.8 ms | 2503.8 ms | 1724.2 ms |

The current providers both use 1024-token prefill. MLX before-values come from
this round's frozen baseline; the palettized before-values come from the
preceding research receipt for the same compiled artifact. Those before-runs
used one warmup and three measurements. Resident totals exclude file decoding,
model loading, and first-use compilation, which are recorded separately.

The frontend mel tensors are bit-identical to the frozen baseline on both
recordings. Short-recording feature time falls from about 29 to 2.7 ms; long
feature time falls from about 60 to 16.6 ms. The final current-runtime
qualification selects the existing `palettized` graph. Its projector tensors
pass the 5% relative-RMSE and 0.999 cosine limits, and every greedy token matches.

The uncompressed delayed graph's diagnostic compute plan assigns all 23,856
runtime operations to ANE. A separate final Instruments recording establishes
actual ANE execution for the selected palettized artifact with the updated
native pipeline. See the
[hardware event export](./sov-transcribe-coreml-kernels-2026-11-13-ane-events.xml).

## Native pipeline changes

The larger gain comes from DFlash verification. The previous implementation
replayed every partially accepted block through the target model. The accepted
KV rows and hidden states are already available in the verification result.
The new implementation appends those rows to the untouched base cache and
reuses the corresponding logits and hidden states. It preserves history that
the trial's sliding window may have evicted. It collects all layer updates
before mutating the base and retains replay when the trial cannot supply every
proposed row.

With the original 256-token prefill, this removes 12 repeated target forwards
on the long MLX recording. Generation falls from 1.432 to 0.989 seconds and
total resident inference from 2.110 to 1.659 seconds. All 138 tokens match on
five repetitions. The short recording already accepts its first draft block,
so this change alone gives it no material gain.

The benchmark now records `reusedVerificationRounds`. Older receipts decode
with this optional field absent. Tests cover wrapped full and sliding caches,
partial acceptance, an evicted-row fallback, a varying target sequence, and EOS.

Prefill now uses chunks of up to 1024 tokens. An opt-in resident tuning test
compares 256, 512, and 1024 with one prepared model and records token parity.
The tuner uses a debug build and its timings are excluded from release speed
claims. The release CLI exposes `--prefill-chunk-size` and records the chosen
size. Provider comparisons reject different prefill sizes; historical receipts
without the field use the previous implicit size of 256.

The shared mel extractor skips zero filter coefficients while retaining the
order of every nonzero accumulation, and reuses its FFT input buffer. SOV also
stops transforming the waveform after a complete zero FFT window beyond the
last real sample. It broadcasts that window's normalized silence column into
the remaining mel frames. Global normalization still precedes window splitting;
all 3000 mel frames, 1500 encoder positions, and 375 projector tokens remain.
The frontend tests compare the shortened transform with a full padded transform
and retain the independent NumPy checks across a 30-second boundary.

## Further rejected encoder candidates

Masked padding to 1536 encoder positions preserves the FP32 function and
excludes every artificial key, but raises block latency to 8.16 ms. Batched-head
attention takes about 7.9 ms. Per-head QKV projection and channel-first FFN
scheduling remain around 6.8–6.9 ms per block. These do not supply a useful
full-model improvement.

The full calibrated input-A8 graph reaches about 195 ms per window but fails
token parity and the cosine threshold. Splitting the FFN into int8 regular
neurons and an FP16 outlier branch also reaches about 196 ms per window, but
its full projected-tensor relative RMSE exceeds 5% on both recordings. It is
rejected. Calibration and outlier selection use only the independent training
split.

The new delayed-normalization palette graph passes with the earlier replay
decoder, then changes **received** to **receive** with prefix reuse. It is
rejected for the updated runtime. The previous `palettized` preset and the
uncompressed delayed graph pass the 256-token prefill compatibility check. With
1024-token prefill, the uncompressed delayed graph also changes that long-clip
token and is rejected. The original FP16 `tiled` preset passes its long-clip
compatibility check with the new defaults. The raw
receipt keeps each decoder phase separate; a faster isolated encoder does not
guarantee fewer DFlash verification rounds.

## Validation

The release build, strict SwiftLint, 25 focused Swift tests (one opt-in asset
test skipped), and nine independent graph tests pass. The real-checkpoint
MLX/Core ML test passes for both recordings and asserts that rejected rounds
reuse verified prefixes. The CLI inventory was regenerated and verified.

The complete repository gate was attempted and encountered macOS permission
failures when reopening protected temporary run records. A standalone protected
file reproduces Cocoa 257 / POSIX EPERM. The user explicitly deferred that gate;
file protection remains enabled. CLI help and hygiene checks that followed the
blocked test stage were run separately.

## Reproduce

The selected graph uses the existing palette preset:

```bash
uv run scripts/model-conversion/convert_sov_coreml.py /path/to/checkpoint /path/to/palettized \
  --optimization palettized
```

Export the uncompressed delayed-normalization research graph:

```bash
uv run scripts/model-conversion/convert_sov_coreml.py /path/to/checkpoint /path/to/delayed \
  --optimization feedforward --attention-layout streaming --query-chunk-frames 500 \
  --channel-normalization --normalization-scale 8
```

The delayed graph's earlier passing token check uses
`--prefill-chunk-size 256`; it fails the current 1024-token qualification and is
not the selected runtime artifact. Adding weight palettes to that layout also
fails. The default converter preset and speech provider are unchanged. The
[runtime guide](../runtime/sov-transcribe.md) documents native benchmarking,
tensor comparison, and Instruments capture.

Prepare the pinned Parquet file and run independent component probes:

```bash
uv run scripts/model-conversion/prepare_sov_coreml_calibration.py /path/to/checkpoint \
  /path/to/validation-00000-of-00001.parquet /path/to/calibration
uv run scripts/model-conversion/profile_sov_ane_kernels.py /path/to/checkpoint \
  /path/to/calibration /path/to/probes --variants block-delayed block-delayed-a8
uv run scripts/model-conversion/test_sov_coreml_graphs.py
```

For the calibrated full-model experiment, add
`--activation-calibration /path/to/calibration --feedforward-chunk-frames 1500`
to the uncompressed export command. These overrides require full qualification
before use. The optional direct-ANE probe is:

```bash
uv run scripts/model-conversion/profile_sov_direct_attention.py /path/to/checkpoint \
  /path/to/calibration /path/to/direct.json --backend e5rt-delayed
```
