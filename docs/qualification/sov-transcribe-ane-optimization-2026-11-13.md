# SOV ANE optimization follow-up

The parity-preserving ANE path is faster than the initial conversion. It still
does **not beat MLX** on the tested M5 Max. The requested performance target
was not achieved. The default speech execution provider remains MLX.

This follows the [initial qualification](./sov-transcribe-ane-2026-11-13.md).
The [raw receipt](./sov-transcribe-ane-optimization-2026-11-13.json) records
38 isolated component configurations, rejected candidates, individual full-model
runs, exact tokens, compiled artifact hashes, and code hashes.

## Bottlenecks and changes

SOV's Whisper tower has 32 layers with 1280 channels and a 5120-channel
feed-forward expansion. Each padded 30-second window requires approximately
2.29 trillion matrix/convolution floating-point operations before elementwise
work. MLX uses GPU matrix kernels and fused scaled-dot-product attention.

In the initial ANE layout, a four-head score tensor contains 18 MB of FP16
values. The feed-forward expansion contains 15.36 MB. Measurements of isolated
original-weight components showed:

| Component | Original layout | 128-frame tiles |
| --- | ---: | ---: |
| Attention, original orientation | 9.47 ms | 7.95 ms |
| Feed-forward, original normalization | 4.87 ms | 3.33 ms |

The selected conversion tiles **queries**, retaining all 1500 key/value frames,
and tiles the pointwise feed-forward computation. It keeps the original
normalization, attention multiplication orientation, and softmax axis. No audio
frames, layers, heads, or key context are removed. The resulting projected
tensors match the already-qualified ANE conversion bit-for-bit on both fixtures.

Smaller tiles were slower because they added too many operators. Spatial
repacking and QKV fusion did not improve the measured blocks. FastPrediction
increased full-graph latency from 423 ms to 454 ms in its controlled probe.

## Fidelity limits prevented promoting the fastest graph

Channel-oriented softmax and normalization reduced a full ANE prediction to
226 ms, but the long transcript changed **received** to **receive**. That
candidate passed the aggregate tensor tolerance and failed exact token parity.
It was rejected. Keeping the original attention orientation restores identical
FP16 projected outputs while retaining the safe tiling gains.

INT8 activations produced 16.1% relative RMSE in an isolated feed-forward block;
INT4 weights produced 24.1% in a complete block. These failed the numerical
bar and were not promoted. Weight-only INT8 measured 3.58 ms for the
feed-forward component, compared with 3.33 ms for FP16 feed-forward tiling.
The selected conversion keeps the original weights.

## End-to-end rerun

Measured on an M5 Max with 128 GiB, macOS 27.2 (26B123), Xcode 27.2 (27B52).
Both providers use the same pinned BF16 Gemma target, FP32 router, DFlash
companion, prompt, input audio, and 448-token budget. Each process performs one
unmeasured warmup and three measured repetitions. No repository checks,
Instruments recording, or compute-plan inspection ran alongside these timings.
All table values after duration are milliseconds.

| Audio | Provider | Encoder/projector | Prompt/prefill | Generation | Complete prepared pipeline |
| --- | --- | ---: | ---: | ---: | ---: |
| 3.51831 s | MLX | 58.5 | 255.1 | 61.4 | 408.5 |
| 3.51831 s | Optimized ANE + MLX | 329.4 | 257.3 | 60.8 | 682.5 |
| 46.57000 s | MLX | 114.1 | 477.5 | 1434.7 | 2113.6 |
| 46.57000 s | Optimized ANE + MLX | 658.1 | 485.7 | 1447.8 | 2676.1 |

The prepared pipeline includes feature extraction, encoder/projector and host
handoff, prompt/prefill, generation, and cache cleanup. Model load, audio-file
decoding, process startup, and model unload are outside these resident timings.
Fresh-process startup was not rebenchmarked in this follow-up.

Both recordings pass exact greedy-token parity for every measured repetition,
finite projected tensors, relative RMSE at most 5%, and cosine at least 0.999.
The raw receipt retains all token sequences and error values. The selected
artifact also matches the previous qualified ANE projected tensors exactly.

## Hardware verification and validation

A separate native Swift CLI trace confirms actual ANE prediction. The
[exported hardware events](./sov-transcribe-ane-optimization-2026-11-13-ane-events.xml)
retain prediction and first-use compilation intervals. CPU input/output work
is included in the encoder/projector wall timing; it is not the main bottleneck.
A full device-plan export for the large exploratory graph was stopped because
Core ML's structure loading was impractical; no new all-operation placement
claim is made from that unfinished inspection.

The full repository gate passed. Three canonical PyTorch tests independently
check tiled attention against scaled-dot-product attention, full key visibility
across tile boundaries, head ordering, and affine channel normalization. Full
checkpoint comparisons cover both recordings. These checks qualify two English
fixtures; they are not a multilingual accuracy or energy evaluation.

## Reproduce

```bash
uv run scripts/model-conversion/test_sov_coreml_graphs.py
uv run scripts/model-conversion/convert_sov_coreml.py /path/to/sov /path/to/optimized --optimization tiled
uv run scripts/model-conversion/profile_sov_coreml.py /path/to/sov /path/to/mel-dump.safetensors /path/to/profile-output --variants attention-baseline attention-original-query128 mlp-baseline mlp-chunk128
```

Use the [runtime guide](../runtime/sov-transcribe.md#core-ml-encoder-and-projector)
for the paired CLI benchmarks and native Instruments command. The earlier
`--optimization baseline` reproduces the original graph. `--optimization
feedforward` tiles only the feed-forward work. Profiling artifacts are bound to
the converter's source hash so stale graphs cannot silently contaminate a rerun.
