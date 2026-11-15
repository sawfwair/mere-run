# SOV custom dynamic-int8 Metal qualification — 2026-11-13

The native experimental FFN path passes the existing tensor and exact-token
checks on both recorded fixtures. It runs on the M5 GPU, using its TensorOps
matrix hardware. The standalone ANE is not involved. BF16 MLX remains the default.

| Fixture | Encoder/projector: BF16 → custom | End-to-end with DFlash: BF16 → custom | Embedding relative RMSE | Frozen tokens |
| --- | --- | --- | --- | --- |
| short | 58.13 → 48.92 ms | 304.61 → 298.64 ms | 1.63% | 13 / exact |
| long | 109.45 → 93.54 ms | 1451.16 → 1410.69 ms | 3.51% | 138 / exact |

Measurements use the release executable, original pinned SOV and DFlash assets,
two resident warmups and five synchronized measured runs per session, and
1024-token target prefill chunks. Loading, shader compilation, and static weight packing are
outside resident timing. Audio-file decoding is recorded separately.
Every prediction includes live activation packing.
All 3000 mel frames, 1500 encoder frames, and 375 projector tokens per window
remain present. Each provider has ten measured runs per recording, combining
forward and reversed session order. Baseline and candidate mel tensors are bit
identical.
The long recording uses two windows. Processes run serially without Instruments
or concurrent builds.
The raw receipt contains every stage timing and measured token sequence.
The first short-audio session was essentially tied end-to-end; the reversed
session favored the custom kernel. Combined medians show a modest gain, with
most total latency still in the target decoder. These results do not establish
a large end-to-end speedup. Long-audio DFlash accepts 128 draft tokens with
the custom embeddings versus 127 with BF16, while both execute 15 verification
rounds and 12 rejected rounds. End-to-end results include this behavior change.

## Kernel design

Both FFN projections use signed int8 GPU matrix multiplication with int32
accumulation and per-row scales. FC2 applies the same orthonormal block-256
Hadamard transform to activations and weights. The activation transform, range
reduction, quantization, and tile-major packing share one shader invocation.
Original BF16 checkpoint parameters remain available. Attention, convolution,
normalization, projector, target decoder, and DFlash retain native arithmetic.

The first prototype used small per-block power-of-two exponents on both
operands, shifting the integer accumulator between blocks. It improved real
projection latency, but removing those shifts improved complete-encoder latency
without materially changing embedding error on these fixtures. The shared
kernel retains bounded exponent support for numerical research; the native
benchmark candidate fixes the exponent to zero. No modified MLX distribution
or runtime fork is required.

The Python BF16 implementation matches the frozen native embeddings exactly.
Independent CPU checks cover Hadamard rotation, rounding ties, signed integer
rescaling, tile-major operands, padded rows, and dispatch limits. Native tests
cover bias, changing live inputs, both exponent modes, and invalid geometry.
An alternating tile comparison measured 45.36 ms for 32-column tiles and
45.76 ms for 64-column tiles, with bit-identical outputs. The native path keeps
32-column tiles.

The existing qualification limits remain 5% relative RMSE, cosine ≥ 0.999,
finite outputs, and identical frozen tokens in every measured run.

Validation passed: release build, strict lint, package policy in all three
supported modes, 13 SOV runtime GPU tests, nine CLI tests, one generated CLI
inventory test, and three independent Python oracle tests. The FP32 NumPy
fixtures use `MLX_ENABLE_TF32=0` to preserve their established tolerance; the new
int8 GPU tests also pass with the default setting. The BF16 benchmark runs keep
default runtime precision. Existing NumPy fixture tolerances and qualification limits are unchanged.

## Reproduce

```bash
swift build -c release --product mere.run
.build/release/mere.run model benchmark sov recording.wav \
  --model-root /path/to/SOV-Transcribe-1 \
  --draft-model /path/to/SOV-Transcribe-1-DFlash \
  --experimental-dynamic-int8 --prefill-chunk-size 1024 \
  --warmups 2 --repetitions 5 --projected-output /tmp/int8.safetensors --json
```

Omit `--experimental-dynamic-int8` for the reference and compare both dumps.
The flag requires an M5 GPU and macOS 26.2 or newer. It rejects Core ML artifacts.
These two recordings establish local parity, not corpus-wide ASR accuracy;
the option remains experimental and benchmark-only. The full repository gate
was deferred at the user's request after the protected-file read failure.

Raw timings, tokens, source hashes, and tensor errors: [JSON receipt](./sov-transcribe-custom-metal-2026-11-13.json).
