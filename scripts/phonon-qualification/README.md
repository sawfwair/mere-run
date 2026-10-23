# Phonon released-weight qualification

These tools compare the native Swift/MLX model with the pinned Gradium Rust
implementation. Python and Rust are qualification tools; product inference
runs entirely in Swift/MLX.

Use an external scratch folder. Builds and audio do not belong in the repo.
Resolve the root Swift package first so `.build/checkouts/mlx-swift` matches
`Package.resolved`. Then build both release executables:

```sh
python3 scripts/phonon-qualification/prepare.py /path/to/scratch
```

The upstream checkout is pinned to
`49b9033e44ab0fbb935973eca3213757644d1bbb`. The preparation tool registers one
additional example without modifying the upstream library or its benchmark.
The native harness compiles the repository's actual model sources, with test
visibility for injecting noise. It uses the repository's vendored metallib.

Download the original `gradium/phonon-31m-en` assets at
`92741c5b07d085ff840e26fea8ce43110b87852f` into a checkpoint directory. Required
files and SHA-256 values are recorded in
`docs/architecture/phonon-native-qualification.json`. Verify those hashes before
comparing. Q8 is an upstream performance comparator; the native loader uses
`model.safetensors`.

## Numerical comparison

The runtimes have different RNG implementations. Generate one float32 Gaussian
noise tape, then replay it through both implementations. Reset the tape cursor
for each request, and keep it continuous across sentence chunks:

```sh
uv run --with numpy==2.5.4 python - <<'PY'
import json
import numpy as np
noise = np.random.default_rng(8291).normal(0, np.sqrt(.8), 65536).astype(np.float32)
with open('/path/to/scratch/noise.json', 'w') as f:
    json.dump(noise.tolist(), f)
PY

/path/to/scratch/upstream-target/release/examples/mere_qualify \
  /path/to/checkpoint scripts/phonon-qualification/corpus.json \
  /path/to/scratch/upstream-replay /path/to/scratch/noise.json none
/path/to/scratch/native-harness/.build/release/PhononQualification \
  /path/to/checkpoint scripts/phonon-qualification/corpus.json \
  /path/to/scratch/native-replay gpu 1 0.8 /path/to/scratch/noise.json
uv run --with numpy==2.5.4 python scripts/phonon-qualification/compare.py \
  /path/to/scratch/native-replay /path/to/scratch/upstream-replay \
  /path/to/scratch/replay-comparison.json
```

The corpus includes 20 prose/spelled-value cases and four raw structured-value
diagnostics. `none` disables upstream text rewriting to isolate model parity.
For a product-quality comparison, also run the upstream harness with `en` and
compare intelligibility on the supported cases; inspect the raw structured
cases separately. Native structured-text rewriting is outside this adapter.

Use `transcribe.py CLI OUTPUT_JSON NATIVE_DIR UPSTREAM_DIR NORMALIZED_DIR`
to collect local Parakeet transcripts and `score-asr.py TRANSCRIPTS OUTPUT_JSON`
to score them. Record the installed ASR checkpoint identity with the results.
The scorer retains literal number formatting, so ASR rendering spoken values
as digits can count as word edits even when the speech is intelligible.

The harness writes float32 PCM, per-frame latent/EOS traces, reference chunk
plans, and timings. Full autoregressive waveforms can drift from small numeric
differences. Report length, early-step errors, correlation, signal-to-error
ratio, and intelligibility separately. ASR word error rate measures
intelligibility; it does not establish listener preference or naturalness.

A small upstream trace is retained as an opt-in regression oracle:

```sh
MERERUN_PHONON_CHECKPOINT=/path/to/checkpoint MERERUN_TEST_MLX_DEVICE=gpu \
  swift test --filter PhononReleasedCheckpointTests
```

## Performance comparison

Run benchmarks serially with other inference and builds stopped. Use the
`short-hello`, `directions`, and `long-sentence` corpus entries for the native
harness, omit the noise tape, and use 10 measured iterations at temperature 0.
The native harness warms each prompt once. It includes text planning and voice
conditioning per request and excludes model/voice file loading. Time to first
audio ends at the first 1,920-sample callback.

```sh
python3 scripts/phonon-qualification/benchmark.py \
  /path/to/scratch/upstream-target/release/examples/bench \
  /path/to/checkpoint /path/to/scratch/upstream-benchmark
```

The unmodified upstream benchmark uses three warmups, 10 measured iterations,
four CPU threads, Accelerate, and KleidiAI where the host supports it. Its
`Synth::stream` path overlaps flow and Mimi workers and caches the voice prefix.
It excludes model loading and voice conditioning. Label these boundaries when
comparing against native Metal timings. Identical integer seeds alone do not
produce identical noise across runtimes; temperature 0 removes that variable.
Report RTF as generation seconds divided by output audio seconds, using each
run's actual frame count. Separate original F32 from Q8 results.
