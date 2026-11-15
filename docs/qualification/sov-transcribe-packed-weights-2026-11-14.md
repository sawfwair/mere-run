# SOV streamed and saved Q8 weights — 2026-11-14

SOV can convert its MoE experts once, save a native target checkpoint on an
external SSD, and load it directly on subsequent requests. The existing Q8
packing path is the numerical reference. Default transcription remains BF16;
`--sov-packed-weights` explicitly selects Q8 and can change recognition output.

## Results

Apple M5 Max, 128 GiB, Version 27.2 (Build 26B123). Source BF16 and the
packed target were on the same external APFS SSD, `/Volumes/MODELS`.
Both paths used identical Q8 Metal experts, attention optimization, rollback,
DFlash, and the 46.57-second fixture. There were six fresh CLI processes per
mode across two sessions with opposite starting orders. Each process retained
only its own model. Values below are medians across all six runs.

| Measurement | Load BF16 then pack Q8 | Load saved Q8 |
| --- | ---: | ---: |
| Model loading | 7.27 s | 3.30 s |
| Load plus first transcription | 8.48 s | 4.74 s |
| Peak physical footprint at end of loading | 49.18 GiB | 29.28 GiB |
| Peak MLX allocation at end of loading | 48.47 GiB | 29.11 GiB |

The ratio of median model-load times is **2.20×**.
Physical loading peak fell **40.46%**.
The median adjacent-pair loading reduction was 57.12%,
with 5/6 paired wins. This is a loading improvement;
resident inference uses the existing Q8 kernels and arithmetic.

Before each timed process, the harness sequentially read that mode's weight
files. Pre-read time is outside the reported loading interval. This does not
guarantee file-cache residency during subsequent allocation and loading.
These results are **not certified cold-disk timings**. All samples are included:
old-path loads ranged 6.44–13.92 s,
and native loads ranged 2.62–13.09 s.
One native load took 13.09 s with heavy host page-ins. The reversed session added
per-process disk I/O counters; those native runs read only approximately
11–60 MB from storage and loaded in 3.19–3.65 s. The raw report preserves all
host VM statistics and available process counters. Background work was not
terminated. Do not extrapolate these times to every SSD or smaller Mac.

“Load plus first transcription” sums audio-read time, model preparation, and
first inference inside the CLI. It excludes CLI initialization and result
formatting. The JSON also preserves total process wall times. Kernel
`ri_lifetime_max_phys_footprint` was sampled immediately after preparation,
before inference. MLX loading peak was captured at the same boundary.

## One-time conversion and storage

The successful conversion loaded the full model in
55.36 s, including
52.48 s validating, streaming, and saving
the target pack. Loading peak physical footprint was
29.32 GiB.
The saved target occupies 27.08 GiB.
The complete source checkpoint is 49.62 GiB;
the target pack excludes the encoder, projector, tokenizer, and draft.

The initial implementation used Foundation `FileHandle` reads through the
shared streaming reader. MLX peak was only 29.11 GiB, but retained CPU read
buffers raised the physical loading peak to 76.37 GiB. That implementation was
rejected. The final converter explicitly evaluates one native MLX CPU load
node before constructing GPU quantization work, removes consumed load nodes,
and releases retired allocations. It avoids a checkpoint-sized Foundation
allocation. Its first transcript matches the rejected converter's output.
The first attempt's measurements are retained in the JSON; its generated pack
was removed before the corrected fresh conversion.

## Parity and validation

- **777 stored target tensors:** every byte digest matches the original
  load-then-Q8 path. Stored shard keys exactly equal the compared digest keys.
  This covers packed codes, affine scales/biases, FP32 routers, dense weights,
  attention, embeddings, and normalization parameters.
- All 12 long-fixture startup transcripts matched exactly (138 tokens).
- Short audio and the 93.14-second repeated fixture crossing the sliding-cache
  wrap matched the old Q8 path
  token for token. The public `speech transcribe --sov-packed-weights` command
  produced the same short-fixture transcript.
- Mixed Q4/Q6/Q8 synthetic conversion and reload match the old packed arrays
  and logits. Tests cover missing tensors, incompatible source/precision,
  changed shards, and cleanup without publishing a partial pack.
- Focused CLI, Studio, and documentation checks: 40 passed, two optional checks
  skipped. GPU packing/low-bit regression: five passed, one asset check skipped.
  The later installed-model run passed all three packed-weight tests, including
  the full checkpoint comparison. Strict lint and whitespace checks passed.
  The full repository gate remains deferred under the user's earlier instruction.

The first full-model digest assertion compared generic Gemma's entire parameter
object, including randomly initialized per-layer-input slots that SOV disables.
There were 62 differences, all in that inactive feature. SOV's validated
`hidden_size_per_layer_input = 0` and the existing loader exclude these slots.
The corrected test asserts that configuration, compares the actual checkpoint
parameter set, and passes. Both receipts are retained locally. No inference
code or stored weight was changed to resolve that test-scope error.

Parity here qualifies storage against the existing Q8 path. Q8 recognition
quality remains governed by the earlier multilingual study; it is opt-in.
No claim of BF16 output identity or new smaller-memory hardware qualification
follows from this result.

## Format and use

```bash
mere.run speech transcribe ./recording.wav --backend sov \
  --model /Volumes/MODELS/MereRunModels/sov-packed-qualification/source-bf16 \
  --draft-model /path/to/sov-dflash \
  --sov-packed-weights /Volumes/MODELS/MereRunModels/sov-packed-qualification/q8-g64-v1
```

The first request creates an absent destination; later requests load its native
shards directly. The pack is published by an atomic directory rename only after
all expected tensors and the manifest are written. Concurrent conversions use
separate staging directories; one can publish a given destination. Existing
incompatible or incomplete directories fail without being overwritten.

`sov-packed.json` records format/packer versions, group size, each layer's bit
width, source config/index hashes, and shard size/mtime/header identities.
Startup validation uses these identities without rereading all tensor payloads
for a corruption scan. The source directory is still required for configuration,
tokenizer, encoder/projector, and source identity checks. The draft remains
separate. No persistent GPU shader binary or derived Q6 prefill cache is stored.
Q8 and Q8 Metal share the same weight format; dispatch is selected at runtime.

For Q4/Q6/mixed benchmark policies, use `model benchmark sov --packed-weights`
with the corresponding explicit expert policy and recipe. The production
`speech transcribe` option selects uniform affine Q8/group-64 Metal experts.

Reproduce the measured startup study with
`scripts/validation/benchmark_sov_packed.py --prewarm`; add `--packed-first`
for the reversed session. The [machine-readable report](./sov-transcribe-packed-weights-2026-11-14.json)
contains the complete parameter digests, manifest, measurements, input identity,
source hashes, and binary identity.
