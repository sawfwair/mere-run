# Portable SOV Q8 bundle qualification

The portable bundle loads the existing native Q8 weights without the original
BF16 decoder, a quantization pass, source file timestamps, or a conversion-cache
directory. It includes the original audio encoder/projector, tokenizer/configuration,
and an optional unchanged DFlash checkpoint in `draft/`.

- Model repository: [Sawfwair/SOV-Transcribe-1-mere-run-q8](https://huggingface.co/Sawfwair/SOV-Transcribe-1-mere-run-q8).
- Qualified runtime: `6385b49f5bd5bd021ff5de3fc68b719fe0d8ac7d`.
- Hardware: M5 Max, 40 GPU cores, 128 GiB, macOS 27.2 (26B123).
- Size: about 29.14 GiB including DFlash. Large artifacts remain on external storage.
- Precision: affine Q8 group-64 MoE experts; non-expert target parameters retain
  their original bytes and dtypes, including FP32 routers.

## Exact parity

The installed GPU test compared every loaded target and audio parameter with
the original checkpoint and existing native Q8 cache. All **777 target**, **487
encoder**, and **2 projector** tensors match byte-for-byte. The copied draft
configuration and weights match their pinned upstream files exactly.

Three recordings, at 3.52, 46.57, and 93.14 seconds, each ran through the cache
and portable loaders with one warmup and two measured repetitions. All **18
outputs** have identical tokens, text, proposal/acceptance lengths, verification
counts, and reused verification counts. The portable case selected Q8 Metal
automatically without a decoder flag. All measured kernels selected the
qualified M5 distributed reduction and pipelined DFlash scheduling.

Four public CLI runs, comparing both formats with and without DFlash, produce
the same short-fixture transcript. Bundle file timestamps were changed before
qualification. A synthetic regression also deletes the original source/cache
directories before loading the portable copy and rejects duplicate tensors.

## Loading and memory observations

Both formats report 29.109 GiB peak MLX memory during loading, including DFlash.
Portable loading took 2.950–57.182 seconds across these fresh processes; cache
loading took 2.708–37.882 seconds. These ranges include different file-cache
conditions and different external drives. A repository gate also ran on the
shared host. They are diagnostic observations and do not establish a storage-
controlled startup speedup or regression. The numerical representation and
resident kernel selection are unchanged.

## Validation and provenance

`./scripts/check.sh` passes at the qualified runtime, including strict lint,
package policies, the full test suite, CLI help, and hygiene checks. The three
Python exporter tests pass. The installed full-tensor test passes with zero
failures. Hugging Face checksum verification checked 24 original target files,
2 draft files, and all 31 files in the initial private bundle upload.

The model repository includes Apache 2.0, upstream attribution and modification
notices, the original Whisper MIT notice, pinned source revisions, a portable
manifest, and full-file SHA-256 checksums. Its card identifies this as a native
mere.run format and requires the supporting runtime revision.

The [machine-readable receipt](./sov-transcribe-portable-bundle-2026-11-15.json)
includes per-parameter target hashes, per-recording measurements, binary and
manifest hashes, and validation log hashes. Exact parity is relative to native
Q8; quantization can change output relative to BF16. Broad multilingual accuracy
and other hardware remain unqualified.

See the [usage and export guide](../runtime/sov-transcribe.md#download-a-portable-q8-bundle).
