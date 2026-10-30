# Nemotron Transcribe BF16 versus routed-expert INT4, 2026-10-28

Native Swift/MLX transcription was compared on twelve recordings totaling
177.73 seconds and 522 reference words on an Apple M5 Max with 128 GiB unified
memory and macOS 27.2. Both formats used the same release CLI and greedy
prompt/decoding settings. Every measured CLI run published a durable
`succeeded` result. See `report.json` for all transcripts, errors, times,
memory measurements, source hashes, and checkpoint conversion receipts.

## Accuracy and size

| Metric | Original BF16 | Routed-expert INT4 |
| --- | ---: | ---: |
| Word errors / reference words | 2 / 522 | 4 / 522 |
| Aggregate WER | 0.383% | 0.766% |
| Weight file bytes | 67,496,013,734 | 25,067,694,940 |
| Largest observed process memory footprint | 61.74 GiB | 21.83 GiB |
| Largest observed maximum resident size | 54.38 GiB | 21.10 GiB |

WER is word-level Levenshtein distance after lowercasing, normalizing curly
apostrophes, and ignoring punctuation. Apostrophes inside words are preserved;
spelling variants count as errors. INT4's additional substitutions were
`Kaliko` → `Calico` and `nor` → `or`. Both formats emitted `scepter` for the
reference's `sceptre`, which counts as an error under this strict metric.
Aggregate WER counts each recording once (the first measured run), so
repeated timing cases don't receive extra weight.
The earlier standalone BF16 receipt used a different binary and the runtime's
default TF32 setting; this paired comparison reruns both formats with TF32 disabled.

| Recording | Seconds | Reference words | BF16 errors | INT4 errors |
| --- | ---: | ---: | ---: | ---: |
| short | 3.52 | 11 | 0 | 0 |
| human | 46.57 | 129 | 1 | 2 |
| 84-121550-0013 | 19.11 | 45 | 0 | 0 |
| 174-168635-0002 | 15.86 | 42 | 0 | 0 |
| 251-118436-0003 | 11.02 | 33 | 0 | 0 |
| 777-126732-0005 | 11.12 | 40 | 0 | 0 |
| 1462-170142-0005 | 11.08 | 37 | 0 | 0 |
| 1988-147956-0000 | 14.95 | 39 | 0 | 0 |
| 1993-147964-0005 | 10.52 | 36 | 0 | 0 |
| 2035-147960-0000 | 9.02 | 33 | 0 | 0 |
| 2412-153948-0000 | 11.66 | 36 | 0 | 1 |
| 2428-83699-0000 | 13.30 | 41 | 1 | 1 |

## End-to-end latency and peak memory

The short sentence and the 46.57-second, two-speaker recording were each
measured three times per format. Other recordings were measured once per
format: 32 scored CLI runs plus one unscored short-clip warm-up per format.
The table reports median wall time and the largest process memory footprint
among the three runs. All individual values are retained in `report.json`.

| Recording | BF16 median | INT4 median | Speedup | BF16 peak | INT4 peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 14.94 s | 3.28 s | 4.56× | 61.39 GiB | 21.23 GiB |
| human | 46.74 s | 6.45 s | 7.25× | 61.65 GiB | 21.83 GiB |

Each run starts a fresh CLI process. Wall time includes checkpoint loading,
audio decoding/encoding, language prefill, token decoding, and durable output
publication. Format order alternates by case and repetition. File and shader
caches were not cleared. `MLX_ENABLE_TF32=0` was set for both formats. No
concurrent build or test workload ran during measurement. These are complete
CLI comparisons, not isolated GPU throughput measurements. Packed prefill
processes 128 positions together; BF16 uses four-position gathers to bound
materialized expert weights. Memory values come from macOS `/usr/bin/time -l`;
process footprint and maximum resident size are distinct metrics.

## Quantization and integrity

The source is NVIDIA's BF16 revision
`ef905575a106fdad904f3b0ce661dccad654607d`, verified against SHA-256
`03142dde5176c1d4df21ee829642745a1d670ea4d79e1a4684c64d53195acff2`.
The converter uses MLX 0.32.0 affine 4-bit weights, group size 64, and BF16
scales/biases. It transposes the 46 main-decoder routed projections from
`[expert, input, output]` to `[expert, output, input]` before packing.
`gatherQuantizedMM` consumes those weights without expanding full expert banks.

All other 1,223 tensors were independently verified bit for bit against the
source (8,544,190,336 bytes), including audio encoders, routers,
shared experts, Mamba layers, and retained but unexecuted MTP tensors.
The result contains 1,361 tensors. Each output shard's SHA-256 and size are
recorded in the conversion receipt embedded in `report.json`. Original model
licenses are retained. This is a local MLX affine conversion, distinct from
NVIDIA's NVFP4 release.

The two installed checkpoints passed the exact-sentence regression. Packed
expert tests cover repeated/out-of-order selection and prefill/decode shapes,
with a strict FP32 dequantized oracle and device-specific BF16 rounding bounds.
MLX's scalar CPU BF16 accumulator rounds after each product; Metal accumulates
in FP32. Loader tests reject missing, mixed, wrong-shape, and wrong-dtype
packed tensors and unsupported metadata.

The full `./scripts/check.sh` gate passed with native SwiftPM: 5,612 XCTest
cases (470 optional skips), zero failures, and 196 Swift Testing cases passed.
Lint, package policy, CLI help/hygiene, documentation examples, and the docs
build also passed. All 23 focused Metal tests passed, including both installed
checkpoints; the packed expert tests also passed on CPU.

## Reproduce

From the repository root (the comparison script requires Python 3.11 or later):

```bash
swift build --build-system native -c release --product mere.run
uv run scripts/model-conversion/quantize_nemotron_transcribe_experts.py \
  --source "$HOME/Library/Application Support/MereRun/models/speech-asr-nemotron-35-transcribe-bf16" \
  --output /absolute/path/to/experts-int4
python3 scripts/validation/compare_nemotron_transcribe.py \
  --bf16 "$HOME/Library/Application Support/MereRun/models/speech-asr-nemotron-35-transcribe-bf16" \
  --int4 /absolute/path/to/experts-int4 \
  --archive /absolute/path/to/dev-clean-2.tar.gz \
  --output /absolute/path/to/new-comparison --repeats 3
```

The original checkpoint requires explicit installation/license acceptance.
The native SwiftPM build system avoids the existing missing vendored llama
dSYM path in the default Xcode build system. No vendor files changed.

The additional recordings use [Mini LibriSpeech SLR31](https://www.openslr.org/31/)
under CC BY 4.0. The publisher archive's MD5 is
`6d7ab67ac6a1d2c993d050e16d61080d`. Selection is deterministic: the lowest ten
numeric speaker IDs excluding 1272/5694, then each speaker's first
lexicographic utterance with 30–60 reference words. The existing human fixture
uses those two excluded speakers and crosses the 30-second encoder boundary;
the short sentence is the existing macOS Samantha fixture. References were
used only for selection/scoring, never provided in the model prompt.

These small English cases do not establish general accuracy equivalence.
No noisy/far-field speech, multilingual or translation accuracy, hour-long
recordings, or minimum-memory hardware configuration was qualified.

## Publication and managed-default integration

The [public INT4 checkpoint](https://huggingface.co/Sawfwair/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-MLX-4bit)
is pinned at `c20414875a76f4b9de213a4a084c9731069a29b3`.
`huggingface-publication.json` records the verified public file closure:
46 files, 25,085,565,732 bytes, with all weight and metadata SHA-256 values.

The managed default is `speech-asr-nemotron-35-transcribe` (INT4), with
`speech-asr-nemotron-35-transcribe-bf16` for the original checkpoint.
A real managed force-pull resolved the public immutable revision and reused
local payloads only after checking them against remote metadata. All 45
installed published files except the locally rewritten consent manifest were
SHA-256 verified again. The legacy BF16 install was preserved under its
explicit ID before replacement. Both the default INT4 route and explicit BF16
route exited zero and emitted the exact 11-word reference sentence.

`managed-install-validation.json` records these integration checks separately
from the earlier latency benchmark. Both private durable run receipts were
verified as `succeeded`, with the expected effective model IDs. The final
`./scripts/check.sh` gate passed with native SwiftPM: 5,617 XCTest cases
(470 optional skips), zero failures, and 196 Swift Testing cases passed.
All 13 focused Metal tests passed, including both installed checkpoints and
the five preservation/completeness regressions. Documentation build and
example checks passed.
