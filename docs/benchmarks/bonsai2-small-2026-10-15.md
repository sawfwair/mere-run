# Bonsai 2 Small native checkpoint qualification

These results cover the pinned 2B, 4B, and 9B MLX checkpoints through the native Swift/MLX release runtime. Python orchestrates requests and scores saved outputs; it does not execute model inference.

The later [upstream parity investigation](./bonsai2-small-upstream-parity-2026-10-16.md) found native vision discrepancies and changed the runtime. The measurements below belong to the original executable identified here; they do not qualify the corrected implementation. Fresh measurements of the corrected executable are in the [fixed-runtime rerun](./bonsai2-small-fixed-2026-10-16.md).

The [machine-readable summary](./bonsai2-small-2026-10-15-summary.json) contains redacted copies of the cases and selected outputs, fixture hashes, and timing samples. Original prompts and outputs remain unchanged on T7; response hashes refer to those originals.

## Scope and provenance

- Host: Apple M3 Max (14 CPU cores, 30 GPU cores), 36 GiB unified memory, macOS 26.6.2.
- Model runtime implementation: `1f1b1b852d618a7420661ea4658903ffb9e91092` on `codex/native-bonsai2-small`.
- Tested release executable SHA-256: `60de2728001459877ff32fe20ce26630b178e871aa06068a85a7f3dd8110b9cd`.
- Model payloads, metadata, prompts, images, requests, outputs, and timing receipts are retained under `/Volumes/T7/mere-run-qualification/bonsai2-small-2026-10-15`.
- All source files were checked against pinned upstream LFS SHA-256 or Git blob identities. The initial API cache copy was moved to T7 with a verified compatibility link.
- No conversion, extra quantization, MTP, admission override, or parallel inference was used.

| Model | Pinned revision | Model snapshot bytes |
| --- | --- | ---: |
| 2B | `1c6cbae21f8c38830ce337972b46fdae7b18eaaa` | 1,213,249,532 |
| 4B | `dff2caaa23fa8706164c44c10f1b58feb405f76a` | 1,884,415,990 |
| 9B | `f4e28a59531731c41b6dfac88cf271bd58f0a287` | 3,475,204,252 |

## Quality

One sampled trial per case uses the generation sidecar sampler: temperature 1.0, top-p 0.95, top-k 20, and min-p 0. No presence or frequency penalty is applied in this primary condition. The API uses the catalog thinking default and an 8,192-token context, with up to 4,096 generated tokens per quality case. Its profile rejects deterministic-seed and provider-thinking controls.

The frozen suite contains eight grounded cases, two generated Python functions with five tests each, four images, one arithmetic case, and one reasoning case. Image cases cover OCR/counting, a receipt, a bar chart, and a public-domain NASA photograph. Requests use no output grammar or response-format enforcement. Strict JSON scoring rejects fences, repeated objects, and unfinished reasoning.

| Model | Passed cases | Grounded | Code tests | Vision strict | Arithmetic | Reasoning strict |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 2B | 9/16 | 6/8 | 7/10 | 1/4 | Pass | Fail |
| 4B | 12/16 | 7/8 | 10/10 | 2/4 | Pass | Fail |
| 9B | 12/16 | 7/8 | 10/10 | 1/4 | Pass | Pass |

| Model | Median quality request (s) | Slowest request (s) | Requests reaching 4,096 tokens |
| --- | ---: | ---: | ---: |
| 2B | 5.59 | 32.99 | 2 |
| 4B | 9.00 | 219.78 | 2 |
| 9B | 35.38 | 259.77 | 2 |

Fence-only recovery is a separate diagnostic. It removes one complete outer Markdown fence without changing values or calling the model again. It does not change strict scores.

| Model | Vision after fence-only recovery | Correct vision fields after recovery |
| --- | ---: | ---: |
| 2B | 2/4 | 12/18 |
| 4B | 2/4 | 12/18 |
| 9B | 3/4 | 13/18 |

### Separate sampler and budget probe

A secondary run used the same four selected cases for every model: grounded JSON extraction, interval merging, shape OCR/counting, and the chart. It used presence penalty 1.0, repetition penalty 1.0, an 8,192-token output allowance, and a new resident session with a 16,384-token context. Temperature, top-p, top-k, and min-p stayed the same. The native presence penalty covers the full generated history; the model card prefers a 64-token window where supported. Multiple conditions changed, so this does not isolate the effect of the penalty. Primary scores above are unchanged.

| Model | Secondary passed cases | 8,192-token caps | Slowest request (s) | Failed case IDs |
| --- | ---: | ---: | ---: | --- |
| 2B | 0/4 | 1 | 71.98 | 2b-grounded-3, 2b-code-merge_intervals, 2b-vision-shapes, 2b-vision-chart |
| 4B | 1/4 | 1 | 545.28 | 4b-grounded-3, 4b-vision-shapes, 4b-vision-chart |
| 9B | 3/4 | 1 | 188.07 | 9b-vision-chart |

The code evaluator runs generated functions with closed builtins, no imports or introspection, a CPU limit, and a short wall timeout. Ordinary list/string methods are allowed. An initial evaluator restriction on `list.sort()` was corrected by rescoring the same saved outputs against the same five tests; no model answer was regenerated.

## Throughput

Each model and modality has one discarded 128-token warm-up followed by three serial resident requests. Text requests have a 512-token budget; image requests have a 256-token budget. Native `/runtime/status` snapshots supply per-request generation, prefill, and decode timings. Timing requests are deliberately capped and are not scored for answer completion. Decode TPS includes reasoning tokens and excludes model loading and prefill. Prefix reuse follows the server default; internal first-token timing can precede the visible final answer.

| Model | Text median decode tok/s | Text range | Vision median decode tok/s | Vision range |
| --- | ---: | --- | ---: | --- |
| 2B | 22.52 | 20.25–131.61 | 112.55 | 23.72–125.76 |
| 4B | 15.47 | 15.10–15.61 | 15.41 | 15.30–15.57 |
| 9B | 16.52 | 13.82–17.65 | 18.52 | 18.42–18.61 |

| Model | Modality | Median prefill (s) | Median internal first token (s) | Median request (s) |
| --- | --- | ---: | ---: | ---: |
| 2B | text | 0.001 | 0.037 | 22.74 |
| 2B | vision | 0.599 | 0.616 | 2.88 |
| 4B | text | 0.001 | 0.064 | 33.11 |
| 4B | vision | 1.073 | 1.124 | 17.65 |
| 9B | text | 0.001 | 0.064 | 31.01 |
| 9B | vision | 1.827 | 1.884 | 15.65 |

Measurements were collected on an interactive desktop, with process, memory-pressure, and thermal snapshots retained for every request. The collector lists other native inference and compiler processes; it does not prove that every desktop GPU workload was absent.

The primary quality run included substantial latency variation: one 4B grounded answer took about 220 seconds. A process sample and GPU telemetry were retained for that quality request; profiling was separate from throughput runs. Short warmed decode observations do not establish sustained long-context speed.

## Reproduction and remaining gates

The T7 evidence root includes `download.py`, `metadata/`, `download-receipt.json`, `fixtures/make-fixtures.py`, `fixtures/manifest.json`, `quality-cases.json`, `plan.json`, and every request and response receipt. Native `model repair-manifests` completed managed discovery for all three snapshots.

Use a fresh prepared evidence root for a new trial. Reusing the frozen root resumes saved quality results. Start the normally admitted loopback server before invoking the collector.

```bash
swift build -c release --product mere.run
python3 scripts/validation/qualify-bonsai2-small-api.py \
  --binary .build/release/mere.run \
  --evidence-root /Volumes/T7/mere-run-qualification/bonsai2-small-2026-10-15 \
  --phase quality
```

Start the loopback server with the explicit T7 model path, `--context-size 8192`, and `MERERUN_HUB_CACHE` pointing into the evidence root. Keep normal machine and request admission enabled. The shared benchmark command requires 32 GiB of admission headroom; the resident API acquired its normal 16 GiB class once and retained request admission throughout.

This is a bounded native smoke qualification. These failures do not distinguish checkpoint behavior from native-runtime discrepancies without an upstream parity comparison. Broad benchmarks, additional sampling trials, released-checkpoint parity against another runtime, the declared 262,144-token context, and smaller-memory hardware remain unqualified.

The natural-image fixture comes from the versioned [scikit-image astronaut asset](https://github.com/scikit-image/scikit-image/blob/v0.22.0/skimage/data/astronaut.png). Its [source documentation](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut) records NASA public-domain provenance. The generated fixture source and all image hashes are retained on T7.
