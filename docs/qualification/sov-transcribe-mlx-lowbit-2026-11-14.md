# SOV calibrated expert precision — 2026-11-14

This study tests native Q4/Q6 expert packing and per-layer Q4/Q6/Q8 recipes
against the existing Q8 Metal path. Attention, dense MLPs, routers, embeddings,
and DFlash retain their original weights. Ordinary transcription still uses
BF16; lower precision is selected explicitly in the resident benchmark.

Lower-bit packing reduces memory but has not established a faster qualified
precision policy. Uniform Q6 uses 18.3% less prepared active memory than Q8.
Across 140 held-out clips it adds two word errors. The `q6-matrix` experiment
caches the same quantized values for M5 prefill, but uses more memory, adds five
word errors, and loses the repeated paired latency comparison. These findings
do not support changing the production precision default.

The [JSON receipt](./sov-transcribe-mlx-lowbit-2026-11-14.json) retains native
recognition outputs, dataset manifests and hashes, calibrated recipes, raw
paired reports, source and binary hashes, and validation receipts.

<!-- results -->

## Recognition and memory results

The initial set has 1,703 reference words. Prepared active memory is the native
MLX allocation after loading and packing; it excludes request workspaces and
does not describe peak loading memory. The original BF16 checkpoint is still
loaded before packing.

| Policy | Prepared active memory | Initial WER | Word errors relative to Q8 |
| --- | ---: | ---: | ---: |
| Original BF16 weights | 49.05 GiB | 3.76% | -1 |
| Q8 Metal | 29.11 GiB | 3.82% | reference |
| Uniform Q6 | 23.79 GiB | 3.82% | 0 |
| Uniform Q4 | 18.47 GiB | 4.11% | +5 |
| `mixed-001` (all Q8 control) | 29.11 GiB | 3.82% | 0 |
| `mixed-002` | 26.80 GiB | 3.88% | +1 |
| `mixed-006` | 23.26 GiB | 3.93% | +2 |
| `mixed-048` | 24.15 GiB | 3.88% | +1 |

The all-Q8 recipe reproduces all 70 Q8 token sequences. Uniform Q6 reproduces
66/70; its equal initial WER includes one additional German error and one fewer
Dutch error. Aggregate equality is not language-by-language equivalence.

Q6 was selected for confirmation before examining the second set's predictions.
That set has 1,709 reference words: Q8 scores 2.87% WER and Q6 2.98%, two extra
errors. Across all 140 recordings, Q8 has 114 errors in 3,412 words and Q6 has
116. Their WERs are 3.34% and 3.40%; the paired clip-bootstrap interval for the
difference is -0.058 to +0.178 percentage points.

`q6-matrix` uses 45.06 GiB of prepared active memory. Across those same 140
recordings it has 119 word errors, or 3.49% WER. Its difference from Q8 is
+0.147 percentage points, with a bootstrap interval of -0.088 to +0.395. It
matches 126/140 Q8 token sequences and 130/140 uniform-Q6 sequences. No policy
hits the 448-token output limit. These samples do not establish general quality
equivalence, and uncertainty intervals are not evidence of identical behavior.

| Language | Reference words | Q8 WER | Q6 WER | Q6 matrix WER |
| --- | ---: | ---: | ---: | ---: |
| German | 473 | 1.48% | 1.69% | 1.48% |
| English | 424 | 3.07% | 3.07% | 3.30% |
| Spanish | 530 | 2.26% | 2.26% | 2.26% |
| French | 597 | 6.53% | 6.53% | 6.53% |
| Italian | 473 | 2.33% | 2.33% | 2.11% |
| Dutch | 446 | 3.14% | 3.14% | 3.59% |
| Portuguese | 469 | 3.84% | 4.05% | 4.48% |

## Paired release precision results

Positive reductions mean lower candidate latency relative to Q8 Metal with the
same optimized attention and rollback generation. The repeated `q6-matrix`
study has two warmups per mode, six pairs in the first process, and eight in
the second. The second process reverses recording and starting-mode order.
These are two separately resident models. The combined active allocation is
74.17 GiB. No compiler or profiler from this task runs during these measurements.

| Audio | Resident reduction | Prefill reduction | Generation reduction | Wins/pairs |
| --- | ---: | ---: | ---: | ---: |
| 46.57 s | -13.16% | -7.01% | -13.85% | 4/14 |
| 3.52 s | -1.33% | -0.42% | -9.84% | 4/14 |
| 93.14 s | -9.10% | -8.78% | -7.54% | 5/14 |
<!-- paired-table-end -->

Both models have substantial order effects and background timing drift. The
table reports medians of adjacent-pair ratios; pooled absolute medians can
reverse direction and are not used to claim a gain. None of these comparisons
establishes a faster `q6-matrix` path. Its original short and long fixture tokens
match the Q8 baseline, but its wrapped-recording transcript differs.

The four-pair exploratory long-clip screen gives uniform Q6 a noisy 4.06%
resident reduction, including a 42.28% prefill regression and 13.65% generation
reduction. The Q4/Q8 hardware-aware recipe gives 1.19%, the conservative Q6/Q8
recipe loses 13.30%, and uniform Q4 loses 1.25%. These small screens do not
qualify additional speedups; all raw paired values are retained.

<!-- residency-results -->

## Bounded Metal residency

The pinned allocator initializes its explicit wired limit to zero. SOV had no
wired-memory ticket. This prompted a separate experiment; zero is not proof of
paging or a defect in implicit residency. The benchmark requests active MLX
memory plus 4 GiB, capped below the device's recommended working set, through
MLX's shared wired-memory manager.

The same-model comparison alternates default and wired scopes. Ticket start/end
transitions are outside the measured interval, and no zero-sized outer ticket
remains active during an on/off comparison. This permits full baseline
restoration between modes rather than retaining the limit through the manager's
active-work shrink cooldown. Static `--wire-models` holds its scope across all
sampling, including warmups. Normal transcription defaults are unchanged.

| Target | Audio | Requested budget | Resident reduction | Wins/pairs | Token agreement |
| --- | --- | ---: | ---: | ---: | --- |
| Q8 Metal | 46.57 s | 33.11 GiB | -0.33% | 6/14 | exact |
| Q8 Metal | 3.52 s | 33.11 GiB | -0.10% | 6/14 | exact |
| Q8 Metal | 93.14 s | 33.11 GiB | 0.12% | 7/14 | exact |
| BF16 | 46.57 s | 53.05 GiB | -0.37% | 2/6 | exact |
| BF16 | 3.52 s | 53.05 GiB | -3.78% | 1/6 | exact |
| Q6 matrix | 46.57 s | 49.06 GiB | 2.69% | 5/6 | exact within Q6 matrix |

Q8 combines six plus eight pairs in reversed fresh processes. BF16 uses original
weights, reference attention, and production pipeline scheduling; each recording
has six pairs. The Q6-matrix result is a six-pair exploratory comparison and does
not establish a repeated gain. All 129 recorded outputs in these on/off studies
agree within their respective model configurations.

With both Q8 and Q6 matrix explicitly budgeted together at 78.17 GiB, a separate
six-pair long-clip comparison still makes Q6 matrix 7.05% slower. Its prefill is
1.09% slower and generation 10.50% slower, winning one pair. The cached encoding
therefore does not establish a faster total path even under this residency
configuration. These stages do not isolate the precise kernel responsible.

A separate native API check verifies applied limits of 33, 53, and 78 GiB and
restoration to zero. That check changes the allowed budget without allocating
model-sized buffers; it confirms the mechanism rather than measuring page
residency or attributing timing gaps to swapping. The recorded benchmark values
are requested budgets. No residency policy is promoted to the default.

## Calibration

Calibration uses 14 audio clips from the `train` split of
[Google FLEURS](https://huggingface.co/datasets/google/fleurs), two each in
English, French, German, Spanish, Italian, Portuguese, and Dutch. FLEURS is
licensed CC BY 4.0. The original BF16 target processes audio and up to 64
teacher-forced transcript tokens. Each layer captures routed expert inputs,
indices, weights, and normalized expert outputs at 32 evenly spaced prompt
positions and up to 16 transcript positions.

For each decoder layer, the native runtime packs affine Q4, Q6, and Q8 expert
weights in groups of 64. Calibration measures relative RMS output error after
routing, mixing, BF16 rounding, and the expert output normalization. Fixed
thresholds select the lowest precision meeting the local error bound; Q8 is the
fallback. The held-out recordings do not participate in this selection.

| Recipe | Error threshold | Q4 layers | Q6 layers | Q8 layers |
| --- | ---: | ---: | ---: | ---: |
| `mixed-001` | 1% | 0 | 0 | 30 |
| `mixed-002` | 2% | 0 | 13 | 17 |
| `mixed-006` | 6% | 3 | 27 | 0 |
| `mixed-048` | 8%, Q4/Q8 only | 14 | 0 | 16 |

Across layers, calibration relative RMS error ranges from 5.22–13.89% at Q4,
1.53–3.49% at Q6, and 0.72–1.34% at Q8. These are local normalized-output
errors, not recognition-error predictions. In particular, the conservative
mixed recipe does not necessarily outperform uniform Q6 on held-out recognition.
The Q4/Q8-only recipe was added after inspecting M5 dispatch: it trades larger
byte savings in fewer layers against keeping Q8 in the more sensitive layers.
Its threshold is selected from training scores, without using confirmation
predictions. `select_sov_expert_recipe.py` reproduces selection from native scores.

## Recognition protocol

The first evaluation uses 70 FLEURS `test` clips, ten per language. A second set
uses ten clips per language starting at row offset 100. Source-audio hashes
confirm no overlap between calibration, initial evaluation, and confirmation.
The manifests retain sample IDs, source URLs, references, and audio hashes.

All policies use native MLX audio processing, DFlash, optimized attention, and
rollback KV storage. The BF16 control keeps the original target parameters
with those same benchmark settings. Language is detected without a language
hint. Each model is prepared once. One unmeasured warmup precedes the selected
recordings. Recognition runs use the native Debug test harness;
their timings are not release performance measurements. Some additional quality
runs overlap CPU release compilation, with no other GPU workload from this task.

WER normalization uses Unicode NFKC, case folding, punctuation replaced with
spaces, and whitespace tokenization. It preserves letters, combining marks,
and numbers. This is a documented comparison normalizer, not the publisher's
benchmark implementation. Paired bootstrap intervals resample whole recordings
2,000 times with seed zero. Small language subsets and wide uncertainty remain;
equal aggregate WER can hide language-specific changes.

## Native implementation

`Gemma4SwitchLinear` and `Gemma4SwitchGLU` pack Q4/Q6/Q8 through the existing
MLX affine format. The model preparation API accepts a precision schedule and
validates it before replacing expert modules. Packing is outside timed inference.
Mixed group-64 Q8 layers retain the custom Metal verification kernel; lower-bit
layers use MLX. Routers and non-expert parameters are unchanged.

`SOVExpertRecipe` uses typed decoding. Schema 1 requires group size 32 or 64 and
one precision of 4, 6, or 8 per decoder layer. The generator rejects missing,
incompatible, or extraneous recipes before loading checkpoint tensors. Recipes
and their calibration are specific to the target weights; model changes require
recalibration. The benchmark records the recipe file's SHA-256.

`--interleave-experts` compares a Q4/Q6/mixed candidate with a separately
resident Q8 Metal control. Both have identical encoder, attention, generation,
and request-cleanup policies. Adjacent pairs alternate order, and
`--experts-first` reverses starting order. Two resident models consume more
memory than normal single-model inference; JSON records combined active memory
and baseline preparation time. Loading and packing are excluded from paired
resident timing.

The pinned MLX dispatcher enables the M5 matrix path for affine Q8. Q4/Q6 use
other kernels. Fewer weight bytes therefore do not guarantee lower prefill
latency. This is a source-level dispatch finding; measured stages determine
its end-to-end effect.

`q6-matrix` addresses that dispatch gap by expanding four six-bit integer codes
from three bytes into four eight-bit lanes with a native Metal packing kernel.
It retains Q6 codes for verification and shares their scales and biases with
the expanded encoding. Large sorted prefill batches use the Q8 matrix backend.
Dequantized weight values match exactly in the tests; this does not guarantee
identical matrix accumulation or model predictions. A private non-Module cache
keeps derived arrays out of checkpoint parameters. Explicit parameter-update
invalidation is required because MLX updates an array's backing context without
changing its Swift identity. Tests cover that refresh and canonical Q6
verification arithmetic.

## Reproduce

```bash
swift build -c release --product mere.run -Xswiftc -disable-cmo
.build/release/mere.run model benchmark sov \
  Tests/MereRunCoreTests/Fixtures/Nemotron3/returning-speaker.wav \
  --model-root .build/qualification/sov/target \
  --draft-model .build/qualification/sov/draft \
  --experimental-decoder q6 --experimental-attention \
  --generation-experiment rollback --interleave-experts \
  --warmups 2 --repetitions 8 --json
```

For calibrated precision, replace `q6` with `mixed` and add
`--expert-recipe path.json`. The recipe has this form, with one entry per layer:

```json
{"schemaVersion":1,"groupSize":64,"layerBits":[8,6,6]}
```

The three-entry example illustrates the format; the released SOV target needs
30 entries. `prepare_sov_lowbit_data.py` in `scripts/model-conversion/` downloads
only selected train/test audio and records hashes. It resolves each public
Parquet URL once to avoid repeating Hub redirects for every page request.
`summarize_sov_lowbit.py` in `scripts/validation/` computes the recorded WER and
bootstrap comparisons.

Native calibration uses the optional `SOVLowBitProfileTests` with
`MERERUN_TEST_MLX_DEVICE=gpu`, `MERERUN_TEST_SOV_PATH`,
`MERERUN_SOV_CALIBRATION`, and `MERERUN_SOV_LOWBIT_OUTPUT`. Quality evaluation
uses `MERERUN_TEST_SOV_DFLASH_PATH`, `MERERUN_SOV_EVALUATION`,
`MERERUN_SOV_EXPERT_POLICY`, `MERERUN_SOV_QUALITY_OUTPUT`, and an optional
`MERERUN_SOV_EXPERT_RECIPE`. These checkpoint tests skip in ordinary local gates.

No second checkpoint or persistent packed model was created. Selected audio
and receipts occupy a small fraction of the existing worktree assets. The
original checkpoint remains necessary for load-time packing.

## Validation and limits

- Apple M5 Max, 128 GiB; macOS 27.2, Xcode 27.2. Target revision
  `505d8fae243924c0d8acdc622f7f0abab344eba3`, draft revision
  `8c0b3f3a98507d2e3dd35b1bee0f147fa3bded05`, MLX Swift revision
  `1e077ba968b291e0237d1ab522b271d433fa1713`.
- Twenty-four focused GPU regressions passed after the packing-cache change.
  The final 18 CLI tests, including native budget application and restoration,
  passed. Three CPU model tests passed with one GPU-only skip; 17 final CPU CLI
  tests passed with the optional native check skipped.
- Exact operator checks cover Q4/Q6/Q8 packing, the 704-wide tail, expanded code
  weight parity, unchanged Q6 verification, prefill dispatch, distinct per-layer
  precision, parameter replacement, and derived-cache exclusion from parameters.
- Release builds, CLI inventory generation, strict SwiftLint, Python syntax,
  and whitespace checks passed. Independent numerical unit checks disable TF32;
  checkpoint recognition and release measurements use default precision.
- The receipt contains 840 held-out recognition outputs, excluding 11 recorded
  recognition warmups, and 277 recorded CLI outputs. Warmups not retained in the
  CLI reports are excluded from those counts. Precision changes are evaluated
  by recognition error; no claim of universal frozen-token parity is made.
- The new working assets occupy about 0.1 GiB, and `.build` remains about 80 GiB.
  No packed checkpoint or duplicate source checkpoint is persisted.
- The full repository gate remains deferred at the user's request. This study
  does not qualify other Apple Silicon devices, all languages' full evaluation
  sets, or a new production default.
