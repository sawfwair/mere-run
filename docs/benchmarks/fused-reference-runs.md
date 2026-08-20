# Fused Comprehensive reference runs

These are completed local reference receipts captured on August 18–20, 2026.
They are **not** upstream leaderboard scores. They measure the pinned Mere
Comprehensive subset, scorers, sampling profiles, and runtime identified below.
Future runner, fixture, sandbox, model, or profile changes require a new dated
receipt; they do not silently replace these results.

## Completed profile scope

Laguna XS 2.1 and Nemotron Lightning each completed their single native profile.
The Qwen3.8 plan declares low, medium, and xhigh reasoning profiles; this receipt
intentionally stops after all 550 low-profile rows. Medium and xhigh remain at
zero and are not represented by the score.

| Model and profile | Completed scope | Scored | Unscored | Strict passes | Strict pass rate | Mean score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 27B, low reasoning | 550/550 low; 550/1,650 plan | 550 | 0 | 404/550 | 73.5% | 0.8170 |
| Laguna XS 2.1, native | 550/550 | 500 | 50 | 247/500 | 49.4% | 0.6473 |
| Nemotron 3.5 Lightning, native | 550/550 | 500 | 50 | 172/500 | 34.4% | 0.5794 |

`passed` is the all-or-nothing case contract. `score` retains partial credit
where a scorer supports it, so strict pass rate and mean score answer different
questions. Unscored capability rows are reported separately and are not
converted into failures.

Qwen3.8 advertised image input and therefore scored all 50 vision rows. Laguna
and Nemotron did not, so their vision rows were unscored. The fair like-for-like
comparison removes Qwen's vision rows and uses the same 500 text, code, and tool
rows for every model:

| Model and profile | Non-vision strict passes | Strict pass rate | Mean score |
| --- | ---: | ---: | ---: |
| Qwen3.8 27B, low reasoning | 359/500 | 71.8% | 0.8031 |
| Laguna XS 2.1, native | 247/500 | 49.4% | 0.6473 |
| Nemotron 3.5 Lightning, native | 172/500 | 34.4% | 0.5794 |

## Results by source family

Each cell shows `strict passes / scored rows (rate) · mean score`. Counts are
case-trials, not unique prompts.

| Source family | Rows | Qwen3.8 low | Laguna XS 2.1 | Nemotron Lightning |
| --- | ---: | ---: | ---: | ---: |
| Mere authored chat | 260 | 199/260 (76.5%) · 0.9223 | 155/260 (59.6%) · 0.8478 | 105/260 (40.4%) · 0.7448 |
| Mere authored tools | 50 | 45/50 (90.0%) · 0.9800 | 37/50 (74.0%) · 0.9410 | 10/50 (20.0%) · 0.7517 |
| BFCL v3 | 50 | 20/50 (40.0%) · 0.4000 | 20/50 (40.0%) · 0.4000 | 7/50 (14.0%) · 0.1400 |
| OpenAI HumanEval | 15 | 15/15 (100%) · 1.0000 | 10/15 (66.7%) · 0.6667 | 12/15 (80.0%) · 0.8000 |
| EvalPlus HumanEval+ | 30 | 23/30 (76.7%) · 0.7667 | 5/30 (16.7%) · 0.1667 | 6/30 (20.0%) · 0.2000 |
| EvalPlus MBPP+ | 30 | 25/30 (83.3%) · 0.8333 | 11/30 (36.7%) · 0.3667 | 23/30 (76.7%) · 0.7667 |
| LiveCodeBench | 30 | 22/30 (73.3%) · 0.7333 | 8/30 (26.7%) · 0.2667 | 8/30 (26.7%) · 0.2667 |
| LongBench v1 | 35 | 10/35 (28.6%) · 0.2209 | 1/35 (2.9%) · 0.0618 | 1/35 (2.9%) · 0.0706 |
| Mere authored vision | 50 | 45/50 (90.0%) · 0.9560 | 0 scored; 50 unscored | 0 scored; 50 unscored |

Within these exact receipts, Qwen3.8 low had the highest strict-pass result in
every scored source family except BFCL, where it tied Laguna. This comparison
does not predict the unfinished Qwen medium or xhigh profiles and is not a claim
about the complete upstream benchmark collections.

## Shared run contract

| Field | Recorded value |
| --- | --- |
| Host | Apple M4 Max, 16 logical processors, 128 GiB unified memory, arm64 |
| Operating system | macOS 26.5.2, build 25F84 |
| mere.run | 0.40.1 |
| Runner executable | 204,648,000 bytes; SHA-256 `71cb98eeda7fb71f6e92399de7753faadcfc84c5b6b6f19da107a2d7e7633b5f` |
| Suite manifest | `mere-fused-v1` 1.2.0; SHA-256 `4db423313708db492caf9e5eb70ab0fb2e18021a1eb0dee74c83b3b2b5d5e954` |
| Per-profile shape | 110 cases × 5 trials = 550 rows |
| Quality lane | Sampled final target, exact policy, logprob summaries |
| Context | 32,768 tokens |
| Code execution | `/usr/bin/python3`, 5-second timeout, `macos-sandbox-exec` |
| Response capture | Disabled; the receipts do not contain visible model responses |
| Duplicate rows | Zero in each schema-v2 checkpoint |

### Qwen3.8 27B low-reasoning receipt

- Model id: `vision-chat-q38-27b`
- Artifact: `Qwen/Qwen3.8-27B`
- Artifact revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Runtime-manifest SHA-256:
  `c5813b1044dff6f7aec2464c31d829f96aba19671839b70d83d617a6a1c42ec0`
- Profile: `qwen3.8-native-low`; reasoning effort 0.2, thinking enabled,
  temperature 1.0, top-p 0.95, top-k 20, min-p 0
- Checkpoint window: `2026-08-20T01:15:56Z` to `2026-08-20T09:21:19Z`
- Completion scope: low 550/550; medium 0/550; xhigh 0/550
- Plan SHA-256:
  `64a9b0fa53478c177947101cb385ef73ac6025827fbb8be2d9031a719c13d5b2`
- Frozen receipt filename: `qwen38-low-comprehensive-b944c0e7.json`
- Receipt SHA-256:
  `c6b70396f183ec7183563d0b2789a32893841285e843cafb8d3ff796e18698f4`

### Laguna XS 2.1 receipt

- Model id: `text-chat-laguna-xs-2-1`
- Artifact: `poolside/Laguna-XS-2.1-NVFP4-mlx`
- Artifact revision: `841778bda563a36104dd521e37d99218e46f4f25`
- Runtime-manifest SHA-256:
  `213b9c6ee642824a2294715aa37c84ee5772e45ca2d671f3cc2aafa2e5c368f7`
- Profile: temperature 1.0, top-p 1.0, top-k 20, min-p 0.02
- Checkpoint window: `2026-08-19T12:47:30Z` to `2026-08-19T14:47:50Z`
- Plan SHA-256:
  `ba0c0785c17f857a58428c024eae9448130b749e39c4e525eb60910c16720af1`
- Receipt filename: `laguna-xs-2-1-comprehensive-b944c0e7.json`
- Receipt SHA-256:
  `6b8ddccd034cd00515becbb6f7eb31d12a2f8aae2ccb93d00cc7212237dd522d`

### Nemotron Lightning receipt

- Model id: `text-chat-nemotron-35-lightning`
- Managed artifact:
  `Sawfwair/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-MLX`
- Managed artifact revision:
  `6699e5fd3f0c5b392bb3f8bac2443276bb41958a`
- Pinned base-model revision:
  `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4@e0b753dc24903ad4d62f5696077da22020eca89a`
- Runtime-manifest SHA-256:
  `369da6748557f0a3a05256cbeea933e789b7a9b15c5e08403a24946ec6b453e7`
- Profile: temperature 1.0, top-p 0.95, top-k disabled, min-p 0
- Checkpoint window: `2026-08-18T18:20:13Z` to `2026-08-19T12:24:41Z`
- Plan SHA-256:
  `bd04d29ac3a67b23d9c23ffc9477b7284274adf22de14af65cd7ac14de5aa8a7`
- Receipt filename: `nemotron-lightning-comprehensive-b944c0e7.json`
- Receipt SHA-256:
  `b1631e7504c74fcf41a34b44c1969dc2d6984b75e43a3bdac29fe9894b3ecd68`

## Interpretation limits

- These results apply to the exact Mere subset and hashes above, not the full
  HumanEval+, MBPP+, LiveCodeBench, BFCL, or LongBench leaderboards.
- Qwen3.8 medium and xhigh were intentionally deferred. The Qwen checkpoint is
  complete for low reasoning and partial for its three-profile plan.
- The two text-only catalog profiles skipped vision by capability declaration.
  Compare their scored text/code/tool rows; do not call the vision lane failed.
- Nemotron and Qwen were paused and resumed. Their checkpoint timestamps are
  preservation windows, not throughput measurements. Logprob capture also makes
  these quality runs unsuitable as performance benchmarks.
- Failing code-row diagnostics on this host often included an `xcrun_db` cache
  warning before the candidate's Python syntax, name, assertion, or test error.
  Passing code rows were still recorded. Treat any rerun after sandbox or
  toolchain changes as a new receipt rather than rewriting a historical one.
- The raw checkpoints and normalized external fixtures remain local, ignored
  artifacts. Their hashes are recorded here so a later summary cannot silently
  substitute different result bytes.

See [Mere fused model evaluation](../benchmark-fused.md) for the source mix,
question and answer examples, scoring semantics, and reproduction procedure.
