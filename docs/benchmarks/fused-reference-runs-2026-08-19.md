# Fused Comprehensive reference runs: Laguna XS 2.1 and Nemotron Lightning

These are completed local reference receipts captured on August 18–19, 2026.
They are **not** upstream leaderboard scores. They measure the pinned Mere
Comprehensive subset, scorers, sampling profiles, and runtime identified below.
Future runner, fixture, sandbox, or model changes require a new dated receipt;
they do not silently replace these results.

## Overall results

Both models completed all 110 cases across five sampled trials. The 50 vision
case-trials were unscored for both models because their catalog profiles did not
advertise image input. Strict pass rate and mean score therefore use the 500
scored rows as their denominator.

| Model | Completed | Scored | Unscored | Strict passes | Strict pass rate | Mean score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laguna XS 2.1 | 550/550 | 500 | 50 | 247/500 | 49.4% | 0.6473 |
| Nemotron 3.5 Lightning | 550/550 | 500 | 50 | 172/500 | 34.4% | 0.5794 |

`passed` is the all-or-nothing case contract. `score` retains partial credit
where a scorer supports it, so strict pass rate and mean score answer different
questions. Unscored capability rows are reported separately and are not
converted into failures.

## Results by source family

Each cell shows `strict passes / scored rows (rate) · mean score`. Counts are
case-trials, not unique prompts.

| Source family | Rows | Laguna XS 2.1 | Nemotron Lightning |
| --- | ---: | ---: | ---: |
| Mere authored chat | 260 | 155/260 (59.6%) · 0.8478 | 105/260 (40.4%) · 0.7448 |
| Mere authored tools | 50 | 37/50 (74.0%) · 0.9410 | 10/50 (20.0%) · 0.7517 |
| BFCL v3 | 50 | 20/50 (40.0%) · 0.4000 | 7/50 (14.0%) · 0.1400 |
| OpenAI HumanEval | 15 | 10/15 (66.7%) · 0.6667 | 12/15 (80.0%) · 0.8000 |
| EvalPlus HumanEval+ | 30 | 5/30 (16.7%) · 0.1667 | 6/30 (20.0%) · 0.2000 |
| EvalPlus MBPP+ | 30 | 11/30 (36.7%) · 0.3667 | 23/30 (76.7%) · 0.7667 |
| LiveCodeBench | 30 | 8/30 (26.7%) · 0.2667 | 8/30 (26.7%) · 0.2667 |
| LongBench v1 | 35 | 1/35 (2.9%) · 0.0618 | 1/35 (2.9%) · 0.0706 |
| Mere authored vision | 50 | 0 scored; 50 unscored | 0 scored; 50 unscored |

On this receipt, Laguna led the overall, authored-chat, authored-tool, and BFCL
strict-pass results. Nemotron led HumanEval and MBPP+. The two models had the
same strict-pass count on LiveCodeBench and LongBench, although their generated
answers and partial scores were not identical.

## Shared run contract

| Field | Recorded value |
| --- | --- |
| Host | Apple M4 Max, 16 logical processors, 128 GiB unified memory, arm64 |
| Operating system | macOS 26.5.2, build 25F84 |
| mere.run | 0.40.1 |
| Runner executable | 204,648,000 bytes; SHA-256 `71cb98eeda7fb71f6e92399de7753faadcfc84c5b6b6f19da107a2d7e7633b5f` |
| Suite manifest | `mere-fused-v1` 1.2.0; SHA-256 `4db423313708db492caf9e5eb70ab0fb2e18021a1eb0dee74c83b3b2b5d5e954` |
| Plan shape | 110 cases × 5 trials × 1 native model profile = 550 rows |
| Quality lane | Sampled final target, exact policy, logprob summaries |
| Context | 32,768 tokens |
| Code execution | `/usr/bin/python3`, 5-second timeout, `macos-sandbox-exec` |
| Response capture | Disabled; the receipts do not contain visible model responses |
| Duplicate rows | Zero in each schema-v2 checkpoint |

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
- The two text-only catalog profiles skipped vision by capability declaration.
  Compare their scored text/code/tool rows; do not call the vision lane failed.
- Nemotron was paused and resumed. Its checkpoint timestamps are a preservation
  window, not a throughput measurement. Logprob capture also makes these
  quality runs unsuitable as performance benchmarks.
- Failing code-row diagnostics on this host often included an `xcrun_db` cache
  warning before the candidate's Python syntax, name, assertion, or test error.
  Passing code rows were still recorded. Treat any rerun after sandbox or
  toolchain changes as a new receipt rather than rewriting this historical one.
- The raw checkpoints and normalized external fixtures remain local, ignored
  artifacts. Their hashes are recorded here so a later summary cannot silently
  substitute different result bytes.

See [Mere fused model evaluation](../benchmark-fused.md) for the source mix,
question and answer examples, scoring semantics, and reproduction procedure.
