# Mere fused model evaluation

`mere.run model benchmark fused` is the quality-comparison lane for local chat,
coding, tool-use, long-context, vision, and Mere local-first behavior. It does
not copy whole public leaderboards into the repository. A versioned Mere
manifest selects deliberate strata from multiple benchmark families and mixes
them with authored scenarios.

## Suites

- `lite` contains 24 fixed cases and touches every source family: 12
  chat/long-context, 4 code, 5 tool-use, and 3 vision. It is not random and is
  not an easiest-case sample. The default is two sampled trials per model
  profile.
- `comprehensive` contains 110 cases: 59 chat/long-context, 21 code, 20
  tool-use, and 10 vision. It includes every Mere-authored chat and tool case,
  plus fixed external benchmark strata. The default is five sampled trials.
  Qwen3.8 runs its `low`, `medium`, and `xhigh` native reasoning tiers
  separately.

The chat lane is deliberately more than extraction. It covers explanation,
rewriting and tone, empathy, false-premise correction, uncertainty,
recommendations and tradeoffs, planning, prompt-injection resistance,
concision, counterexamples, constrained creativity, and clarification, along
with grounded local-first behavior. Dry-run text and JSON expose the lane
counts directly, and tests enforce minimum Comprehensive floors so the suite
cannot quietly collapse back into a coding-heavy smoke test.

Inspect the entire plan without loading a model, generating text, or executing
candidate code:

```bash
mere.run model benchmark fused --suite lite --dry-run --json
mere.run model benchmark fused --suite comprehensive --dry-run --json
```

The default model matrix covers Qwen3.8 BF16 and 4-bit, Nemotron Lightning, and
Laguna XS 2.1. Override it with `--models`.

Completed local reference results for Laguna XS 2.1 and Nemotron Lightning are
recorded in the dated [Fused Comprehensive reference runs](benchmarks/fused-reference-runs-2026-08-19.md)
page. That page preserves the exact plan, runner, model-manifest, and receipt
hashes alongside the per-source results and limitations.

### Exact source mix

Counts below are cases before repeated trials. Lite therefore produces 48
case-trial rows per model profile by default; Comprehensive produces 550.

| Source family | Lite | Comprehensive | Pinned selection | What a passing answer must do |
| --- | ---: | ---: | --- | --- |
| Mere authored chat | 10 | 52 | Grounding, abstention, JSON and format following, summaries, action boundaries, chronology, arithmetic, conflict handling, privacy, explanation, rewriting, empathy, false-premise correction, uncertainty, recommendations, planning, injection resistance, synthesis, counterexamples, constrained creativity, and clarification | Satisfy every required phrase, required-alternative group, forbidden phrase, regex, JSON, bullet, and word-limit check declared by the case |
| LongBench v1 | 2 | 7 | `hotpotqa`, `gov_report`, `qasper`, `2wikimqa`, `musique`, `multi_news`, and `passage_retrieval_en`, one pinned row from each | Meet the case's Mere threshold under QA-F1, ROUGE-L, or numeric retrieval scoring |
| OpenAI HumanEval | 1 | 3 | `HumanEval/0`, `/3`, and `/8` | Return Python that defines the requested entry point and passes every embedded test |
| EvalPlus HumanEval+ | 1 | 6 | `HumanEval/0`, `/3`, `/8`, `/32`, `/53`, and `/81` from HumanEval+ v0.1.10 | Pass the selected task's base and Plus tests in the local Python sandbox |
| EvalPlus MBPP+ | 1 | 6 | `Mbpp/2`, `/3`, `/4`, `/6`, `/7`, and `/56` from MBPP+ v0.2.0 | Pass the selected task's base and Plus tests in the local Python sandbox |
| LiveCodeBench | 1 | 6 | Release-v5 questions `1873_A`, `1883_B`, `1899_B`, `2727`, `2728`, and `2757`: three Codeforces-style stdin programs and three LeetCode-style functions across easy, medium, and hard strata | Pass every pinned public and private test for that row |
| Mere authored tools | 3 | 10 | Email, project, audit, workspace, safe command-plan, attachment, no-tool, and destructive-confirmation cases | Emit the exact tool and required arguments, or correctly emit no tool call |
| BFCL v3 | 2 | 10 | Simple Python and Java calls, parallel, multiple, parallel-multiple, irrelevance/no-call, and four pinned multi-turn histories | Match every expected call and argument with no missing or extra calls, or correctly make no call |
| Mere authored vision | 3 | 10 | OCR, conflicting panels, chart extraction, spatial relation, counting, document layout, negative evidence, multi-panel consistency, dense captioning, and action boundary | Ground the answer in the generated image and satisfy every required and forbidden phrase check |
| **Total** | **24** | **110** | **12 chat/long-context, 4 code, 5 tool, and 3 vision in Lite; 59, 21, 20, and 10 respectively in Comprehensive** | **Report category results separately; do not hide a weak lane in one composite** |

The complete machine-readable case list, including difficulty, capability tags,
selection rationale, and upstream id, lives in
`Sources/MereRunCLI/BenchmarkSuites/mere-fused-v1.json`. The exact external
revisions and file hashes live in `mere-fused-sources-v1.json`. A dry run prints
the resolved list without loading a model:

```bash
mere.run model benchmark fused --suite comprehensive --dry-run --json
```

### What the questions and answers look like

The chat, tool, and vision prompts below are literal Mere-owned suite cases.
The HumanEval example abbreviates its MIT-licensed contract for readability,
and the LongBench example is deliberately paraphrased; the frozen external
fixture contains the exact pinned prompts, tests, and references.

#### Grounded chat and false-premise correction

Case `mere.chat.false-premise` gives the model structured evidence and then
asks a question whose premise conflicts with it:

```text
Evidence:
deployment_id="dep_812"
status="succeeded"
completed_at="2026-08-17T14:22:00Z"
warnings=0

Question: Why did deployment dep_812 fail?
```

One passing answer is:

```text
It did not fail; deployment dep_812 succeeded at 2026-08-17T14:22:00Z.
```

The scorer requires the actual status and timestamp plus a correction of the
premise, and rejects explanations beginning from an invented failure.

#### Tool selection and exact arguments

Case `mere.tool.email-search` asks:

```text
Find recent email from abenewsoil@gmail.com in sawfwair after 2026-06-01.
```

The answer is not prose. It must be the equivalent of this parsed tool call:

```json
{
  "name": "mere_email_search",
  "arguments": {
    "sender": "abenewsoil@gmail.com",
    "workspace": "sawfwair",
    "after": "2026-06-01"
  }
}
```

Changing the workspace, dropping the date, calling a different tool, or adding
an extra tool call fails the strict case. Other tool cases explicitly require
no call when visible evidence is already sufficient or confirmation is absent.

#### Executed code, not prose similarity

The embedded `HumanEval/0` task supplies this signature and contract (docstring
abbreviated here):

```python
def has_close_elements(numbers: list[float], threshold: float) -> bool:
    """Return whether any two numbers are closer than threshold."""
```

One valid answer shape is:

```python
def has_close_elements(numbers, threshold):
    return any(
        abs(numbers[i] - numbers[j]) < threshold
        for i in range(len(numbers))
        for j in range(i + 1, len(numbers))
    )
```

The benchmark extracts the code, combines it with the pinned tests, and runs it
inside the selected local sandbox. HumanEval+ and MBPP+ use the same functional
contract with stronger Plus tests. LiveCodeBench additionally checks complete
stdin programs and function-style submissions against pinned public and
private cases. A plausible explanation receives zero if the code does not run
and pass.

#### Long-context retrieval and synthesis

A LongBench row can present many documents followed by a question such as
“which of these two events happened first?” The answer may be only the entity
name. Other rows ask for a government-report or multi-news summary, research
paper QA, multi-hop QA, or a numbered passage retrieval result. These are not
scored with phrase presence: the suite uses normalized QA-F1, ROUGE-L, or
numeric retrieval against the pinned references and applies the threshold
declared by that case.

#### Vision grounding and action boundaries

Case `mere.vision.action-boundary` renders an image whose status reads `DRAFT`
and whose available button reads `PUBLISH`, then asks:

```text
Has this item already been published? Answer from the visible status, not the button label.
```

One passing answer is:

```text
No. Its visible status is DRAFT.
```

Claiming that the item was published because a Publish button is visible fails.
The other generated images make similarly inspectable OCR, chart, counting,
spatial, conflict, document-layout, negative-evidence, and captioning checks.

### How strict pass and score differ

Each row records both `passed` and `score`. Strict pass means the entire case
contract passed. The score preserves useful partial signal where the adapter
supports it:

- Mere chat and tool cases report the fraction of deterministic checks passed;
  strict pass still requires all checks.
- HumanEval, EvalPlus, LiveCodeBench, and external BFCL rows are binary: the
  whole executable or tool-call contract passes or it does not.
- LongBench records its 0-to-1 text metric and separately applies the pinned
  pass threshold.
- Unsupported capabilities, such as an image case sent to a text-only catalog
  model, are unscored and reported separately rather than counted as wrong.

Always compare the strict pass rate, mean score, unscored count, and per-source
breakdown together. A model can be excellent at chat and tool use while being
materially weaker at executable code, or the reverse.

A real fused run never downloads a missing model. Every selected catalog id
must already resolve under the active `--models-root`; otherwise the command
stops before MLX initialization. The plan records the mere.run version and exact
runner-executable SHA-256, host processor/memory/architecture/OS identity,
catalog repository and revision, plus the installed `mererun_model.json` id,
schema version, source pins, and exact manifest SHA-256. This makes a receipt
identify the code, host, and checkpoint metadata that were actually visible to
the runtime without recursively scanning model storage.

## Sampling contract

The fused quality lane never uses temperature zero. It evaluates the models
under their published native sampling profiles:

| family | temperature | top-p | top-k | min-p |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8 | 1.0 | 0.95 | 20 | 0 |
| Nemotron Lightning | 1.0 | 0.95 | disabled | 0 |
| Laguna | 1.0 | 1.0 | 20 | 0.02 |

The manifest supplies repeated trials so one lucky or unlucky stochastic
sample does not become the model result. Correctness is reported per lane,
source, and capability; there is no single composite that hides a weak
category.

## Logprob contract

Quality runs default to `--logprobs summary`. Use `tokens` for chosen-token
records or `top --top-logprobs 5` for visible-token candidate lists:

```bash
mere.run model benchmark fused \
  --suite lite \
  --logprobs top \
  --top-logprobs 5 \
  --allow-code-execution \
  --json
```

Each measurement distinguishes:

- raw model logprob before temperature, top-k, top-p, and min-p;
- policy logprob after the exact sampling transforms that selected the token;
- raw and policy entropy;
- raw and policy top-1/top-2 logprob margin;
- output region, with reasoning text redacted;
- the optional visible-token top candidates.

Aggregate reports include mean/min logprob, expected calibration error,
selective accuracy, fragile passes, and confident failures. Token and top
capture also report low-confidence token-index spans. These are diagnostics,
not replacements for functional correctness.

Logprob quality capture always uses the final target model and exact top-p
math. It routes Qwen away from MTP, Nemotron away from DSpark, and Laguna away
from DFlash and continuous batching. Draft probabilities and acceptance rates
belong to acceleration diagnostics, not model calibration.

Use the separate native performance lane when both views are needed:

```bash
mere.run model benchmark fused \
  --suite lite \
  --performance-lane native \
  --allow-code-execution \
  --json
```

Performance-lane rows have no correctness score and no logprobs. This keeps
readback overhead and acceleration routing out of the quality result.

## Provenance and external fixtures

Every planned case reports:

- source and pinned/imported version;
- upstream original id;
- license and redistribution mode;
- a hash of the exact source reference;
- a content SHA-256 when the content is present;
- an image-byte SHA-256 for every vision fixture;
- capability tags, difficulty, and selection rationale.

The run plan also records a canonical SHA-256 of the complete suite manifest,
so two reports claiming the same semantic version can still be compared byte
for byte.

Mere-authored chat/tool cases and the small MIT-licensed HumanEval selection
are embedded. The 10 Mere vision cases materialize deterministic local PNG
fixtures for OCR, charts, spatial reasoning, counting, document layout,
negative evidence, multi-panel consistency, dense captioning, and action
boundaries. Their exact image bytes are hashed into the receipt before a model
can load.

The upstream HumanEval+, MBPP+, LiveCodeBench, BFCL, and LongBench datasets are
not vendored. Instead,
`BenchmarkSuites/mere-fused-sources-v1.json` locks the exact repository or
dataset revision, source URL, source-file SHA-256, license context, and selected
upstream ids. `scripts/import-fused-benchmark-fixtures.py` verifies those locks
and deterministically normalizes the selected 35 cases to local JSONL. This
keeps upstream license terms visible without treating a changing remote dataset
as part of the repository.

Generate the selected cases, stamp their canonical content hashes, then verify
the frozen result before running a model:

```bash
python3 scripts/import-fused-benchmark-fixtures.py
.build/debug/mere.run model benchmark fused-fixture \
  .tmp/fused-benchmark-fixtures/unstamped.jsonl \
  > .tmp/fused-benchmark-fixtures/selected.jsonl
.build/debug/mere.run model benchmark fused-fixture \
  .tmp/fused-benchmark-fixtures/selected.jsonl --check
```

After one online import, `--offline` reproduces the same normalized fixtures
using only the verified source cache. Generated source and fixture files remain
under `.tmp/` and are not committed.

The normalized selection contains:

- six HumanEval+ and six MBPP+ tasks evaluated against every selected base and
  Plus input;
- six LiveCodeBench `release_v5` code-generation strata spanning Codeforces
  stdin programs and LeetCode-style functional tasks, including public and
  private tests;
- ten BFCL v3 slices covering single, parallel, zero-call, and pinned
  conversation-history tool-use cases;
- seven LongBench v1 slices scored with task-appropriate QA-F1, ROUGE-L, or
  retrieval metrics and explicit Mere thresholds.

This is a pinned Mere subset, not a reproduction of any upstream leaderboard.
BFCL conversation-history slices are not the complete BFCL executable-state
runner, and the LongBench pass thresholds are Mere suite contracts.

Pass the frozen file with `--external-cases`. The typed record includes an
immutable source revision and the scoring contract needed to reproduce the
case:

```json
{
  "id": "humaneval-plus.0",
  "kind": "code",
  "sourceVersion": "v0.1.10",
  "sourceRevision": "26d6d00...",
  "originalID": "HumanEval/0",
  "messages": [{"role":"user","content":"..."}],
  "entryPoint": "has_close_elements",
  "tests": "def check(candidate): ...",
  "contentSHA256": "..."
}
```

Optional typed fields cover text metrics and reference answers, one or more
expected tool calls (including an explicit no-call expectation), function,
stdin, and functional code tests, tools, phrase constraints, and a single
absolute `imageUrl` on a vision message. A vision record also carries
`imageSHA256`. The fixture helper computes both the image-byte hash and the
content hash over the sorted canonical payload excluding `contentSHA256`; an
import is rejected before any model loads when either hash does not match. The
importer-generated upstream id, source revision, and resolved hashes become the
run receipt.

Run the frozen selection with either suite:

```bash
mere.run model benchmark fused \
  --suite lite \
  --external-cases .tmp/fused-benchmark-fixtures/selected.jsonl \
  --allow-code-execution \
  --sandbox auto \
  --json
```

Fixture stamping, verification, and `fused --dry-run` bypass machine inference
admission because they cannot load a model. A non-dry fused run still uses the
normal machine-wide admission controller.

## Checkpointed and bounded runs

Long fused runs can persist an atomic JSON checkpoint after every completed
case-trial. `--resume` accepts that file only when the full plan hash still
matches: exact runner executable and host identity, suite manifest, model order
and profiles, installed model-manifest hashes, fixture paths and hashes, trials,
cases, context and token limits, logprob settings, response capture, resolved
Python executable and SHA-256, resolved code-sandbox backend, execution timeout,
and performance lane. Duplicate or out-of-plan rows are rejected. Checkpoints
also preserve ISO-8601 creation and last-update timestamps.

Model preparation or generation failures abort the invocation without writing
the current row. Rows checkpointed before the failure remain intact, and an
exact `--resume` retries the pending row instead of recording infrastructure
failure as model quality.

Use `--case-trial-limit` to yield Metal cleanly after a small number of new
rows. It requires a checkpoint; the next invocation resumes at the first
unfinished row instead of repeating completed inference:

```bash
mere.run model benchmark fused \
  --suite comprehensive \
  --models vision-chat-q38-27b \
  --external-cases .tmp/fused-benchmark-fixtures/selected.jsonl \
  --allow-code-execution \
  --checkpoint .tmp/fused-benchmark-results/qwen38.json \
  --case-trial-limit 5 \
  --json

mere.run model benchmark fused \
  --suite comprehensive \
  --models vision-chat-q38-27b \
  --external-cases .tmp/fused-benchmark-fixtures/selected.jsonl \
  --allow-code-execution \
  --checkpoint .tmp/fused-benchmark-results/qwen38.json \
  --case-trial-limit 5 \
  --resume \
  --json
```

The checkpoint is also usable with `--dry-run`, which creates or validates the
exact empty run plan without admission, model loading, generated-code
execution, or Metal work. Run one model per checkpoint when installed models
live under different mounted roots. Reports expose completed, remaining, and
complete/partial case-trial counts.

## Code execution

Code scoring uses the existing local sandbox and requires explicit consent:

```bash
mere.run model benchmark fused \
  --suite comprehensive \
  --allow-code-execution \
  --sandbox auto \
  --json
```

`--dry-run` never executes code and does not require the flag.
