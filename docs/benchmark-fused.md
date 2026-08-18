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
can load. HumanEval+, MBPP+, LiveCodeBench, BFCL, and LongBench remain
reference-only. This avoids silently vendoring changing public datasets or
misrepresenting their licenses.

Normalize selected external cases to one JSON object per line and pass one or
more files with `--external-cases`. The typed record contains:

```json
{
  "id": "humaneval-plus.0",
  "kind": "code",
  "sourceVersion": "<pinned upstream revision>",
  "originalID": "HumanEval/0",
  "messages": [{"role":"user","content":"..."}],
  "entryPoint": "has_close_elements",
  "tests": "def check(candidate): ...",
  "contentSHA256": "..."
}
```

Optional fields cover tools, required/forbidden phrases, expected tool name,
expected arguments, and a single absolute `imageUrl` on a vision message. A
vision record also carries `imageSHA256`. The fixture helper computes both the
image-byte hash and the content hash over the sorted canonical payload excluding
`contentSHA256`; an import is rejected before any model loads when either hash
does not match. The importer-generated upstream id and resolved hashes become
the run receipt.

Stamp a normalized file to stdout, then verify the frozen result:

```bash
mere.run model benchmark fused-fixture unstamped.jsonl > selected.jsonl
mere.run model benchmark fused-fixture selected.jsonl --check
```

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
