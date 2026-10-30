# External evaluation packs

External evaluation packs let a separate repository use `mere.run`'s local model
runtime, adapters, logprobs, scoring lifecycle, gates, and promotion receipts
without moving that repository's content into `mere.run`.

Packs draw an ownership boundary, not a packaging convention:

- `mere.run` owns the generic schema, validation, execution, checkpointing,
  calibration, and receipt formats.
- The external repository owns its questions, expected behavior, prompts,
  policies, private datasets, personas, scorers, and release decisions.
- Packs load from a path and are not copied into this repository or the public
  model catalog.
- The public source tree contains one synthetic fixture solely to test the
  contract.

Evaluation packs support text chat and image-conditioned vision-language model
(VLM) chat through the shared managed runtime. Image generation, audio, video,
grounding, segmentation, optical character recognition (OCR), and training use
separate commands.

## Lifecycle

Validate and content-address a pack without loading a model:

```bash
mere.run eval pack validate /path/to/pack --json
```

Review the exact plan without loading models or executing a scorer. In these
examples, replace `text-chat-model` with an installed managed model ID. A
`--model` value without a slot name binds the `base` slot, and an `--adapter`
value without a slot name binds `candidate`:

```bash
mere.run eval run /path/to/pack \
  --model base=text-chat-model \
  --adapter candidate=/path/to/adapter.safetensors \
  --dry-run \
  --json
```

Run the evaluation and checkpoint every completed case and trial row. To resume
an interrupted run, repeat the command with `--resume`:

```bash
mere.run eval run /path/to/pack \
  --model base=text-chat-model \
  --adapter candidate=/path/to/adapter.safetensors \
  --checkpoint ./results/checkpoint.json \
  --output ./results/report.json

mere.run eval run /path/to/pack \
  --model base=text-chat-model \
  --adapter candidate=/path/to/adapter.safetensors \
  --checkpoint ./results/checkpoint.json \
  --resume \
  --output ./results/report.json
```

Issue a content-addressed promotion receipt only after all expected rows are
present and every required gate passes:

```bash
mere.run eval promote ./results/report.json \
  --output ./results/promotion-receipt.json
```

The runner doesn't download a missing model. It accepts a resume only when the
pack, model, adapter, prompt, sampling, logprob, and runner identities match
the checkpoint plan exactly. To stop cleanly after a fixed number of new rows,
add `--case-trial-limit` along with `--checkpoint`.

`eval run` can override some pack defaults for one run: `--trials`,
`--max-tokens`, `--context-size`, `--logprobs`, and `--top-logprobs`. Use
`--json` to write the report to stdout as well as, or instead of, `--output`.

When an evaluated adapter has the native adjacent training manifest written by
`mere.run text train-lora`, the plan pins that manifest and carries forward its
format, base model, dataset fingerprint, seed, and completion status. It does
not copy training examples, source labels, adapter names, or local paths. A pack
can require the manifest, completed status, and an exact base-model match with
`adapter_requirements`; those checks connect training to qualification without
moving the owning repository's corpus.

## Pack layout

A pack is a directory with an `eval-pack.json` manifest. Case files are JSONL;
prompt sets and an optional scorer executable are separate declared files.

```text
example-pack/
├── eval-pack.json
├── cases.jsonl
├── images/
│   └── scene.png          # optional; declare the file in image_files
├── prompts/
│   └── concise.txt
└── score                 # optional executable
```

The runner reads only files that the manifest names. Paths must stay inside
the pack, and declared files can't be symlinks. Validation records the SHA-256 digest
and byte count of every declared file, including each image. Validation also
records a digest for every canonical case and the aggregate pack. Every
declared image must be referenced by a case. Every case image must appear in
`image_files`.

Pack validation rejects unsupported fields at every manifest and case contract
level. A misspelled or unavailable field can't validate and then disappear
before inference.

## Cases

Each JSONL row has an ID, a split, capability tags, chat messages, optional
built-in assertions, and optional string metadata. For example:

```json
{"id":"synthetic/color/001","split":"held-out","capability_tags":["instruction-following","conciseness"],"messages":[{"role":"user","content":"Name the color made by mixing blue and yellow. Reply with one lowercase word."}],"assertions":[{"id":"contains-green","kind":"contains","value":"green","case_insensitive":true},{"id":"no-markdown","kind":"excludes","value":"```"}],"metadata":{"source":"synthetic"}}
```

An acceptable response is `green`. The example is synthetic; a pack owner
defines the real cases and keeps them in its own repository.

To attach an image, add the local file to `image_files` in the manifest:

```json
{
  "case_files": ["cases.jsonl"],
  "image_files": ["images/scene.png"]
}
```

Then, add `image_file` to the user message that describes the image task:

```json
{
  "id": "synthetic-vision/001",
  "split": "held-out",
  "capability_tags": ["visual-grounding", "negative-evidence"],
  "messages": [
    {
      "role": "user",
      "content": "Is a triangle visible? Answer yes or no.",
      "image_file": "images/scene.png"
    }
  ],
  "assertions": [
    {
      "id": "answers-no",
      "kind": "contains",
      "value": "no",
      "case_insensitive": true
    }
  ]
}
```

The `image_file` value must be a normalized relative path. The runner resolves
the path inside the validated pack and sends the image through the multimodal
chat contract. If the selected model doesn't support image input, the run fails
before generation. Evaluation packs don't fetch images from the network or
read undeclared files.

Available splits are `training`, `validation`, `development`, `held-out`,
`regression`, `stability`, and `sealed-frontier`. Evaluation reports record the
split, capability tags, case hash, and score, but omit the original case text.
They also omit raw model responses unless you pass `--log-responses`.

Built-in assertion kinds are:

- `contains` and `excludes`.
- `regex` and `not-regex`.
- `valid-json-object`.

Use assertions for mechanical checks. Use an external scorer for richer domain
judgment.

## Matched arms and adapter comparisons

The manifest declares named model and adapter slots rather than machine-local
paths. An arm selects a model slot, optional adapter slot and scale, optional
prompt set, and one or more sampling profiles. A four-arm design can isolate
the effects of an adapter and a prompt:

| Arm | Adapter | Prompt set |
| --- | --- | --- |
| `base-neutral` | none | none |
| `base-prompted` | none | `concise` |
| `adapter-neutral` | `candidate` | none |
| `adapter-prompted` | `candidate` | `concise` |

The resulting report pins the model identity and the adapter content hash. It
does not serialize a private pack path or adapter path into the plan.

## Sampling and logprobs

For quality lanes, use a pinned, non-greedy profile and repeated trials. A
temperature-zero lane suits compliance checks, replay, and narrow runtime
comparisons, but it doesn't measure a model's normal quality distribution.

Each profile pins `temperature`, `top_p`, `top_k`, `min_p`, optional reasoning
effort, and thinking visibility. The defaults select the trial count, token and
context budgets, and logprob mode. Logprob modes are `none`, `summary`,
`tokens`, and `top`; `top` also uses `top_logprobs`.

Reports derive calibration diagnostics for each arm and profile, including expected
calibration error, selective accuracy, fragile passes, and confident failures.
These diagnostics add to correctness gates; they don't replace them.

For a JSON object response, set `defaults.response_format` to `json_object` and
set `defaults.logprobs` to `none`. The run plan pins the response format and
uses the shared structured-output capability checks. Omit `response_format`,
or set it to `text`, for unconstrained text.

## Scorers and gates

The built-in assertion scorer doesn't launch another process. An
`external-process` scorer must be a declared, executable file inside the pack,
and it's pinned with the rest of the pack. It doesn't run during validation or a
dry run. A real evaluation requires the explicit `--allow-external-scorer`
authorization flag.

The runner launches the scorer directly, not through a shell. The scorer's
environment is reduced to `HOME`, locale, `PATH`, and `TMPDIR`, so it doesn't
inherit tokens or service-specific environment variables. Temporary response
files are private to the process user, and the runner forcibly reaps a
timed-out scorer.

For each completed row, the scorer receives one versioned JSON request on
standard input. It includes pack/case hashes, arm and profile IDs, trial, model
ID, adapter hash, response, and response hash. It must return one versioned JSON
object containing:

```json
{"schema_version":1,"passed":true,"score":0.92,"metrics":[{"id":"synthetic-quality","value":0.92}],"hard_failures":[]}
```

The runner enforces the pack's scorer timeout and caps output. A scorer can
emit named metrics without exposing its rubric to `mere.run`.

Gates aggregate filtered rows using `pass-rate`, `mean-score`,
`hard-failure-count`, or `metric-mean` with `>=`, `<=`, or `==`. Filters can
select splits, arms, profiles, and capability tags. Required gates determine
promotion eligibility; optional gates remain visible diagnostics.

## Public-repository boundary

`scripts/check-evaluation-boundary.sh` rejects tracked `eval-pack.json`
manifests outside the synthetic contract fixture and rejects pack directories
under public source targets. An organization can add a private CI check without
publishing the terms it protects:

```bash
MERERUN_PROPRIETARY_MARKERS_FILE=/secure/path/markers.txt \
  ./scripts/check-evaluation-boundary.sh
```

The marker file is newline-delimited; blank lines and lines beginning with `#`
are ignored. The file must stay outside this repository. The check is defense
in depth; pack owners still need their own access controls and review
policies.
