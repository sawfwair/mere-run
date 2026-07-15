# Structured Runs, Preflights, and Declarative Actions

This document defines the headless-first run contract for
`mere.run`: structured `--json` output, cheap `--preflight` checks, and
declarative next actions that can later power optional local UIs without making
the UI the source of behavior.

The immediate goal is not to build another standing app surface. The goal is to
make every substantial command able to describe what it will do, what it did,
what artifacts it produced, and what the caller can safely do next.

## Product thesis

`mere.run` is a headless local runtime first.

The optional macOS app and existing LoRA `--visualize` dashboard are secondary
surfaces. They should read the same structured truth that scripts, agents, and
terminal users read. A UI can appear when a job benefits from inspection, but it
must not become the only place where behavior exists.

The ladder is:

1. `--json`: emit machine-readable truth to stdout.
2. `--preflight`: inspect the requested work before expensive compute.
3. run plans and workflow graphs: describe work as portable, replayable data.
4. declarative actions: tell callers what safe next commands or artifact
   operations are available.
5. remote executors: let another machine run the same contract and return the
   same run report and artifacts.
6. future `--visualize`: render the same contract in a loopback UI when useful.

The durable rule is: UI reads reports and actions emitted by the CLI. The CLI
and runtime remain the behavioral source of truth.

## Existing pattern

The LoRA viewer already proves the local-control-room model:

- `image train-lora --visualize` starts a loopback dashboard for a live run.
- `image visualize-run <directory>` reopens a completed run directory.
- The viewer reads run artifacts from disk instead of duplicating runtime logic.
- The dashboard is useful, but the most important primitive is the run
  directory itself: manifest, metrics, events, samples, checkpoints, and final
  adapter.

This plan generalizes that pattern without requiring the next phase to build UI.

## Non-goals

- Do not build a ComfyUI-style node canvas as the first product surface.
- Do not make the macOS app the canonical workflow engine.
- Do not add hosted-service, billing, app-store, or private deployment surfaces.
- Do not emit JSON mixed with diagnostics on stdout.
- Do not use untyped `[String: Any]` payload construction.
- Do not make preflight download large models, run training, or consume paid GPU
  time.
- Do not require `--visualize` for the first implementation milestone.

## Design principles

### Headless first

Every feature must work from the CLI and be useful to:

- humans in a terminal
- scripts
- Codex or other coding agents
- local desktop wrappers
- optional loopback viewers
- remote GPU workers that run the public CLI

### stdout is contract, stderr is diagnostics

This repo already treats stdout as machine-readable output and stderr as
diagnostic progress. The new JSON surfaces must preserve that contract.

When `--json` is set:

- stdout contains exactly one JSON document, unless the command is explicitly a
  streaming command with a documented JSON Lines mode.
- diagnostics, progress, warnings, and logs go to stderr.
- errors should still produce useful stderr text, and where practical should
  produce a JSON error envelope before exiting nonzero.

### Typed contracts

All output contracts should be typed Swift structs with `Codable` conformance.
Avoid ad hoc dictionaries. This keeps tests, docs, and future UI clients aligned
with compiler-checked fields.

### Stable but evolvable schemas

Every emitted envelope carries:

- `schema_version`
- `mere_run_version`
- `command`
- `mode`

Additive fields are allowed. Renames and semantic changes require a schema
version bump.

### Cheap truth before expensive work

`--preflight` should answer:

- Can this run start?
- What exactly will it run?
- What will it probably cost in local resources?
- What inputs look suspicious?
- What commands are safe next steps?

It should avoid irreversible or expensive work.

### Actions are suggestions, not hidden logic

Actions describe CLI commands or artifact operations. They do not invoke private
APIs. A UI or agent can render the actions, but all action behavior maps back to
public commands, local files, or documented URLs.

### Paths are explicit

Payloads may include local absolute paths when the command is running locally,
but portable reports should also include relative paths when possible. Secrets
and API keys must be masked.

## Public CLI surface

### `--json`

Add or normalize `--json` across commands that produce structured state.

For commands that currently print a single path or human text, `--json` should
emit an envelope and keep the existing default output unchanged.

Example:

```bash
mere.run image train-lora \
  --data ./dataset \
  --output ./style.safetensors \
  --preflight \
  --json
```

### `--preflight`

Add `--preflight` to commands where mistakes can waste meaningful time,
compute, or model downloads.

Initial target:

- `image train-lora`

Next targets:

- `image generate`
- `video generate`
- `sfx video generate`
- `vision track`
- `api serve`
- `model benchmark ...`
- `model pull`

Preflight should exit:

- `0` when no hard blockers are found, even if warnings exist
- nonzero when hard blockers exist and the command cannot safely proceed

The JSON payload should still include blockers and suggested actions.

### `--dry-run` relationship

Use `--preflight` for input, configuration, model, resource, and action
analysis.

Use `--dry-run` only when a command already has a dry execution semantic, such
as preparing manifests, resolving a training plan, or proving a request shape
without producing the main artifact.

If both exist:

- `--preflight` answers "should this run?"
- `--dry-run` answers "what would the runtime do?"

For `image train-lora`, the first implementation should prefer `--preflight`
and avoid overloading `--dry-run`.

### `--actions`

Avoid adding a separate public `--actions` flag in the first phase. Include
actions in `--json` payloads by default when they are available.

If payload size becomes a problem later, add:

```bash
--include-actions
--no-actions
```

Defaulting actions into JSON is useful because agents and UIs need the same
next-step information without a second command.

### Run directory options

Commands that produce multi-artifact runs should converge on a run directory
contract.

Candidate flags:

```bash
--run-dir ./runs/my-run
--run-name my-run
```

Do not force this into the first `--preflight --json` slice. For
`image train-lora`, continue deriving the run directory from the output path
until the shared run contract exists.

## Workflow graphs and remote jobs

Workflow Graph V1 is the portable composition contract. References are typed
objects, never interpolated strings:

```json
{ "$ref": "inputs.prompt" }
```

```json
{ "$ref": "nodes.generate-frame.outputs.image" }
```

`graph materialize` and `graph export-job` produce the same generic immutable
job bundle. There is no relay-specific export payload and no executor URL,
credential, or machine path inside the bundle.

```text
job.json
graph.json
inputs.json
assets.json
assets/sha256/<digest>
```

The same bundle is accepted by the local runner, `graph worker execute`, the SSH
executor, and relay graph nodes. `mere.run` owns graph validation, node adapters,
preflight, canonical fingerprints, execution events, and run artifacts. SSH and
relay transport that contract. Provider-specific fleet implementations remain
outside this repository.

```bash
mere.run graph catalog --json
mere.run graph validate workflow.json --inputs-json inputs.json --json
mere.run graph preflight workflow.json --inputs-json inputs.json --executor relay:fleet --json
mere.run graph export-job workflow.json --inputs-json inputs.json --output ./job-bundle --json
mere.run graph run workflow.json --inputs-json inputs.json --run-dir ./runs/job --json
mere.run graph submit workflow.json --inputs-json inputs.json --executor ssh:gpu-box --run-dir ./runs/job --json
mere.run graph submit workflow.json --inputs-json inputs.json --executor relay:fleet --run-dir ./runs/job --json
```

Remote readback uses strict executor-qualified references:

```bash
mere.run run inspect ssh://gpu-box/<job-id> --json
mere.run run watch relay://fleet/<job-id> --json-stream
mere.run run fetch relay://fleet/<job-id> --into ./runs/job --json
mere.run run cancel relay://fleet/<job-id> --json
mere.run run retry relay://fleet/<job-id> --json
```

The public schemas live under `/schemas/`. See [Portable Workflows](../workflows.md)
for the complete graph, bundle, executor, worker, and run-directory contract.

## JSON envelope

All structured command payloads should use a common envelope.

```json
{
  "schema_version": 1,
  "mere_run_version": "0.16.0",
  "command": ["image", "train-lora"],
  "mode": "preflight",
  "status": "blocked",
  "created_at": "2026-07-02T12:00:00Z",
  "cwd": "/path/to/project",
  "summary": "Dataset has 38 usable images and 2 hard blockers.",
  "request": {},
  "result": {},
  "diagnostics": [],
  "actions": []
}
```

Fields:

- `schema_version`: integer contract version.
- `mere_run_version`: CLI version string.
- `command`: public command path, not process argv.
- `mode`: `preflight`, `run`, `dry_run`, `status`, or `inspection`.
- `status`: `ok`, `warning`, `blocked`, `running`, `finished`, or `failed`.
- `created_at`: ISO-8601 timestamp.
- `cwd`: current working directory used to resolve relative inputs.
- `summary`: short human-readable summary for logs and UI titles.
- `request`: typed request summary with secrets masked.
- `result`: command-specific typed payload.
- `diagnostics`: typed blockers, warnings, notes, and estimates.
- `actions`: typed next actions.

## Diagnostic model

Diagnostics should be first-class data, not strings that clients parse.

```json
{
  "id": "caption_missing",
  "severity": "blocker",
  "title": "Missing captions",
  "message": "4 image files do not have matching .txt captions.",
  "locations": [
    {
      "kind": "file",
      "path": "/path/to/project/dataset/frame-004.png"
    }
  ],
  "suggested_action_ids": ["open-dataset", "write-missing-captions"]
}
```

Severity values:

- `blocker`: command should not proceed.
- `warning`: command can proceed, but the result is likely degraded.
- `note`: useful context.
- `estimate`: time, memory, storage, or model-size estimate.

## Declarative actions

Actions describe what a caller may do next.

```json
{
  "id": "start-training",
  "label": "Start training",
  "kind": "command",
  "style": "primary",
  "enabled": false,
  "disabled_reason": "Resolve hard blockers first.",
  "command": {
    "argv": [
      "mere.run",
      "image",
      "train-lora",
      "--data",
      "./dataset",
      "--output",
      "./style.safetensors",
      "--recipe",
      "klein-fast-style"
    ],
    "cwd": "/path/to/project"
  },
  "requires": ["preflight.passed"]
}
```

Action kinds:

- `command`: run a public `mere.run` command.
- `open_file`: open a local artifact.
- `reveal_file`: reveal a local artifact in the file manager.
- `open_directory`: open a local directory.
- `open_url`: open a loopback or docs URL.
- `copy_text`: copy text such as a command preview.
- `none`: informational action for clients that only render guidance.

Action styles:

- `primary`: the recommended next step.
- `secondary`: useful but not central.
- `danger`: destructive, requires confirmation.
- `link`: opens docs, files, or URLs.

Action safety fields:

- `enabled`
- `disabled_reason`
- `requires`
- `confirmation`
- `destructive`
- `secrets_masked`

Actions should never embed unmasked API keys or private auth tokens.

## Artifact model

Artifacts describe local files produced or consumed by a command.

```json
{
  "id": "latest-sample",
  "kind": "image",
  "role": "sample",
  "path": "/path/to/project/runs/style/samples/step-0500.png",
  "relative_path": "runs/style/samples/step-0500.png",
  "content_type": "image/png",
  "size_bytes": 1842210,
  "created_at": "2026-07-02T12:10:00Z"
}
```

Artifact roles:

- `input`
- `output`
- `sample`
- `checkpoint`
- `metric`
- `event_log`
- `manifest`
- `report`
- `cache`

## Run directory contract

Multi-artifact commands should converge on this shape:

```text
run.json
events.jsonl
metrics.csv
actions.json
plan.json
artifacts/
samples/
checkpoints/
logs/
```

Required files:

- `run.json`: latest run envelope and request summary.
- `events.jsonl`: append-only event stream.
- `actions.json`: next actions valid for the current run state.
- `plan.json`: normalized executable request, when the run came from a saved
  plan.

Optional files:

- `metrics.csv`: scalar metrics such as loss, throughput, memory, latency.
- `artifacts/`: final outputs and reports.
- `samples/`: previews or intermediate generations.
- `checkpoints/`: intermediate model artifacts.
- `logs/`: diagnostic logs, when logs are too large for events.

The first image slices keep the existing LoRA layout while adding plan
materialization through `image run-plan --materialize`. That creates `plan.json`,
`actions.json`, `run.json`, a `*.events.jsonl` stream, and the standard
`samples/`, `checkpoints/`, `artifacts/`, and `logs/` directories before work
starts. Executing the materialized `plan.json` resumes that event stream so the
same directory moves from `run_planned` to `run_started`, progress when the
workflow emits it, and a final `run_finished` or `run_failed` event.

## First vertical slice: `image train-lora --preflight --json`

This is the most valuable starting point because LoRA training is expensive
enough that bad input should fail early.

### Command behavior

```bash
mere.run image train-lora \
  --data ./style-dataset \
  --output ./style.safetensors \
  --recipe klein-fast-style \
  --preflight \
  --json
```

Should:

1. resolve CLI options and named recipes
2. inspect dataset files and captions
3. validate output path and parent directory
4. resolve model id or model path without pulling large missing models
5. estimate training image buckets
6. estimate memory and disk footprint when possible
7. report known incompatibilities
8. emit diagnostics and actions
9. exit before loading the full training runtime

### Preflight checks

Dataset checks:

- dataset directory exists
- supported image extensions exist
- every training image has a matching caption
- captions are non-empty
- repeated exact captions are flagged
- obvious placeholder captions are flagged
- optional trigger token appears where expected
- preview images are excluded or explicitly accepted
- image dimensions are readable
- low-information images are flagged when cheap heuristics can catch them
- duplicate images are flagged when cheap hashing is available

Recipe checks:

- recipe name is known
- resolved model id is shown
- recipe-derived width, height, rank, LR, steps, and target presets are shown
- incompatible options are blockers
- ignored or overridden options are warnings

Model checks:

- managed model id is known
- local model availability is reported
- missing model is a blocker for a real run
- missing model suggests a `model pull` action
- model path exists if a local path is provided
- LoRA target preset is compatible with the selected model family

Output checks:

- output extension is `.safetensors`
- output parent directory exists or can be created
- existing output is warned or blocked depending on overwrite policy
- run directory derivation is shown
- checkpoint directory plan is shown when checkpoints are enabled

Resource estimates:

- training sample count
- effective resolution or bucket plan
- total training steps
- checkpoint count
- sample generation count
- approximate disk footprint
- memory class, if the runtime can cheaply infer it
- "unknown" where estimates would require expensive runtime loading

### Example payload

```json
{
  "schema_version": 1,
  "mere_run_version": "0.16.0",
  "command": ["image", "train-lora"],
  "mode": "preflight",
  "status": "warning",
  "created_at": "2026-07-02T12:00:00Z",
  "cwd": "/path/to/project",
  "summary": "38 usable images, 2 warnings, ready to train.",
  "request": {
    "data": "./style-dataset",
    "output": "./style.safetensors",
    "model": "image-klein-base-9b",
    "recipe": "klein-fast-style",
    "training_steps": 1000,
    "width": 1024,
    "height": 1024,
    "rank": 16
  },
  "result": {
    "dataset": {
      "directory": "/path/to/project/style-dataset",
      "image_count": 38,
      "caption_count": 38,
      "usable_pair_count": 38,
      "missing_caption_count": 0,
      "duplicate_caption_count": 4,
      "low_information_image_count": 1
    },
    "model": {
      "id": "image-klein-base-9b",
      "installed": true,
      "family": "klein"
    },
    "plan": {
      "recipe": "klein-fast-style",
      "training_steps": 1000,
      "checkpoint_interval": 250,
      "expected_checkpoint_count": 4,
      "max_resolution": 512,
      "low_ram": true,
      "no_compile": true
    }
  },
  "diagnostics": [
    {
      "id": "duplicate_captions",
      "severity": "warning",
      "title": "Repeated captions",
      "message": "4 captions are exact duplicates. Training can proceed, but style learning may be weaker."
    }
  ],
  "actions": [
    {
      "id": "start-training",
      "label": "Start training",
      "kind": "command",
      "style": "primary",
      "enabled": true,
      "command": {
        "argv": [
          "mere.run",
          "image",
          "train-lora",
          "--data",
          "./style-dataset",
          "--output",
          "./style.safetensors",
          "--recipe",
          "klein-fast-style"
        ],
        "cwd": "/path/to/project"
      },
      "requires": ["preflight.passed"]
    }
  ]
}
```

## Implementation phases

### Phase 0: contract and inventory

Deliverables:

- this planning document
- inventory of current commands with existing `--json` support
- inventory of commands where preflight is valuable
- list of current output conventions that must not break

Likely files:

- `Sources/MereRunCLI/MereRunCLI.swift`
- `Sources/MereRunCLI/Commands/*`
- `Sources/MereRunCLI/Support/CLIOutput.swift`
- `Sources/MereRunCLI/Support/CLIStderr.swift`
- tests under `Tests/MereRunCLITests`

Acceptance:

- no behavior changes
- document reviewed against current command tree

### Phase 1: shared typed contracts

Add shared support types for structured CLI output.

Candidate files:

- `Sources/MereRunCLI/Support/StructuredRunEnvelope.swift`
- `Sources/MereRunCLI/Support/PreflightDiagnostic.swift`
- `Sources/MereRunCLI/Support/DeclarativeAction.swift`
- `Sources/MereRunCLI/Support/RunArtifactSummary.swift`

Types:

- `StructuredRunEnvelope<Request: Encodable, Result: Encodable>`
- `StructuredRunMode`
- `StructuredRunStatus`
- `PreflightDiagnostic`
- `PreflightSeverity`
- `DeclarativeAction`
- `DeclarativeCommandAction`
- `RunArtifactSummary`
- `MaskedCommandPreview`

Support helpers:

- stable JSON encoder
- timestamp provider
- command preview with secret masking
- relative path helper
- diagnostics-to-status reducer

Acceptance:

- unit tests cover encoding shape
- no command behavior changes yet
- no `[String: Any]`

### Phase 2: LoRA preflight analyzer

Extract preflight analysis from `ImageTrainLoRA` into typed code.

Candidate files:

- `Sources/MereRunCLI/Support/ImageLoRAPreflight.swift`
- or `Sources/MereRunCLI/Support/LoRATrainingPreflight.swift`

Responsibilities:

- accept resolved training options
- inspect dataset
- inspect output plan
- inspect model availability
- emit typed result and diagnostics
- emit declarative actions

Avoid:

- full model loading
- optimizer setup
- GPU allocation
- training runtime initialization

Acceptance:

- `image train-lora --preflight --json` works for a valid tiny fixture
- blockers produce nonzero exit
- warnings keep exit zero
- stderr remains diagnostic only

### Phase 3: command integration

Update `ImageTrainLoRACommand.swift`.

Add:

```swift
@Flag(name: [.customLong("preflight")], help: "Inspect the LoRA training request without running training.")
var preflight: Bool = false

@Flag(name: [.customLong("json")], help: "Emit a structured JSON report.")
var json: Bool = false
```

Behavior matrix:

| Flags | Behavior |
| --- | --- |
| none | current behavior |
| `--json` | final success emits structured run result, if feasible in this phase |
| `--preflight` | human-readable preflight summary |
| `--preflight --json` | structured preflight report |
| `--preflight --quiet` | either reject as incompatible or print only essential summary |

For the first slice, it is acceptable to implement only
`--preflight --json` and a human-readable `--preflight` summary, then add
full-run `--json` in a later phase.

Acceptance:

- existing parsing tests updated
- existing default stdout contract unchanged
- `--quiet` behavior remains compatible

### Phase 4: tests

Add focused tests before broad rollout.

Test categories:

- argument parsing includes `--preflight` and `--json`
- valid preflight fixture emits expected JSON fields
- missing dataset is a blocker
- missing captions are blockers
- duplicate captions are warnings
- missing model suggests `model pull`
- existing output warning or blocker follows chosen overwrite policy
- stdout JSON decodes with `JSONDecoder`
- stderr contains diagnostics but stdout is clean JSON
- action argv is shell-safe and secret-masked

Fixtures:

- tiny image dataset with captions
- dataset with missing captions
- dataset with duplicate captions
- dataset with preview images

Acceptance:

- targeted Swift tests pass
- `./scripts/check.sh` passes before PR

### Phase 5: docs

Update public docs after the first command is real.

Likely files:

- `docs/cli.md`
- `docs/runtime/image.md`
- `Sources/MereRunCLI/Guides/image-train-lora.md`
- `README.md`, only if the behavior is important enough for top-level examples

Docs should show:

- preflight before real training
- JSON consumption example
- common blocker examples
- relationship to `--visualize`, clearly marked as separate

Acceptance:

- docs examples use repo-relative paths
- no workstation-specific paths
- no stale model IDs

### Phase 6: structured run directories

After the first preflight slice is stable, extend real long-running commands to
write `run.json`, `actions.json`, and normalized events. The first image steps
now materialize saved LoRA training and generation plans into this layout with
`image run-plan --materialize`.

Start with:

- `image train-lora`, because it already has run artifacts
- `image generate`, because wrappers need durable output directories before
  expensive image work starts

Then:

- `video generate`
- `sfx video generate`
- `vision track`
- `model benchmark`

Acceptance:

- existing LoRA viewer continues to work
- `image visualize-run` can read both old and new run directories
- new run files are additive

### Phase 7: cross-command adoption

Adopt the contract where it buys real value.

High-value commands:

- `model pull --preflight --json`
  - model size, source, installed state, disk estimate, cache path, actions
- `api serve --preflight --json`
  - implemented: host, port, auth requirement, selected engine, model availability, runtime limits, companion models, actions
- `image generate --preflight --json`
  - model availability, dimensions, reference images, output path, actions
- `video generate --preflight --json`
  - implemented: frame count, dimensions, model availability, image inputs, output path, actions
- `vision track --preflight --json`
  - implemented: video input path, model availability, prompt parsing, output/json/mask destinations, actions
- `sfx video generate --preflight --json`
  - implemented: raw video or feature input, VFlow/DVFlow model availability, Synchformer requirement, output path, denoise plan, actions
- `model benchmark ... --json`
  - benchmark plan, live metrics, final report, rerun action

Lower-value commands can wait.

### Phase 8: workflow graphs

Once single-command run plans are stable, introduce workflow graphs as the
composition layer.

Start with graph authoring and validation:

- `graph validate ./workflow.json --json`
- `graph preflight ./workflow.json --json`
- `graph materialize ./workflow.json --run-dir ./runs/name --json`

Then add local execution:

- execute nodes in dependency order
- map node outputs into downstream node inputs
- write one parent run directory with per-node child run directories
- preserve every child command's structured report
- emit parent `events.ndjson` with node start, progress, completion, and failure

Acceptance:

- invalid graph shape blocks before any node runs
- missing model/input/capability diagnostics include the node ID
- a graph containing one `image.generate` node produces the same command plan as
  `image generate --preflight --json`
- a two-node graph can pass an image output into a video input without a caller
  inventing temp paths
- `run inspect` can summarize the parent graph run and each child run

### Phase 9: remote graph jobs

After graph preflight and local graph execution exist, add exportable remote job
payloads.

The local CLI should not own authentication, fleet policy, hosted queues, or
remote asset storage. It should prepare a portable job description that an
external relay can submit to a GPU node.

Candidate command:

```bash
mere.run graph export-job ./workflow.json \
  --target relay \
  --asset-root ./assets \
  --json
```

The result should include:

- graph payload
- content-addressed local asset manifest
- capability requirements
- expected output descriptors
- suggested upload/submission actions
- local run metadata for later readback import

Relay integration can then live in the relay repo:

- client API accepts graph job payloads
- scheduler matches graph requirements to online node capabilities
- node downloads or receives graph assets
- node runs `mere.run graph run` or equivalent public CLI operations
- relay uploads graph outputs and final run report
- clients poll, stream, cancel, and read final artifacts through existing relay
  job status patterns

Acceptance:

- a graph job can be exported without network access
- a remote executor can run the exported graph using only public `mere.run`
  commands and declared assets
- relay results can be imported or inspected using the same run-report shape as
  local execution
- no relay secrets, tokens, account URLs, or private scheduling code land in
  this public repo

### Phase 10: optional viewers

Only after JSON and run directories are stable, add viewers as thin clients.
The headless readback layer is `mere.run run list --root <path> --json` plus
`mere.run run inspect <path> --json`. `run list` discovers durable run
directories, structured report JSON files, and saved run-plan JSON files under a
workspace root; `run inspect` drills into one selected artifact. Future viewers
should treat those reports as their first data source rather than rediscovering
files or actions themselves.

Legacy/plugin run directories remain explicitly marked as non-native. When
their `run.json` matches a known plugin manifest shape, `run inspect` emits a
warning-level `legacy_manifest` summary instead of treating the directory as an
opaque blocker. Unknown or corrupt manifests still block inspection.

Run-directory inspection also emits a compact `metrics` summary derived from
loss CSVs and discovered artifacts, so clients can show latest loss, step range,
sample count, checkpoint count, and adapter count without parsing artifact
files themselves.

`run list` entries include `created_at` and `updated_at` where those timestamps
are available from native manifests, event streams, legacy/plugin manifests,
structured reports, or saved plans. Clients should sort locally rather than
requiring another CLI mode for each view.

Viewer rule:

- viewer routes read the same report, events, metrics, artifacts, and actions
  that the CLI already writes
- viewer buttons render declarative actions
- viewer does not invent hidden command behavior

Possible future commands:

```bash
mere.run run view ./runs/my-run
mere.run image generate --visualize
mere.run api serve --visualize
```

This is intentionally not part of the first implementation milestone.

## Command adoption matrix

| Command | `--json` | `--preflight` | actions | run directory | viewer |
| --- | --- | --- | --- | --- | --- |
| `image train-lora` | phase 3 | phase 3 | phase 3 | phase 6 | existing, later upgrade |
| `image generate` | implemented | implemented | implemented | implemented for saved plans | later |
| `video generate` | implemented for preflight | implemented | implemented | phase 7 | later |
| `sfx video generate` | implemented for preflight | implemented | implemented | phase 7 | later |
| `vision track` | implemented for preflight | implemented | implemented | phase 7 | later |
| `api serve` | implemented for preflight | implemented | implemented | not needed | later |
| `model pull` | implemented | implemented | implemented | not needed | not needed |
| `run list` | implemented | not needed | implemented | discovers existing | not needed |
| `run inspect` | implemented | not needed | implemented | reads existing | not needed |
| `graph validate` | phase 8 | not needed | phase 8 | not needed | not needed |
| `graph preflight` | phase 8 | phase 8 | phase 8 | not needed | later |
| `graph run` | phase 8 | phase 8 | phase 8 | phase 8 | later |
| `graph export-job` | phase 9 | phase 9 | phase 9 | not needed | not needed |
| `model benchmark` | phase 7 | phase 7 | phase 7 | phase 7 | later |

## Error handling

Preflight blockers should produce structured diagnostics and a nonzero exit.

For `--preflight --json`, prefer emitting the JSON report even when blocked.
This lets agents fix the issue without parsing stderr.

For unexpected internal failures:

- stderr should include the localized error
- stdout may be empty if the command cannot safely construct the envelope
- tests should cover expected blocker paths, not every fatal runtime exception

## Security and privacy

The payload is local-first, but it can still expose sensitive details if copied.

Rules:

- mask API keys and auth tokens
- do not include full environment dumps
- do not include secret-bearing command arguments
- include local paths because local tools need them, but also include relative
  paths where possible
- keep non-loopback server action suggestions auth-aware
- treat `open_url` actions to non-loopback hosts carefully

## Compatibility

The implementation should be additive.

Must preserve:

- current default stdout behavior
- current `--quiet` behavior unless explicitly documented
- current `--visualize` behavior
- existing LoRA run directories
- existing docs examples until the new feature lands

The first PR should not require users to change existing commands.

## Open decisions

1. Should `--preflight` without `--json` print a compact human summary or require
   `--json` for the first implementation?
2. Should blockers exit nonzero even though the JSON payload is successfully
   emitted? Recommended: yes.
3. Should missing optional models be blockers or warnings? Recommended: blocker
   when the run cannot start without them, warning when the command can pull or
   fallback.
4. Should `actions.command.argv` include `mere.run` as argv[0] or only the
   public command path? Recommended: include `mere.run` for copy/paste actions
   and include `command_path` separately if clients need it.
5. Should the run directory contract live in `MereRunCLI` or `MereRunCore`?
   Recommended: start in `MereRunCLI/Support`; move shared artifact/event types
   lower only if runtimes need to write them directly.
6. Should graph jobs enter relay as a new first-class work kind or as a
   tool-style request carrying a `mere.run/workflow-graph` payload? Recommended:
   first-class long-term, tool-style only as a compatibility bridge.
7. Should `mere.run` submit directly to relay? Recommended: no for the public
   CLI. Export a relay-ready job payload and action first; keep authenticated
   submission in the relay/client layer unless a generic remote-executor plugin
   boundary exists.
8. Should graph nodes be limited to public commands or allowed to call plugin
   tools? Recommended: start with public `mere.run` command nodes, then allow
   plugin nodes only when their manifests declare typed inputs, outputs, and
   artifact behavior.

## PR sequencing

### PR 1: contract and LoRA preflight JSON

Scope:

- shared typed envelope, diagnostics, actions
- `image train-lora --preflight --json`
- tests and docs for the new preflight path

Avoid:

- generalized run directories
- viewer changes
- cross-command adoption

### PR 2: LoRA run actions and normalized files

Scope:

- write `run.json` and `actions.json` for LoRA training
- keep existing viewer working
- add compatibility tests for old and new run dirs

### PR 3: model and image generation preflights

Scope:

- `image generate --preflight --json`
- generation model/input/LoRA/structured-prompt/output diagnostics
- `start-generation`, `pull-model`, input/open, LoRA reveal, and output actions
- `result.run_plan` for `image.generate`
- `image run-plan` replay, preflight, and materialization for saved generation
  plans
- `model pull --preflight --json`
- pull model/source/support/install/disk diagnostics
- `pull-model`, `pull-models`, model-store, and hub-cache actions
- `api serve --preflight --json`
- serving host/port/auth/model/runtime/KV diagnostics
- redacted start-server, status, model-store, and pull actions
- shared model availability diagnostics

### PR 4: long-running media commands

Scope:

- `video generate --preflight --json`
- video model/input/output/duration/frame diagnostics
- `start-video-generation`, `pull-model`, input reveal, and output actions
- `vision track --preflight --json`
- tracking video/model/prompt/output/json/mask diagnostics
- `start-tracking`, `pull-model`, input reveal, output, and mask actions
- `sfx video generate --preflight --json`
- VFlow/DVFlow model, raw-video Synchformer, feature-input, output, and denoise diagnostics
- `start-sfx-video-generation`, `pull-model`, `pull-synchformer`, input reveal, and output actions
- benchmark commands where useful

### PR 5: run inspection

Scope:

- generic `run list --json` discovery under workspace roots
- generic `run inspect --json` readback for run directories, report files, and
  plan files

### PR 6: workflow graph contract

Scope:

- graph schema and typed Swift decoding
- `graph validate`
- `graph preflight`
- graph actions and diagnostics
- one-node graph fixtures for `image.generate`
- two-node fixture that passes image output into video input

Avoid:

- remote submission
- visual graph editor
- private relay assumptions

### PR 7: local graph run and relay job export

Scope:

- `graph run`
- graph parent run directory with child node reports
- `graph export-job --target relay`
- asset manifest and capability requirement payload
- docs describing how an external relay should consume the payload

Avoid:

- embedding relay auth or hosted scheduling in this repo
- requiring a GPU node for local tests

### PR 8: optional viewer refresh

Scope:

- render actions in the LoRA viewer
- optionally introduce a generic `run view`
- keep viewer loopback-only

## Acceptance criteria for the first milestone

The first milestone is complete when:

- `mere.run image train-lora --preflight --json` emits a typed, decodeable JSON
  envelope to stdout
- blockers and warnings are structured diagnostics
- hard blockers exit nonzero
- no expensive model load or training starts during preflight
- valid preflight includes a `start-training` action
- missing model includes a `model pull` action
- missing captions include precise file locations
- default `image train-lora` behavior is unchanged
- targeted CLI tests pass
- `./scripts/check.sh` passes

## Why this matters

This turns `mere.run` into a better headless runtime without expanding the
standing UI surface.

The runtime becomes easier to drive from scripts, Codex, remote workers, and
small local dashboards. Bad runs fail earlier. Good runs explain themselves.
Optional UIs can appear later as faithful renderers of structured truth instead
of becoming a second workflow engine.
