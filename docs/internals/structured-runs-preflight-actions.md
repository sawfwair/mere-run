# Structured runs, preflights, and declarative actions

This page defines the headless-first run contract for `mere.run`: structured
`--json` output, low-cost `--preflight` checks, and declarative next actions
that optional local user interfaces (UIs) can render without becoming the
source of behavior.

The contract lets a substantial command describe what it will do, what it did,
which artifacts it produced, and what the caller can safely do next. The
contract sections describe behavior as of v0.67.1 and are normative. The
[design record](#design-record) at the end summarizes how the contract was
rolled out.

## Principles

`mere.run` is a headless local runtime first. The optional macOS app and the
LoRA `--visualize` dashboard are secondary surfaces. They read the same
structured output that scripts, agents, and terminal users read. A UI can
appear when a job benefits from inspection, but it's never the only place where
behavior exists: UIs read reports and actions that the CLI emits, and the CLI
and runtime stay the source of truth.

The contract builds up in layers:

1. `--json` emits machine-readable output to stdout.
2. `--preflight` inspects the requested work before expensive compute.
3. Run plans and workflow graphs describe work as portable, replayable data.
4. Declarative actions tell callers which next commands or artifact operations
   are safe.
5. Remote executors run the same contract on another machine and return the
   same run report and artifacts.
6. Viewers, such as `image visualize-run`, render the same contract in a
   loopback UI.

### Headless first

Every feature works from the CLI and serves humans in a terminal, scripts,
coding agents, local desktop wrappers, optional loopback viewers, and remote
GPU workers that run the public CLI.

### Stdout is the contract, stderr is diagnostics

When `--json` is set:

- Stdout contains exactly one JSON document, unless the command is a streaming
  command with a documented JSON Lines mode, such as `--json-stream`.
- Diagnostics, progress, warnings, and logs go to stderr.
- Errors still produce useful stderr text and, where practical, a JSON error
  envelope before the command exits nonzero.

### Typed contracts

Every output contract is a typed Swift structure with `Codable` conformance. Ad
hoc dictionaries and `[String: Any]` payloads aren't allowed, which keeps
tests, documentation, and UI clients aligned with compiler-checked fields.

### Stable but evolvable schemas

Every envelope carries `schema_version`, `mere_run_version`, `command`, and
`mode`. Additive fields are allowed. Renames and semantic changes require a
schema version bump.

### Cheap answers before expensive work

`--preflight` answers these questions without irreversible or expensive work:

- Can this run start?
- What exactly will it run?
- What will it probably cost in local resources?
- Which inputs look suspicious?
- Which commands are safe next steps?

Preflight never downloads large models, runs training, or consumes paid GPU
time.

### Actions are suggestions, not hidden logic

Actions describe CLI commands or artifact operations. They don't invoke
private APIs. A UI or agent can render them, but every action maps back to a
public command, a local file, or a documented URL.

### Paths are explicit

Payloads can include absolute local paths when the command runs locally.
Portable reports also include relative paths where possible. Secrets and API
keys are always masked.

### Out of scope

- A ComfyUI-style node canvas as the primary product surface
- The macOS app as the canonical workflow engine
- Hosted-service, billing, app-store, or private deployment surfaces
- JSON mixed with diagnostics on stdout

## Public CLI surface

### `--json` and `--preflight`

`--preflight` inspects a request without running it, and `--json` turns the
report into a structured envelope. As of v0.67.1, these commands accept both
flags:

| Area | Commands |
| --- | --- |
| Image | `image train-lora`, `image generate`, `image run-plan` |
| Video | `video generate`, `video animate`, `video prepare-masks` |
| Vision | `vision track`, `vision segment`, `vision ground`, `vision serve` |
| Audio | `sfx video generate` |
| Text | `text chat` |
| Geospatial | `geo flood`, `geo fire`, `geo tessera`, `geo olmoearth` |
| Serving and models | `api serve`, `model pull` |

`text classify`, `text decide`, and `text extract` accept `--preflight`
without `--json`. `graph validate`, `graph preflight`, `run list`, `run
inspect`, and the `model benchmark` subcommands accept `--json`. Run
`mere.run <command> --help` to confirm a specific command.

```bash
mere.run image train-lora \
  --data ./style-dataset \
  --output ./style.safetensors \
  --recipe klein-fast-style \
  --preflight \
  --json
```

Preflight exits with status `0` when it finds no blockers, even if it reports
warnings. It exits nonzero when a blocker means the command can't safely
proceed. With `--json`, it still emits the full report, including blockers and
suggested actions, so an agent can fix the issue without parsing stderr.

### `--dry-run`

Use `--preflight` for input, configuration, model, resource, and action
analysis. Use `--dry-run` only where a command already has a dry execution
meaning, such as preparing a training manifest or proving a request shape
without producing the main artifact:

- `--preflight` answers "should this run?"
- `--dry-run` answers "what would the runtime do?"

### Actions

There's no separate `--actions` flag. Structured payloads include actions by
default when they're available, so agents and UIs get next-step information
without a second command.

### Run directories

Commands that produce multi-artifact runs converge on a run directory. Graph
commands take `--run-dir` explicitly, and `image generate` accepts `--run-dir`.
`image train-lora` derives its run directory from the output path.

## Workflow graphs and remote jobs

Workflow Graph V1 is the portable composition contract. References are typed
objects, never interpolated strings:

```json
{ "$ref": "inputs.prompt" }
```

```json
{ "$ref": "nodes.generate-frame.outputs.image" }
```

`graph materialize` and `graph export-job` produce the same generic,
immutable job bundle. There's no relay-specific export payload, and the bundle
contains no executor URL, credential, or machine path:

```text
job.json
graph.json
inputs.json
assets.json
assets/sha256/<digest>
```

The local runner, `graph worker`, the SSH executor, and relay graph nodes all
accept the same bundle. `mere.run` owns graph validation, node adapters,
preflight, canonical fingerprints, execution events, and run artifacts. SSH
and relay only transport that contract; provider-specific fleet
implementations live outside this repository.

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

The public JSON schemas live under `docs/public/schemas/` and are served at
`/schemas/`. For the complete graph, bundle, executor, worker, and
run-directory contract, see [Portable workflows](../workflows.md).

## JSON envelope

Structured command payloads share one envelope, `StructuredRunEnvelope` in
`Sources/MereRunCLI/Support/StructuredRunOutput.swift`:

```json
{
  "schema_version": 1,
  "mere_run_version": "0.67.1",
  "command": ["image", "train-lora"],
  "mode": "preflight",
  "status": "blocked",
  "created_at": "2026-10-29T12:00:00Z",
  "cwd": "/path/to/project",
  "summary": "Dataset has 38 usable images and 2 hard blockers.",
  "request": {},
  "result": {},
  "diagnostics": [],
  "actions": []
}
```

The fields are:

- `schema_version`: integer contract version.
- `mere_run_version`: CLI version string.
- `command`: public command path, not process argv.
- `mode`: `preflight`, `run`, `dry_run`, `status`, `inspection`, or
  `materialize`.
- `status`: `ok`, `warning`, `blocked`, `planned`, `queued`, `assigned`,
  `running`, `finished`, `failed`, or `cancelled`.
- `created_at`: ISO 8601 timestamp.
- `cwd`: current working directory used to resolve relative inputs.
- `summary`: short human-readable summary for logs and UI titles.
- `request`: typed request summary with secrets masked.
- `result`: command-specific typed payload.
- `diagnostics`: typed blockers, warnings, notes, and estimates.
- `actions`: typed next actions.

A preflight's status follows its diagnostics: any blocker makes it `blocked`,
otherwise any warning makes it `warning`, otherwise it's `ok`.

## Diagnostics

Diagnostics are data, not strings that clients parse. `PreflightDiagnostic`
lives in `MereRunRelayKit` so that graph clients share it:

```json
{
  "id": "missing_captions",
  "severity": "blocker",
  "title": "Missing captions",
  "message": "4 image file(s) do not have matching .txt captions.",
  "locations": [
    {
      "kind": "file",
      "path": "/path/to/project/dataset/frame-004.png"
    }
  ],
  "suggested_action_ids": ["open-dataset"]
}
```

A location has a `kind` and an optional `path` and `line`. Severity values are:

- `blocker`: the command can't proceed.
- `warning`: the command can proceed, but the result is likely degraded.
- `note`: useful context.
- `estimate`: a time, memory, storage, or model-size estimate.

## Declarative actions

An action describes what a caller can do next. `DeclarativeAction` lives in
`StructuredRunOutput.swift`:

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
      "mere.run", "image", "train-lora",
      "--data", "./style-dataset",
      "--output", "./style.safetensors",
      "--recipe", "klein-fast-style"
    ],
    "command_path": ["image", "train-lora"],
    "cwd": "/path/to/project"
  },
  "requires": ["preflight.passed"],
  "destructive": false,
  "secrets_masked": true
}
```

`command.argv` starts with `mere.run` so it can be copied and run;
`command_path` carries the public command path separately. Other action
fields are `path`, `url`, `text`, `candidates`, and `confirmation`. A `select`
action offers `candidates`, each with an ID, label, value, and optional path,
status, description, command, and `replace` patches.

Action kinds:

- `command`: run a public `mere.run` command.
- `open_file`: open a local artifact.
- `reveal_file`: reveal a local artifact in the file manager.
- `open_directory`: open a local directory.
- `open_url`: open a loopback or docs URL.
- `copy_text`: copy text such as a command preview.
- `select`: choose one of several candidates.
- `none`: informational guidance only.

Action styles:

- `primary`: the recommended next step.
- `secondary`: useful but not central.
- `danger`: destructive; requires confirmation.
- `link`: opens docs, files, or URLs.

Actions never embed unmasked API keys or private auth tokens.

## Run directory contract

Multi-artifact commands converge on this shape:

```text
run.json
events.jsonl
actions.json
plan.json
artifacts/
samples/
checkpoints/
logs/
```

- `run.json`: latest run envelope and request summary.
- `events.jsonl`: append-only event stream. Image run plans use a
  `*.events.jsonl` name next to the output.
- `actions.json`: next actions valid for the current run state.
- `plan.json`: normalized executable request, when the run came from a saved
  plan.
- `artifacts/`, `samples/`, `checkpoints/`, and `logs/`: final outputs,
  previews, intermediate model artifacts, and logs too large for events.

LoRA training runs keep their established layout, including `*.loss.csv` loss
history. `image run-plan --materialize` creates `plan.json`, `actions.json`,
`run.json`, the event stream, and the standard directories before work starts.
Running the materialized `plan.json` resumes that event stream, so the same
directory moves from `run_planned` to `run_started`, through any progress
events, and ends with `run_finished` or `run_failed`.

## Run inspection

`run list --root <path> --json` discovers durable run directories, structured
report files, and saved run-plan files under a workspace root.
`run inspect <path> --json` drills into one of them and reports its kind:
`run_directory`, `report_file`, `plan_file`, `unsupported_file`, or `missing`.

Run-directory inspection reports each artifact with its `kind`, `name`,
`path`, `exists`, `size_bytes`, and `is_image` fields. It also emits a compact
`metrics` summary derived from loss CSVs and discovered artifacts: the latest
loss, step range, and sample, checkpoint, and adapter counts. Clients don't
need to parse artifact files themselves.

`run list` entries include `created_at` and `updated_at` when native
manifests, event streams, legacy manifests, structured reports, or saved plans
provide them. Clients sort locally instead of asking for another CLI mode.

Legacy and plugin run directories stay marked as non-native. When their
`run.json` matches a known plugin manifest shape, `run inspect` emits a
warning-level `legacy_manifest` summary instead of blocking. Unknown or corrupt
manifests still block inspection.

### Viewers

Viewers are thin clients:

- Viewer routes read the same report, events, metrics, artifacts, and actions
  that the CLI already writes.
- Viewer buttons render declarative actions.
- A viewer never invents hidden command behavior.

There's no generic run-view command. `mere.run image visualize-run` is the
supported viewer for compatible local image-training runs. Other clients build
views from `mere.run run inspect --json`.

## Image LoRA training preflight

LoRA training is expensive enough that bad input should fail early, so
`image train-lora --preflight` was the first command on this contract. The
analysis lives in `Sources/MereRunCLI/Support/LoRATrainingPreflight.swift`
and `LoRATrainingDatasetInspection.swift`. Without loading the training
runtime, it:

1. Resolves CLI options and named recipes.
2. Inspects dataset files and captions.
3. Validates the output path and parent directory.
4. Resolves the model ID or model path without pulling a missing model.
5. Reports the training plan and known incompatibilities.
6. Emits diagnostics and actions.

Dataset diagnostics include a missing or unreadable dataset, no training
images, `missing_captions`, `empty_captions`, `placeholder_captions`,
`duplicate_captions`, and `preview_images_included`. Model and output
diagnostics include `model_missing`, `model_unknown`,
`model_family_unsupported`, `output_extension_invalid`, `output_exists`, and
`recipe_written_for_other_trainer`. A valid preflight offers a
`start-training` action, and a missing model offers `pull-model`.

The following abbreviated payload shows the shape of a passing report with a
warning:

```json
{
  "schema_version": 1,
  "mere_run_version": "0.67.1",
  "command": ["image", "train-lora"],
  "mode": "preflight",
  "status": "warning",
  "summary": "38 usable images, 1 warning, ready to train.",
  "request": {
    "data": "./style-dataset",
    "output": "./style.safetensors",
    "model": "image-klein-base-9b",
    "recipe": "klein-fast-style"
  },
  "result": {},
  "diagnostics": [
    {
      "id": "duplicate_captions",
      "severity": "warning",
      "title": "Repeated captions",
      "message": "4 captions are exact duplicates."
    }
  ],
  "actions": [
    {
      "id": "start-training",
      "label": "Start training",
      "kind": "command",
      "style": "primary",
      "enabled": true
    }
  ]
}
```

## Error handling

For `--preflight --json`, the report is emitted even when blocked, and the
command exits nonzero. For unexpected internal failures, stderr includes the
localized error and stdout can be empty if the command can't safely build the
envelope. Tests cover expected blocker paths, not every fatal runtime
exception.

## Security and privacy

Reports are local-first, but they can still expose sensitive details if they're
copied elsewhere:

- Mask API keys and auth tokens.
- Don't include full environment dumps or secret-bearing command arguments.
- Include local paths because local tools need them, and add relative paths
  where possible.
- Keep server action suggestions for non-loopback hosts auth-aware.
- Treat `open_url` actions for non-loopback hosts with care.

## Compatibility

Every structured surface is additive. It preserves the default stdout behavior,
`--quiet` behavior, `--visualize` behavior, existing LoRA run directories, and
documented examples. No existing command requires new flags.

## Design record

The contract was built in phases, starting with
`image train-lora --preflight --json`. This summary replaces the original
phase-by-phase plan.

| Phase | Outcome |
| --- | --- |
| Shared typed contracts | `StructuredRunEnvelope`, `DeclarativeAction`, and `StructuredRunOutput` in `MereRunCLI/Support`; `PreflightDiagnostic` and `StructuredRunStatus` in `MereRunRelayKit` |
| LoRA preflight | `image train-lora --preflight` with `--json`, dataset inspection, and `start-training` and `pull-model` actions |
| Generation, model, and serving preflights | `image generate`, `model pull`, and `api serve` preflights; saved image plans through `image run-plan` |
| Long-running media preflights | `video generate`, `vision track`, `sfx video generate`, and later the other vision, video, text, and geospatial commands |
| Run directories | `plan.json`, `actions.json`, `run.json`, and event streams for image training and generation plans |
| Run inspection | `run list` and `run inspect` for run directories, reports, and plans |
| Workflow graphs | `graph validate`, `graph preflight`, `graph materialize`, and `graph run` with parent and per-node run records |
| Remote graph jobs | One generic job bundle from `graph export-job`, SSH and relay executors through `graph submit`, and `run watch`, `fetch`, `cancel`, and `retry` |
| Viewers | `image visualize-run` reads the same run artifacts; no generic `run view` command |

Several open questions from the original plan were settled this way:

- `--preflight` without `--json` prints a human-readable summary.
- Blockers exit nonzero even when the JSON report is emitted.
- `actions.command.argv` includes `mere.run`, and `command_path` is separate.
- The envelope and actions live in `MereRunCLI/Support`; only diagnostics and
  statuses moved down to `MereRunRelayKit`, where graph clients need them.
- `mere.run` submits graph jobs to SSH and relay executors through one generic
  bundle; relay authentication and fleet scheduling stay outside this
  repository.
