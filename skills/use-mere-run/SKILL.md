---
name: use-mere-run
description: Operate the public mere.run CLI from discovery and model setup through preflight, execution, artifact verification, and recovery. Use for local AI, media, training, workflow graphs, plugins, or API and relay services. Source-code development belongs to the mere-run skill.
---

# Use mere.run

Help the user produce and verify the requested result with the public
`mere.run` CLI. Use the executable's contracts and bundled model handbooks to
resolve details that vary by version, model, hardware, or execution host.

## Establish the execution context

Locate the executable with `command -v mere.run`, then inspect it:

```bash
mere.run --version
mere.run --help
```

For a requested source checkout, use `swift run mere.run` from that checkout.
Do not substitute an installed release for changed source. Outside a checkout,
use the [mere.run installation instructions](https://mere.run/releases) when the
binary is missing; do not assume an end user has Swift or repository sources.

Keep the same executable, working directory, model store, cache, and executor
through discovery, preflight, and execution. A global `--models-root` goes
before the subcommand. A generated action may omit that override: preserve it
when adapting the action. Record the version and explicit model in the result.

## Load the operational reference for the task

Read the relevant reference before its first operation. These files ship with
this skill and work without a source checkout.

| Task | Reference |
| --- | --- |
| Every expensive run or model download | [Preflight and actions](references/preflight-and-actions.md): preparation checks, report schemas, blockers, and follow-up actions |
| Run, monitor, verify, cancel, or recover | [Execution and results](references/execution-and-results.md): stdout, receipts, progress, tool loops, and failure handling |
| Install models, manage storage, use adapters or plugins | [Models, storage, and plugins](references/models-storage-and-plugins.md): support, terms, caches, bindings, and installation modes |
| Build graphs, use remote workers, or serve an API | [Workflows and services](references/workflows-and-services.md): graph schemas, job identity, remote lifecycle, and API compatibility |
| Choose modality controls, train, or evaluate | [Modalities and training](references/modalities-and-training.md): input constraints, model guidance, and quality checks |

## Discover the supported workflow

Run only the discovery commands relevant to the user's task:

```bash
mere.run catalog --json
mere.run model capabilities --recommended --json
mere.run model list --json
mere.run guide --list
mere.run guide --list-models --json
```

Use `--help` for accepted arguments and the full command tree. Use `catalog`
for structured capability metadata; it is not a complete replacement for help.
Use `model capabilities --all --json` to inspect support and restrictions beyond
the recommended set. Support, installation, runtime readiness, and suitability
for the user's goal are separate facts.

Load both the command cookbook and the selected model handbook:

```bash
mere.run catalog image.generate --json
mere.run image generate --help
mere.run guide image generate
mere.run guide --model image-zimage-nano
```

`guide --model` without a command path selects the model handbook. Its provider
advice and examples do not certify inference on this machine. If a discovery
command is missing in an older release, use that release's help and explain the
gap. Do not invent flags or silently upgrade the user's installation.

## Follow the operating loop

1. Identify the user's inputs, output requirements, execution host, and quality
   constraints. Inspect input files and choose a fresh output location.
2. Select a supported model from live discovery. Read its handbook before
   choosing model-specific flags, prompts, reference inputs, or sampling values.
3. Preflight the needed model pull. Inspect download size, storage locations,
   runtime readiness, companion models, and any terms that need acceptance.
   Pull only the selected dependencies within the user's authorized task.
4. Preflight the exact run where supported. Read stdout, stderr, and process
   exit status. Inspect JSON `status` and diagnostics even when exit is zero.
   Resolve blockers and repeat the check with the same request and context.
5. Run the checked request. Remove preflight-only flags; add receipts and
   structured progress only where help advertises them. Capture both streams
   separately and retain the process or remote job identity.
6. Wait for completion, verify the declared artifacts or response, and inspect
   quality against the request. Report unresolved warnings or validation gaps.
   Recover from saved state before retrying work that may already have run.

These are task decisions, not a requirement to execute every command on every
turn. Continue already-authorized work without repeated confirmations. An
unresolved blocker or missing permission requires a concrete next step; it is
not a reason to guess a bypass flag.

## Route to a command family

Use `mere.run <group> --help` for subcommands. This is a starting map, not a
frozen list of supported models or every leaf command.

| Desired result | Command families |
| --- | --- |
| Chat, code, embeddings, anonymization, or text training | `text` |
| Image generation, editing, training, or 3D reconstruction | `image` |
| Speech, transcription, diarization, or voice profiles | `speech` |
| Captioning, OCR, grounding, tracking, faces, pose, depth, or geometry | `vision` |
| Earth-observation inference | `geo` |
| Audio enhancement, music, or sound effects | `audio`, `music`, `sfx` |
| Video generation, animation, editing, or world sessions | `video`, `world` |
| Pipelines, remote execution, or saved results | `graph`, `executor`, `relay`, `run` |
| Evaluation packs and reproducible comparisons | `eval` |
| Models, adapters, configuration, or health | `model`, `adapter`, `config`, `status` |
| Local API or a companion interface | `api`, `open-webui` |
| Companion plugins or guided setup | `plugin`, `setup`, `agent` |

Use `gate --help` when the user requests installed-model qualification. An
operating task does not require running the repository's development gate.

## Completion standard

A successful pull is not a successful model load. A passing preflight is not
inference. A queued job is not finished. A receipt identifies outputs; inspecting
those outputs establishes whether they satisfy the user's request.

Finish with the output paths or job reference, what you verified, and any
remaining limitation. Preserve useful diagnostics and intermediate work when a
run fails. Avoid claims of model quality or runtime readiness from help, catalog
metadata, or a generated filename alone.
