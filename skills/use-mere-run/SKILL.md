---
name: use-mere-run
description: Operate the public mere.run CLI for local AI, model setup, media generation and analysis, workflow graphs, or API serving. Use for choosing supported models, running workflows, and troubleshooting an installed CLI. Source-code development belongs to the mere-run skill.
---

# Use mere.run

Help the user get an output from the public `mere.run` CLI. Start with their
requested result and discover the matching commands on their machine.

## Identify the executable

Use the installed binary for operator tasks:

```bash
mere.run --version
mere.run --help
```

In a source checkout, use `swift run mere.run` when the user wants to run that
checkout. Do not silently substitute an installed release for changed source.
If the binary is missing outside a checkout, use the installation instructions
on the [mere.run releases page](https://mere.run/releases). Do not assume a
Swift toolchain or source repository is available to an end user.

Examples in this skill are starting points. The selected binary's `--help`
owns accepted flags and defaults. If a discovery command is unavailable, use
that version's help and explain the version gap.

## Discover a workflow and its models

Run the discovery commands relevant to the task:

```bash
mere.run catalog --json
mere.run model capabilities --recommended --json
mere.run model list --json
mere.run guide --list
mere.run guide --list-models --json
```

`catalog` describes command capabilities. `model capabilities` describes
hardware support; `model list` describes installation status. A supported model
is not necessarily installed or suitable for the user's quality and latency
requirements. Use `model capabilities --all --json` for models outside the
recommended set and read the reported restrictions.

Load the command cookbook for workflow controls and the model handbook for
provider-specific prompting, input requirements, and validation limits:

```bash
mere.run catalog image.generate --json
mere.run image generate --help
mere.run guide image generate
mere.run guide --model image-zimage-nano
```

Use `guide --model` without a command path to select a model handbook. Command
cookbooks can cover several models; a guide example is not evidence that a
particular checkpoint was tested on this machine.

## Route tasks to the right command family

Discover subcommands with `mere.run <group> --help`; this table is a starting
map, not a complete command list.

| Desired result | Command families to inspect |
| --- | --- |
| Chat, code, embeddings, anonymization, or text training | `text` |
| Image generation, editing, training, or 3D reconstruction | `image` |
| Speech, transcription, diarization, or voice profiles | `speech` |
| Captioning, OCR, grounding, tracking, faces, pose, depth, or geometry | `vision` |
| Earth-observation inference | `geo` |
| Audio enhancement, music, or sound effects | `audio`, `music`, `sfx` |
| Video generation, animation, editing, or world sessions | `video`, `world` |
| Repeatable pipelines, remote execution, or saved results | `graph`, `executor`, `relay`, `run` |
| Evaluation packs and reproducible comparisons | `eval` |
| Model storage, adapters, configuration, or health | `model`, `adapter`, `config`, `status` |
| Local API or a companion interface | `api`, `open-webui` |
| Companion plugins or guided setup | `plugin`, `setup`, `agent` |

For graph tasks, start with `graph catalog`, `graph validate`, and `graph
preflight` help. Inspect the selected executor and authentication before
submitting work. A graph that validates has not executed successfully.

## Select a model explicitly

Check support and available storage before a large download. Pull only the
model needed for the task, then pass that same ID to the inference command.
Do not rely on a default to select the model you just downloaded. Some commands
accept only local paths; inspect their help before passing a managed ID.

The following examples generate files or start a server when run. Adapt the
model choice using the machine's capability report and the user's request.

Generate an image:

```bash
mere.run model pull image-zimage-nano
mere.run image generate --model image-zimage-nano \
    --prompt "a ceramic mug in soft morning light" --output ./mug.png
```

Chat with an explicit model:

```bash
mere.run model pull text-chat-gemma4-12b-4bit
mere.run text chat --model text-chat-gemma4-12b-4bit \
    --prompt "Explain local inference in one paragraph." --stream
```

Generate speech:

```bash
mere.run model pull speech-tts-qwen3-nano
mere.run speech synthesize "Hello from mere.run" \
    --model speech-tts-qwen3-nano --output ./hello.wav
```

Serve a selected local chat model:

```bash
mere.run model pull text-chat-gemma4
mere.run api serve --engine text-chat-gemma4 --model text-chat-gemma4
```

For API work, inspect `api serve --help` for supported engines, endpoints, and
preflight options. Model IDs and engine names are different contracts. Keep
loopback binding unless the user requests network access. Non-loopback binds
require `--api-key` or `MERERUN_API_KEY`; do not use a sample key as a credential.

For music and video, read the selected model's handbook before choosing steps,
frame counts, reference inputs, or output modes. These controls differ across
families. For speech transcription, inspect backend and execution-provider
options instead of assuming one runtime fits every audio file.

## Storage and troubleshooting

The default model store is
`~/Library/Application Support/MereRun/models`. Use `--models-root` or
`MERERUN_MODELS_DIR` to select another store. `MERERUN_HUB_CACHE` controls the
Hugging Face download cache; changing the model store does not move that cache.
Inspect `model storage --help` before planning a move or cleanup.

- Missing command or option: record the executable path and version, then read
  its help. Do not guess flags from another release.
- Missing model: inspect `model list` and `model info` before pulling the
  selected ID or supplying a command-supported local model path.
- Unsupported hardware or memory pressure: inspect `model capabilities --all`
  and choose a supported model. A forced pull does not establish runtime support.
- Download failure: check free space, model-store and cache locations, and any
  model access requirements reported by the CLI.
- Weak output: use the cookbook and handbook to adjust the prompt and relevant
  model controls. Keep the seed fixed when comparing one change.
- API failure: use `status`, server diagnostics, and the requested engine's
  compatibility information. A healthy server does not prove every endpoint
  or model works.

Use `--json` where the command supports it and keep stderr diagnostics separate
from machine-readable stdout. After generation, inspect the resulting artifact
and report what was actually verified. Use `run inspect` for durable workflow
reports; successful parsing or preflight alone is not successful inference.
