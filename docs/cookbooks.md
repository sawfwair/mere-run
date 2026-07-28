# Cookbooks

Every command carries its own manual. `mere.run guide` is a cookbook reader
built into the binary, and it works with the network off — on a plane, in a
locked-down lab, or on a machine that has never had an API key. It prints
Markdown by default, emits JSON for agents and tools, and renders the topic
list as a Markdown table with `--markdown`.

```bash
mere.run guide --list
mere.run guide --list --markdown > guides.md
mere.run guide image generate
mere.run guide music analyze --model music-acestep-xl-turbo-lm4b
mere.run guide music generate --model music-acestep
mere.run guide music generate --model music-magenta-rt2-small
mere.run guide music transcribe --model music-muscriptor-medium
mere.run guide sfx generate --model sfx-woosh-dflow
mere.run guide open-webui
mere.run guide video generate --json
```

Each guide is written to be used, not skimmed: what the command does, which
model to install, which flags actually change the output, prompting patterns,
worked examples, iteration tips, troubleshooting, and links into the source.

## Available Topics

Creative and runtime workflows:

- `image generate`
- `image train-lora`
- `image validate`
- `text chat`
- `text code`
- `text embed`
- `text anonymize`
- `speech synthesize`
- `speech transcribe`
- `speech profile`
- `vision caption`
- `vision inspect`
- `vision ground`
- `vision segment`
- `vision track`
- `vision track-live`
- `vision pose`
- `vision flow`
- `vision geometry`
- `vision geometry-multiview`
- `vision image-to-3d`
- `vision image-to-3d-trellis2`
- `vision image-to-3d-multiview`
- `vision depth-video`
- `vision ocr`
- `music analyze`
- `music generate`
- `music transcribe`
- `sfx generate`
- `video generate`
- `video cosmos3`
- `video export-latents`
- `api serve`
- `open-webui`
- `plugin`
- `status`

Operational workflows:

- `model list`
- `model runtime`
- `model benchmark`
- `model capabilities`
- `model info`
- `model pull`
- `model remove`
- `model storage`
- `model repair-manifests`
- `setup`
- `agent onboard`
- `agent install-pi`
- `agent start`

## Model-Focused Guides

Some topics support model focus. The CLI validates that the requested model is
covered by the guide before printing:

```bash
mere.run guide image generate --model image-zimage-nano
mere.run guide text chat --model text-chat-gemma4
mere.run guide speech transcribe --model speech-asr-parakeet
mere.run guide music generate --model music-acestep-xl-turbo
mere.run guide music analyze --model music-acestep-xl-turbo-lm4b
mere.run guide music transcribe --model music-muscriptor-medium
```

If the model does not belong to the topic, the command prints a validation error
with the supported model ids.

## Agent Usage

Agents should keep their own prompt context small:

1. Run `mere.run guide --list` to find the topic.
2. Run `mere.run guide <command path>` before coaching a creative or advanced
   workflow.
3. Use `--json` when the agent needs a structured payload with topic metadata
   and Markdown content.

The guide command is the canonical source for per-command usage advice; skills
should route to it instead of duplicating cookbook content.
