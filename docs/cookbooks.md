# Cookbooks

`mere.run guide` is the offline cookbook reader built into the CLI. It prints
Markdown by default and can emit JSON for agents or tools.

```bash
mere.run guide --list
mere.run guide image generate
mere.run guide music generate --model music-acestep
mere.run guide music generate --model music-magenta-rt2-small
mere.run guide video generate --json
```

The cookbooks are packaged with the CLI binary, so installed copies can read
them without internet access. They are meant for practical use: what the command
does, which model to install, which flags matter, prompting patterns, examples,
iteration tips, troubleshooting, and source links.

## Available Topics

Creative and runtime workflows:

- `image generate`
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
- `vision ocr`
- `music generate`
- `video generate`
- `video export-latents`
- `api serve`
- `status`

Operational workflows:

- `model list`
- `model capabilities`
- `model info`
- `model pull`
- `model remove`
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
