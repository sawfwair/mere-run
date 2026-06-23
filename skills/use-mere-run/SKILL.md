---
name: use-mere-run
description: Help a newcomer use the public `mere.run` CLI without needing to know the repo internals. Use when the user wants to install or launch `mere.run`, understand what commands exist, choose and pull a model, run a first image/text/speech/vision/music/video workflow, configure model storage, serve the local API, or troubleshoot beginner CLI errors such as command-not-found, missing models, unsupported hardware, disk/cache location, or API key requirements.
---

# Use mere.run

## Overview

Help the user get useful output from the public local-first `mere.run` CLI. Optimize for a person who does not know the command tree, model IDs, or where models live.

## Operating Stance

- Start with the user's desired result: image, chat, code, speech, transcription, image/video understanding, segmentation/tracking, music, video, model management, API serving, or general setup.
- Prefer doing a tiny diagnostic command over explaining abstractly. Use `--help`, `model capabilities`, and `model list` to discover the local truth.
- If the user has an installed binary, use `mere.run ...`. If they are in a source checkout or the binary is missing, use `swift run mere.run ...`.
- Do not assume a model is installed. Check `mere.run model list` or pull the needed model first.
- Keep commands copy-pasteable and minimal. Add advanced flags only when the user asked for them or the error points there.
- Explain stderr/progress as normal diagnostic output; preserve stdout when the user needs machine-readable results.
- When the user asks for a good creative or advanced result, not just a command, run `mere.run guide <command path>` or read the matching `Sources/MereRunCLI/Guides/*.md` resource first. Use the guide to translate their vague request into concrete prompt ingredients, model pulls, flags, and an iteration plan.

## First Five Minutes

Run these in order, adapting `mere.run` to `swift run mere.run` inside a checkout:

```bash
mere.run --help
mere.run guide --list
mere.run model capabilities
mere.run model list
```

If the user wants guided setup instead of picking commands manually:

```bash
mere.run setup
```

When helping from a source checkout:

```bash
swift run mere.run --help
swift run mere.run guide --list
swift run mere.run model capabilities
swift run mere.run setup
```

## Choose The Command

Use `model capabilities` as the recommendation source before large downloads. Common public pullable IDs include:

- Image: `image-klein-base`, `image-klein-max`, `image-zimage-max`
- Text chat: `text-chat-gemma4`, `text-chat-gemma4-nano`, `text-chat-gemma4-max`, `text-chat-q36-nano`
- Code/agent: `text-code-qwen3`, `text-agent-qwen35-9b`
- Embeddings: `text-embed-qwen3-0.6b`
- Speech: `speech-tts-qwen3-nano`, `speech-tts-qwen3-customvoice`, `speech-asr-qwen3`, `speech-asr-parakeet`
- Vision: `vision-ocr-lighton`, `vision-segment-sam31`, `vision-ground-falcon-perception`
- Music: `music-acestep`
- Video: `video-ltx23-av-mlx`

Avoid local-path-only IDs for first-time users unless they already have the model files.

## Common Workflows

Pull and inspect:

```bash
mere.run model capabilities
mere.run model pull text-chat-gemma4
mere.run model info text-chat-gemma4
```

Chat:

```bash
mere.run text chat --stream --prompt "Explain local inference in one paragraph."
```

Generate an image:

```bash
mere.run model pull image-klein-max
mere.run image generate --prompt "a ceramic mug in soft morning light" --output ./mug.png
```

Speech:

```bash
mere.run model pull speech-tts-qwen3-nano
mere.run speech synthesize "Hello from mere.run" --output ./hello.wav
```

Vision:

```bash
mere.run vision inspect ./image.png "Describe this image."
mere.run model pull vision-segment-sam31
mere.run vision segment ./photo.jpg --prompt "a person" --output ./mask.png
```

Local API:

```bash
mere.run model pull text-chat-gemma4
mere.run api serve --engine text-chat-gemma4
```

For non-loopback API serving, require an explicit key:

```bash
export MERERUN_API_KEY=change-me
mere.run api serve --host 0.0.0.0 --api-key "$MERERUN_API_KEY"
```

## Command Cookbooks

The CLI ships offline cookbooks. Load the relevant guide before coaching creative, model-specific, or advanced workflows:

```bash
mere.run guide --list
mere.run guide image generate
mere.run guide music generate --model music-acestep
mere.run guide video generate --json
```

Inside a source checkout, use:

```bash
swift run mere.run guide <command path>
```

If the binary cannot run yet, read the corresponding resource under `Sources/MereRunCLI/Guides/`. The guide command is canonical for per-command purpose, required models, install/check commands, parameters, prompting patterns, examples, iteration tips, troubleshooting, and source links.

## Model Storage

Default model store:

```text
~/Library/Application Support/MereRun/models
```

Use an external disk for a session:

```bash
export MERERUN_MODELS_DIR=/Volumes/Models/mere.run
mere.run model list
```

Move the Hugging Face snapshot cache when downloads are large:

```bash
export MERERUN_HUB_CACHE=/Volumes/Models/huggingface
mere.run model pull image-zimage-max
```

## Troubleshooting

- `mere.run: command not found`: check `which mere.run`; from a repo checkout use `swift run mere.run ...`.
- Model missing or "not found": run `mere.run model list`, then `mere.run model pull <id>`, or pass a local model path with the command-specific `--model` or `--model-root`.
- Pull blocked by hardware support: run `mere.run model capabilities --all`, choose a smaller recommended model, or use `--allow-unsupported` only when the user explicitly accepts the risk.
- Download or disk-space problems: set `MERERUN_MODELS_DIR` and optionally `MERERUN_HUB_CACHE` to a larger disk, then retry the pull.
- API works locally but not remotely: non-loopback binds require `--api-key` or `MERERUN_API_KEY`.
- A creative command produces weak results: load `mere.run guide <command path>` and iterate prompt, seed, model, and command-specific controls from the guide.
- A command is unfamiliar: run `mere.run <group> --help` and then `mere.run <group> <command> --help`.

When the user is stuck, ask for the exact command they ran, the first error line, and the output of `mere.run model list`.
