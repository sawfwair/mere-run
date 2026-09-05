# Qwen 3.5, 3.6, and 3.8 (Qwen)

## Purpose

This guide is for mere.run Studio and command-line users.

Prepare chat, coding, and image tasks for Qwen 3.5, 3.6, and 3.8.

## Start here

State the requested artifact, the relevant context, and a concrete acceptance
check. In multi-turn work, preserve the facts and constraints needed for the
next step. Attach images through the runtime's image input rather than
describing a missing attachment.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Implement a CSV reader for this format. Preserve quoted commas and empty
fields. Show the code and tests for an empty file, a quoted field, and a
final line without a newline.
```

## Controls and variants

Treat the Qwen release, size, and quantization as part of the recipe. Use
reasoning-effort and history handling only where the local profile exposes them.
Flash-Next and 27B are different checkpoints; their speed and settings don't
establish interchangeable prompt behavior.

## Iterate and review

If the answer repeats, inspect the actual generation settings and shorten
contradictory instructions. For coding, supply the language version and relevant
interfaces. For image tasks, ask about visible evidence and verify small text
separately.

## Read this guide offline

Reading the handbook requires neither model weights nor a network connection.
Before running inference offline, download the checkpoint and any optional
components that your workflow uses.

To read the guide in macOS Studio, follow these steps:

1. In **Help**, select **mere.run Guide**.
2. In **Guide collection**, select **Models**.
3. In the search field, enter a family name or model ID.
4. Select the guide.
5. In **Model**, select a variant.

Prompt examples include a **Copy** control. Input examples serve as
file-preparation checklists. Reading a guide doesn't start inference.

To read this guide in a terminal, run the following command:

```bash
mere.run guide --model text-chat-q36-nano
```

To inspect the available command options, run the following command:

```bash
mere.run text chat --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `text-chat-q36-nano`
- `vision-chat-q38-27b`
- `vision-chat-q38-27b-4bit`
- `vision-chat-q38-flash-next-mixed`
- `vision-chat-q38-flash-next-3bit`
- `vision-chat-q38-flash-next-3bit-native-ple`
- `vision-chat-q38-flash-next-4bit`
- `text-agent-qwen35-9b`
- `text-chat-q36-nano-gguf`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Official series repository](https://github.com/QwenLM/Qwen3.8)
- [Qwen cookbook](https://github.com/QwenLM/Qwen-Cookbook)
- [Text runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/text.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
