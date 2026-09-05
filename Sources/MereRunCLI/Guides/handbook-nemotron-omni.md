# Nemotron 3 Nano Omni (NVIDIA)

## Purpose

This guide is for mere.run Studio and command-line users.

Ask modality-specific questions with Nemotron Nano Omni.

## Start here

Select the supported input modality, attach the source media, and state the
exact question. Keep a factual extraction task separate from interpretation.
Label multiple media inputs consistently.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Review the attached clip. Describe the visible action in order, then list
words that are clearly audible. Separate visual observations from audio
observations and mark uncertain transcription.
```

## Controls and variants

Check the local capability profile before using an upstream audio, image, or
video example. The word "Omni" in a model name doesn't mean every serving route
accepts every modality. Use local file or media inputs supported by the chosen
surface.

## Iterate and review

If the response confuses modalities, ask a narrower question and inspect each
source independently. Verify timestamps and quoted speech against the media.
Long recordings might need task-appropriate segmentation.

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
mere.run guide --model omni-chat-nemotron3-nano-30b-a3b-bf16
```

To inspect the available command options, run the following command:

```bash
mere.run text chat --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `omni-chat-nemotron3-nano-30b-a3b-bf16` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Omni model card](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
- [Text runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/text.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
