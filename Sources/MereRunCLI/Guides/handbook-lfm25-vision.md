# LFM2.5 vision (Liquid AI)

## Purpose

This guide is for mere.run Studio and command-line users.

Ask specific image questions with LFM2.5 VL-3B.

## Start here

Attach one image and state the extraction or reasoning task. For multiple
inputs, use stable labels such as Media-1 and Media-2 and explain the
comparison. Ask for uncertainty when labels or objects are unclear.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Read the visible labels on this equipment panel. Return a list containing
the label text and its approximate position. Mark partially obscured text as
uncertain instead of completing it.
```

## Controls and variants

Start with the local image path and the managed VL-3B entry. Preserve enough
image detail for optical character recognition (OCR) or object localization. The
provider's multi-image and tool examples are useful patterns; verify that the
selected local command exposes the same inputs.

## Iterate and review

If a small label is misread, crop the relevant region and retry the same
question. For grounding, verify coordinate conventions before drawing boxes. A
fluent caption doesn't prove accurate localization.

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
mere.run guide --model vision-chat-lfm25-3b-8bit
```

To inspect the available command options, run the following command:

```bash
mere.run text chat --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `vision-chat-lfm25-3b-8bit` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Vision capabilities](https://docs.liquid.ai/lfm/key-concepts/vision-capabilities)
- [Text runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/text.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
