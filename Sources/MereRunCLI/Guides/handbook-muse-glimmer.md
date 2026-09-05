# Muse Glimmer (Meta)

## Purpose

This guide is for mere.run Studio and command-line users.

Ask grounded visual questions with Muse Glimmer.

## Start here

Attach the image and ask a specific question about visible evidence. State the
response format and how to report uncertainty. For a comparison, identify each
input by a stable label.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Describe the control panel in this image. List each readable label and the
visible state beside it. Mark unreadable labels as unreadable. Do not infer
what an obscured indicator means.
```

## Controls and variants

Use the local model's supported image and reasoning controls. Keep image
resolution sufficient for the requested detail. A managed quantization and an
upstream example might have different resource requirements or output behavior.

## Iterate and review

If labels are wrong, inspect a tighter crop and ask one question per region. If
the response contains unsupported interpretations, separate those statements
from visible observations. Preserve the original image with the response so that
a reviewer can check the claim.

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
mere.run guide --model vision-chat-muse-glimmer-30b
```

To inspect the available command options, run the following command:

```bash
mere.run text chat --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `vision-chat-muse-glimmer-30b` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Muse Glimmer model card](https://huggingface.co/meta-models/Muse-Glimmer-30B)
- [Text runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/text.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
