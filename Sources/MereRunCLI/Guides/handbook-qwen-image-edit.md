# Qwen Image Edit 2511 (Qwen)

## Purpose

This guide is for mere.run Studio and command-line users.

Write targeted edits for Qwen Image Edit 2511.

## Start here

Attach the image through the editing interface, then state what changes and what
stays. Distinguish appearance changes from layout changes. Name the object by a
visible attribute when several similar objects are present.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Replace the handwritten date on the white card with "14 MAY". Keep the
card's position, paper texture, surrounding objects, lighting, and camera
angle unchanged.
```

## Controls and variants

Keep the base 2511 model and Lightning variant on their respective schedules. A
Lightning adapter is a generation configuration, not a prompt keyword. Begin
with one edit and use the same input for comparisons.

## Iterate and review

If the wrong region changes, identify the target more precisely or crop the
input for diagnosis. If typography is inaccurate, shorten the text and inspect
at full resolution. Multiple sequential edits can accumulate drift; compare
against the original image.

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
mere.run guide --model image-qwen-edit-2511
```

To inspect the available command options, run the following command:

```bash
mere.run image generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `image-qwen-edit-2511`
- `image-qwen-edit-2511-lightning`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [2511 model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [Image runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/image.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
