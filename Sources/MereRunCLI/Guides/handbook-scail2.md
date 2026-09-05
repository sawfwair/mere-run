# SCAIL-2 (Z.ai)

## Purpose

This guide is for mere.run Studio and command-line users.

Animate a referenced subject with SCAIL-2.

## Start here

Supply the reference image and mask, driving video and mask, then describe the
desired subject and scene. Keep the prompt consistent with the supplied
appearance and motion. Choose animation or replacement explicitly.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
The red fabric puppet follows the gentle arm gestures in the driving clip.
Preserve its stitched seams, round head, and short limbs. Keep a fixed
camera and an evenly lit neutral background.
```

## Controls and variants

Use the local `video animate` command. Reference and driving masks are required
inputs; inspect them before generation. Keep segment length, overlap, output
frame rate, and audio-source choices recorded. Use only an adapter compatible
with this workflow.

## Iterate and review

If edges flicker, inspect the masks and motion boundaries. If identity changes,
simplify the prompt and use clearer references. Review across segment joins; a
good first frame doesn't establish stable animation.

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
mere.run guide --model video-scail2-14b-mlx
```

To inspect the available command options, run the following command:

```bash
mere.run video animate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `video-scail2-14b-mlx` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [SCAIL-2 model card](https://huggingface.co/zai-org/SCAIL-2)
- [Video runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/video.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
