# DreamX World (GD-ML)

## Purpose

This guide is for mere.run Studio and command-line users.

Describe a coherent continuation for DreamX World.

## Start here

Begin with a visible scene state and describe the next observable change. Keep
the environment, subject identity, and camera behavior consistent across
continuation chunks. Use the local causal-generation controls for sequence
management.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
The camera continues slowly down the same tiled corridor. The open doorway
remains on the right, and the warm ceiling lights stay fixed. A rolling cart
enters from the far end and moves toward the doorway.
```

## Controls and variants

Use the managed DreamX causal path and its supported conditioning. Record the
initial image or context, chunk settings, seed, and output timing. A causal
continuation model and a full-clip video model have different temporal behavior.

## Iterate and review

If the scene resets between chunks, inspect how context and continuation state
are supplied. If objects change shape, reduce simultaneous motion and camera
changes. Review every boundary and the final scene state, not only isolated
frames.

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
mere.run guide --model video-dreamx-world-5b-ar-mlx
```

To inspect the available command options, run the following command:

```bash
mere.run video generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `video-dreamx-world-5b-ar-mlx` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [DreamX World model card](https://huggingface.co/GD-ML/DreamX-World-5B)
- [Video runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/video.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
