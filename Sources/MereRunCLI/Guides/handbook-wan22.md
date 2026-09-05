# Wan 2.2 (Wan)

## Purpose

This guide is for mere.run Studio and command-line users.

Describe motion for Wan2.2 TI2V-5B.

## Start here

Use the source image to establish appearance and write the intended motion
clearly. Describe one main action and one camera behavior. Avoid contradicting
the image's subject, pose, or lighting in the first trial.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
The paper lantern sways gently in the breeze. Its tassel follows with a
slight delay. The camera remains fixed while the background leaves move
softly.
```

## Controls and variants

Use the local Wan TI2V workflow and its image input. Keep the selected 5B
checkpoint's dimensions, frame count, and generation defaults. Provider prompt
expansion is a separate preparation step; save the expanded text if you use one.

## Iterate and review

If appearance drifts, reduce scene changes and keep the motion local to the
visible subject. If a still frame barely moves, make the action observable.
Check the output's beginning, middle, and end for distortion and unwanted camera
motion.

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
mere.run guide --model video-wan22-ti2v-5b-mlx
```

To inspect the available command options, run the following command:

```bash
mere.run video generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `video-wan22-ti2v-5b-mlx` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Wan2.2 repository](https://github.com/Wan-Video/Wan2.2)
- [Video runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/video.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
