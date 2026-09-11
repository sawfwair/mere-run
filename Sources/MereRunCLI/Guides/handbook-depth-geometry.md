# MoGe2, Video Depth Anything, DA3, and Marigold V2

## Purpose

This guide is for mere.run Studio and command-line users.

Prepare images and video for MoGe2, Video Depth Anything, Depth Anything 3
(DA3), and Marigold V2.

## Start here

Choose the task before the model: single-image geometry, single-image depth,
video depth, or ordered multiview geometry. These inputs use image evidence and
camera conventions rather than a text prompt.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Multiview example:
An ordered set of overlapping views of the same static scene, with original
image sizes and any known camera information preserved.
```

## Controls and variants

Use MoGe2 for its supported still-image workflow, Video Depth Anything for video
depth, and DA3 for its local multiview path. Keep relative and metric Video
Depth Anything variants distinct. Relative depth describes distance
relationships. Metric depth estimates distance in physical units. A relative
depth map can't be interpreted as meters without an appropriate calibration.

Use Marigold V2 for single-image depth when edge detail matters more than scale.
Its output is affine-invariant: depth is recovered up to an unknown scale and
shift per image, so two images can't be compared directly and no camera or point
cloud is produced. Prefer MoGe2 when you need intrinsics or metric units. The
shipped checkpoint predicts log depth, which increases with distance. Inference
runs one step at a fixed timestep rather than a sampled trajectory, so there is
no step count or guidance scale to tune. The longest edge is capped by default
because memory grows with resolution; raise it only if the machine has headroom.

## Iterate and review

If scale jumps or geometry bends, inspect moving objects, view overlap, and
camera assumptions. Review temporal consistency for video. The local DA3 guide
describes point-cloud output; don't label it a finished triangle mesh.

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
mere.run guide --model vision-geometry-moge2-small
```

To inspect the available command options, run the following command:

```bash
mere.run vision geometry --help
mere.run vision depth --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `vision-geometry-moge2-small`
- `vision-depth-vda-small`
- `vision-depth-vda-small-metric`
- `vision-geometry-da3-small`
- `vision-depth-marigold-v2`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 11, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For runtime details, see [Vision runtime
documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/vision.md).

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
