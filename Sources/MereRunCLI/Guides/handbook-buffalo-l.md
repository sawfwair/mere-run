# InsightFace Buffalo-L

## Purpose

This guide is for mere.run Studio and command-line users.

Prepare face-analysis inputs for Buffalo-L.

## Start here

Supply an image with sufficient face detail and use the face workflow's
detection and analysis controls. No text prompt is needed. Keep the full
original image when reviewing crops and detections.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Input example:
A well-lit, upright photo with a visible face, accompanied by its original
resolution and any preprocessing record.
Review focus:
Detection box, alignment, blur, occlusion, and whether multiple faces were
handled correctly.
```

## Controls and variants

Keep preprocessing consistent across an analysis batch. Review how the local
command exports detections or embeddings before combining results. Treat any
comparison threshold as task-specific and measured on representative data.

## Iterate and review

If a face is missed, inspect scale, angle, and occlusion. If two crops compare
closely, do not infer verified identity from the score alone. Preserve
uncertainty and review false matches as well as missed detections.

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
mere.run guide --model vision-face-buffalo-l
```

To inspect the available command options, run the following command:

```bash
mere.run vision face detect --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `vision-face-buffalo-l` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For runtime details, see [Vision runtime
documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/vision.md).

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
