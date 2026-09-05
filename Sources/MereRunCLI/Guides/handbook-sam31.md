# SAM 3.1 (Meta)

## Purpose

This guide is for mere.run Studio and command-line users.

Select objects with text, boxes, or points in SAM 3.1.

## Start here

Use a short visible object phrase for text segmentation. When text selects the
wrong object, use the supported geometric prompt to identify the instance. Keep
text concepts and geometric coordinates distinct.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Text prompt: the red backpack
Geometry example: a box around the intended backpack, excluding the
neighboring suitcase.
Tracking task: seed the intended object on a clear frame before propagation.
```

## Controls and variants

Still-image and tracking paths expose different prompt controls. Check the local
segment or track help for point, box, and frame conventions. For video, choose a
frame where the target is visible and not heavily occluded.

## Iterate and review

If several instances are selected, make the phrase more specific or use
geometry. Inspect mask edges and later frames, especially after occlusion. A
segmentation mask is candidate geometry; it doesn't identify a person or prove a
semantic claim.

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
mere.run guide --model vision-segment-sam31
```

To inspect the available command options, run the following command:

```bash
mere.run vision segment --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `vision-segment-sam31` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [SAM repository and notebooks](https://github.com/facebookresearch/sam3)
- [Vision runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/vision.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
