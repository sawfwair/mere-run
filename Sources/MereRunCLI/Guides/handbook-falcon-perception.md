# Falcon Perception (TII)

## Purpose

This guide is for mere.run Studio and command-line users.

Ground visible objects with Falcon Perception.

## Start here

Use concrete referring expressions that distinguish the target: object, color,
visible attribute, and spatial relationship. Begin with one query so that false
matches can be reviewed individually.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
The person in the red jacket holding a blue bicycle.

Comparison query: The blue bicycle beside the stone wall.
```

## Controls and variants

Pass expressions through the local `--query` input. Save the annotated image,
JSON metadata, and masks when downstream work needs to inspect the regions. Keep
thresholds and image resolution recorded with results.

## Iterate and review

If there are no detections, simplify the expression and inspect object size. If
there are too many, add a visible attribute. Avoid claims about hidden intent or
identity. Check the original image before promoting boxes or masks into an
authoritative record.

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
mere.run guide --model vision-ground-falcon-perception
```

To inspect the available command options, run the following command:

```bash
mere.run vision ground --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `vision-ground-falcon-perception` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Falcon Perception model card](https://huggingface.co/tiiuae/Falcon-Perception)
- [Vision runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/vision.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
