# TerraMind fire and flood

## Purpose

This guide is for mere.run Studio and command-line users.

Prepare TerraMind fire and flood inputs.

## Start here

In **Earth**, select the task that matches the fire or flood model. Provide the
expected imagery and band layout. These models use prepared geospatial inputs; a
text prompt doesn't replace missing bands or a mismatched projection.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Preparation example:
A tile with documented sensor, acquisition date, band order, pixel size,
nodata value, and geographic extent. Retain those fields beside the
inference output.
```

## Controls and variants

Keep fire and flood model selection explicit. Use the local task help and
preflight to inspect tensor names and shape requirements. Preserve masks for
nodata and cloud contamination where the workflow expects them.

## Iterate and review

If outputs look plausible but shifted, inspect georeferencing and resizing. If
predictions change sharply between tiles, check normalization, bands, and
acquisition conditions. A model prediction isn't a verified hazard observation.

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
mere.run guide --model vision-flood-terramind-base
```

To inspect the available command options, run the following command:

```bash
mere.run geo flood --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `vision-flood-terramind-base`
- `vision-fire-terramind-base`

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
