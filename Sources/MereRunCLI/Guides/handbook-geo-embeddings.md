# TESSERA and OlmoEarth embeddings

## Purpose

This guide is for mere.run Studio and command-line users.

Prepare temporal geospatial inputs for TESSERA and OlmoEarth.

## Start here

Choose the encoder family and supply its required sensor and temporal inputs.
Maintain band ordering, timestamps, masks, and spatial metadata. An embedding is
a numeric representation of the imagery. The encoder computes embeddings from
images rather than a natural-language prompt.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Preparation example:
A time-ordered tile sequence with consistent extent, documented
observations, and explicit missing-data masks. Keep the encoder model ID and
output dimensions with the resulting tensor.
```

## Controls and variants

Use the local **Earth** controls for the selected encoder. TESSERA size variants
and OlmoEarth size variants have their own input and output contracts. Keep
dimensions, patch settings, and preprocessing stable across a comparison.

## Iterate and review

If embeddings can't be compared, check model identity and feature dimensions
first. If a classifier built on them fails, inspect whether training data
overlaps the evaluation period or includes location-specific patterns that fail
elsewhere. Similarity doesn't itself establish a land-cover class or a change
event.

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
mere.run guide --model vision-embed-tessera-v2-nano
```

To inspect the available command options, run the following command:

```bash
mere.run geo tessera --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `vision-embed-tessera-v2-nano`
- `vision-embed-tessera-v2-small`
- `vision-embed-tessera-v2-medium`
- `vision-embed-tessera-v2-large`
- `vision-embed-tessera-v2-teacher`
- `vision-embed-olmoearth-v12-nano`
- `vision-embed-olmoearth-v12-tiny`
- `vision-embed-olmoearth-v12-small`
- `vision-embed-olmoearth-v12-base`

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
