# Bonsai image (PrismML)

## Purpose

This guide is for mere.run Studio and command-line users.

Generate a compact image baseline with Bonsai binary or ternary.

## Start here

Start with a short description containing one subject, one environment, and one
lighting choice. This guide uses the local mere.run image workflow;
provider-specific prompting advice remains unverified.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
A white ceramic teapot on a dark green cloth, photographed from slightly
above, diffuse window light, simple still-life composition.
```

## Controls and variants

Start with four steps at 512 × 512 pixels or 1,024 × 1,024 pixels. Use the
schedule shift specified in the model manifest. Keep binary and ternary results
in separate comparisons. Use a fixed seed within each model; identical seeds
across different checkpoints need not yield matching images.

## Iterate and review

If fine detail collapses, simplify the subject and compare the ternary model
before adding steps without a measured reason. Check silhouettes and large color
areas first. Record the model ID with the image so that a favorable result can
be reproduced.

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
mere.run guide --model image-bonsai-binary
```

To inspect the available command options, run the following command:

```bash
mere.run image generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `image-bonsai-binary`
- `image-bonsai-ternary`

## Sources and validation

Provider-specific guidance remains unverified. This original mere.run recipe
follows the local command and runtime documentation.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For runtime details, see [Image runtime
documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/image.md).

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
