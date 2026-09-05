# Klein shared components

## Purpose

This guide is for mere.run Studio and command-line users.

Understand the shared Klein component bundle.

## Start here

This catalog entry supplies components used by a image generation model. Before
writing a prompt, select a complete Klein generation model in Studio. Model
resolution connects the model to its required local components.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

To prepare a Klein generation model in Studio, follow these steps:

1. In **Image**, select **Generate**.
2. In **Model**, select **`image-klein-max`**.
3. Confirm that the required model files are available locally.
4. Read the FLUX.2 Klein guide.
5. Generate an image with the example prompt.

## Controls and variants

Keep shared files in the managed model store or a registered location. Use the
`model info` command and model validation to diagnose missing files. Changing a
prompt can't repair an incomplete component installation.

## Iterate and review

If a load reports missing encoders or decoders, inspect the generation model's
manifest and installation. Don't select `image-klein-shared` as a standalone
generator or mix components from unrelated revisions.

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
mere.run guide --model image-klein-shared
```

To inspect the available command options, run the following command:

```bash
mere.run model info --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `image-klein-shared` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For runtime details, see [Image runtime
documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/image.md).

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
