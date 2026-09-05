# Ideogram 4 (Ideogram)

## Purpose

This guide is for mere.run Studio and command-line users.

Create an original local prompt for Ideogram 4 SDNQ.

## Start here

Use a concrete description of the subject, layout, lighting, and any required
text. The local image workflow can expand a short request through a separate
structured-prompt model. Provider-specific version 4 guidance remains
unverified.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
A minimal cream poster with a single dark green pear centered in the lower
half. The title "ORCHARD" appears above it in large dark green serif
letters. Wide margins and a flat paper texture.
```

## Controls and variants

Start with the selected checkpoint's defaults. If using `--structured-prompt`,
install its text model before disconnecting from the network and save the
expanded caption for review. The local cookbook describes a larger prompt token
budget for that path. Expansion can add unintended details, so inspect the
actual submitted caption.

## Iterate and review

If lettering or placement fails, simplify the layout before expanding again.
Avoid treating a successful preflight as evidence of visual correctness. Compare
the direct prompt with the expanded version using recorded settings.

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
mere.run guide --model image-ideogram4-sdnq-uint4
```

To inspect the available command options, run the following command:

```bash
mere.run image generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `image-ideogram4-sdnq-uint4` model.

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
