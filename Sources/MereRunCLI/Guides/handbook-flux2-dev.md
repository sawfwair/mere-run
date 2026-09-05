# FLUX.2 dev (Black Forest Labs)

## Purpose

This guide is for mere.run Studio and command-line users.

Create images with FLUX.2 dev.

## Start here

Build a scene in a consistent order: main subject, relationship to other
objects, materials, light, and framing. Write the desired appearance directly.
Separate a composition change from a texture change so that you can compare the
effects.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
A small brass desk clock stands in front of two closed navy notebooks. The
clock face is fully visible. Warm desk-lamp light, dark walnut surface,
three-quarter view, restrained editorial photograph.
```

## Controls and variants

Use the managed dev defaults before changing steps or guidance. The provider's
pro and max examples can inspire composition, but hosted search, reference
limits, and automatic prompt rewriting are separate features. Check the
command's available inputs before adopting them.

## Iterate and review

If extra objects appear, shorten the scene to its essential objects and
relationships. Keep the prompt and seed fixed while changing one setting.
Compare framing, count, and materials before judging fine detail.

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
mere.run guide --model image-flux2-dev
```

To inspect the available command options, run the following command:

```bash
mere.run image generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `image-flux2-dev` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [FLUX prompting guide](https://docs.bfl.ai/guides/prompting_summary)
- [Image runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/image.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
