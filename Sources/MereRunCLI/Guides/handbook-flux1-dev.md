# FLUX.1 dev (Black Forest Labs)

## Purpose

This guide is for mere.run Studio and command-line users.

Write a clear visual brief for FLUX.1 dev.

## Start here

Use a concrete scene rather than a list of quality adjectives. Name the subject
and its placement before the artistic medium. If text must appear, quote a short
label and describe its location.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
A screen-printed poster of a yellow canoe crossing a dark blue lake. Large
simple shapes and visible paper grain. The short title "NORTH LAKE" sits in
a single line across the top.
```

## Controls and variants

Keep the FLUX.1 generation settings; FLUX.2 editing features and schedules don't
transfer automatically. Establish a baseline with a single scene and a fixed
seed. Change the medium or lighting before adding more objects.

## Iterate and review

If text is wrong, reduce its length and allow more space. If the composition is
crowded, remove secondary objects. Use a separate editing or typography workflow
when the final wording must be exact.

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
mere.run guide --model image-flux1-dev
```

To inspect the available command options, run the following command:

```bash
mere.run image generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `image-flux1-dev` model.

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
