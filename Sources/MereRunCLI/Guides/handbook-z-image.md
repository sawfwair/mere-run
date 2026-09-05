# Z-Image (Tongyi-MAI)

## Purpose

This guide is for mere.run Studio and command-line users.

Write generation prompts for Z-Image base and Turbo packages.

## Start here

Start with a compact natural-language scene. Specify a subject, visible action
or arrangement, setting, and lighting. Add details only when they correct an
observed omission.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
A cyclist in a yellow raincoat waits beside a silver bicycle under a stone
archway. Wet cobblestones reflect the overcast sky. Documentary photograph,
full-body framing.
```

## Controls and variants

Nano, max, and base entries must keep their own defaults. Avoid importing an
undistilled schedule into a Turbo workflow. The local command exposes
dimensions, seed, steps, and guidance; negative conditioning depends on the
active guidance path.

## Iterate and review

If the prompt becomes inconsistent, remove competing lighting or camera
descriptions. Compare one change at a time. For an image-conditioned workflow,
inspect the input and change strength before rewriting every sentence.

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
mere.run guide --model image-zimage-nano
```

To inspect the available command options, run the following command:

```bash
mere.run image generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `image-zimage-nano`
- `image-zimage-max`
- `image-zimage-base`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Official repository](https://github.com/Tongyi-MAI/Z-Image)
- [Image runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/image.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
