# FLUX.2 Klein (Black Forest Labs)

## Purpose

This guide is for mere.run Studio and command-line users.

Compose and edit images with the Klein models.

## Start here

Describe the subject, its action, the scene, and the visual treatment in
ordinary sentences. For an edit, identify each reference by what it contributes:
subject identity, composition, or style. State the intended change and the
details that must remain recognizable.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
A red enamel kettle sits on a pale oak counter beside a folded linen towel.
Soft morning light enters from the left. Eye-level product photograph, clean
cream background, subtle enamel reflections.

Edit: Keep the kettle's shape, handle, and framing. Change only the body
color to deep blue.
```

## Controls and variants

Start with the selected model's defaults. Nano, max, and 9B packages differ in
size and precision; base variants are undistilled and require their own
schedule. Use `--ref-image` for explicit references and `--seed` for a
controlled comparison. Treat `--structured-prompt` as an additional local model
workflow, not a property of Klein.

## Iterate and review

If the edit drifts, reduce the number of simultaneous changes and give each
reference one role. Use a mask when exact pixel preservation matters. Inspect
generated lettering separately; a convincing image can still misspell a label.

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
mere.run guide --model image-klein-nano
```

To inspect the available command options, run the following command:

```bash
mere.run image generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `image-klein-nano`
- `image-klein-max`
- `image-klein-9b`
- `image-klein-base`
- `image-klein-base-9b`

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
