# SenseNova U1.5 (SenseNova)

## Purpose

This guide is for mere.run Studio and command-line users.

Create and edit images with SenseNova U1.5.

## Start here

For a direct generation, describe visible content and layout in natural
language. For editing, pair the requested change with explicit preservation
constraints. More complex briefs can be planned into subject, layout, text, and
edit requirements before submission.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Change the backpack to forest green. Preserve the person's face, posture,
jacket, background, shadows, and camera framing. Keep the backpack's zipper
and straps in their original positions.
```

## Controls and variants

Begin with direct prompting. Provider prompt-enhancement recipes are optional
preparation methods and don't imply that every upstream script is bundled.
Review an expanded prompt before using it. Retain the managed runtime's settings
as the initial baseline.

## Iterate and review

If colors become excessive or details look harsh, compare a lower supported
guidance setting. If several edits conflict, separate them into smaller passes.
Check dense text, small faces, hands, and constrained object counts explicitly.

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
mere.run guide --model image-sensenova-u1-5-8b-mot
```

To inspect the available command options, run the following command:

```bash
mere.run image generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `image-sensenova-u1-5-8b-mot` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [U1.5 cookbook](https://github.com/OpenSenseNova/SenseNova-U1/blob/refs/heads/feat/u1.5/docs/u1.5_best_practices.md)
- [Image runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/image.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
