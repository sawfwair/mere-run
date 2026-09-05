# Krea 2 (Krea)

## Purpose

This guide is for mere.run Studio and command-line users.

Explore styles with Krea 2 and refine a selected direction.

## Start here

Begin with a visual description. After reviewing the results, add a specific
medium, palette, or lighting choice. Preserve the subject and composition as you
refine the style.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
First pass: A tiny greenhouse on a city rooftop.

Refinement: A tiny rooftop greenhouse at blue hour, hand-painted gouache
illustration, muted teal and warm amber, broad visible brush marks.
```

## Controls and variants

Use Krea 2 Turbo for the documented local generation path. The local cookbook
specifies eight steps, classifier-free guidance (CFG) 0.0, and 1,024 × 1,024
pixels as its starting recipe. Raw is also used by the low-rank adaptation
(LoRA) training workflow; check its local capabilities before selecting a task.
The hosted Krea moodboards and style-reference controls aren't automatically
available in the local Turbo path.

## Iterate and review

If every change produces an unrelated direction, retain the seed and change only
the style phrase. Don't pass unsupported input or reference controls to Turbo.
For a trained style, record the adapter and scale as well as the prompt.

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
mere.run guide --model image-krea2-raw
```

To inspect the available command options, run the following command:

```bash
mere.run image generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `image-krea2-raw`
- `image-krea2-turbo`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Krea 2 prompting guide](https://www.krea.ai/blog/krea-2-deep-dive-walkthrough)
- [Image runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/image.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
