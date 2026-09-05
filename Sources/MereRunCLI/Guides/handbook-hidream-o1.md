# HiDream O1 (HiDream-ai)

## Purpose

This guide is for mere.run Studio and command-line users.

Describe generation and reference edits for HiDream O1.

## Start here

Write the intended output as a complete visual description. With references,
make the desired changes explicit and name the features to preserve. Keep
identity, pose, background, and visual style as separate decisions.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Keep the person's face, hairstyle, and seated pose from the reference.
Replace the plain wall with a softly lit bookshop interior. Preserve the
camera angle and use natural indoor colors.
```

## Controls and variants

Use the local image command's reference controls rather than placing file names
in prose. O1 and O1 Dev have separate checkpoints; retain each one's defaults.
Start with a single reference and a single edit before combining several
references.

## Iterate and review

If identity shifts, remove unrelated style demands and inspect the reference
crop. If an untouched region changes, narrow the requested edit or use a
preservation mask. Review the Dev variant independently; this guide doesn't
establish equivalent output quality.

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
mere.run guide --model image-hidream-o1
```

To inspect the available command options, run the following command:

```bash
mere.run image generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `image-hidream-o1`
- `image-hidream-o1-dev`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [O1 model card](https://huggingface.co/HiDream-ai/HiDream-O1-Image)
- [Image runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/image.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
