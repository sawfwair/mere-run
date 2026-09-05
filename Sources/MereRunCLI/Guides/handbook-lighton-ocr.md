# LightOnOCR 2 (LightOn)

## Purpose

This guide is for mere.run Studio and command-line users.

Prepare readable pages for LightOnOCR 2.

## Start here

Use the image and output format expected by the optical character recognition
(OCR) workflow. Start with a single upright page with legible text. This task
uses a defined document-reading input convention rather than an unrestricted
chat prompt.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Input example:
One upright scanned page with the full margins visible.
Review checklist:
Heading order, paragraph boundaries, table cells, punctuation, and numerals.
```

## Controls and variants

Keep cropping, resolution, and page orientation consistent. The provider card
describes the model input convention; let the local OCR runtime build it. For
multipage material, preserve page order and page identifiers.

## Iterate and review

If words merge, inspect the scan resolution and contrast. If reading order
fails, check multi-column layout and tables manually. Don't treat Markdown
formatting as proof that the underlying text is accurate.

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
mere.run guide --model vision-ocr-lighton
```

To inspect the available command options, run the following command:

```bash
mere.run vision ocr --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `vision-ocr-lighton` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [LightOnOCR model card](https://huggingface.co/lightonai/LightOnOCR-2-1B)
- [Vision runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/vision.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
