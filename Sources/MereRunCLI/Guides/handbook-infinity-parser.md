# Infinity Parser2 Pro (infly)

## Purpose

This guide is for mere.run Studio and command-line users.

Extract document structure with Infinity Parser2 Pro.

## Start here

Choose the document parsing output mode supported by the local optical character
recognition (OCR) workflow. Supply a clean page image or document input. Keep
the output schema fixed when comparing base and Int8 results.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Review example:
For an invoice page, verify the reading order, vendor name, each line item's
quantity, and the association between totals and currency symbols.
```

## Controls and variants

Retain enough resolution for small text. Use the local OCR engine and format
controls rather than inventing a chat instruction format. Base and Int8 outputs
require separate checks, especially for dense tables.

## Iterate and review

If columns interleave, inspect page orientation and crop boundaries. If numbers
are corrupted, compare the rendered page at full size. Machine-readable document
output requires value and layout validation before downstream calculation.

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
mere.run guide --model vision-ocr-infinity-pro
```

To inspect the available command options, run the following command:

```bash
mere.run vision ocr --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `vision-ocr-infinity-pro`
- `vision-ocr-infinity-pro-int8`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Parser2 Pro model card](https://huggingface.co/infly/Infinity-Parser2-Pro)
- [Vision runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/vision.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
