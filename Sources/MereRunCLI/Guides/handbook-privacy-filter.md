# Privacy Filter

## Purpose

This guide is for mere.run Studio and command-line users.

Prepare text for Privacy Filter anonymization.

## Start here

Supply the text to anonymize as task input. Select the supported entity-handling
options in the anonymization workflow. Ordinary prose instructions are part of
the text and can themselves be processed; this isn't a chat prompt.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Input example:
Dana can be reached at dana@example.com about reference number 4821.

Review task:
Check that personal identifiers are handled as intended while the useful
sentence structure remains.
```

## Controls and variants

Keep the original in a controlled local location and review the transformed
result. Check the command's entity and output controls before changing the
input. Preserve record boundaries if later work depends on correspondence.

## Iterate and review

If sensitive text remains, inspect the entity coverage and add a human review
step for the dataset. Don't assume an anonymization pass removes every possible
identifying clue. Check both false negatives and unnecessary redactions.

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
mere.run guide --model text-anonymize-privacy-filter
```

To inspect the available command options, run the following command:

```bash
mere.run text anonymize --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `text-anonymize-privacy-filter` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For runtime details, see [Text runtime
documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/text.md).

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
