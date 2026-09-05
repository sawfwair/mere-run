# Bonsai text (PrismML)

## Purpose

This guide is for mere.run Studio and command-line users.

Use a bounded task to compare Bonsai text checkpoints.

## Start here

Write the task plainly and provide only the context needed to answer it. Specify
a compact output format. This is a mere.run starter recipe; the binary and
ternary publishers' exact prompting recommendations remain unverified.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Extract the product name, quantity, and delivery date from the supplied note.
Return null for missing values. Do not guess a year that is not written in
the note.

Note: Three replacement filters are due on May 14.
```

## Controls and variants

Compare the 1-bit and 2-bit entries on the same task and retain their own
checkpoint settings. Use a set of representative inputs, including missing and
contradictory data. A lower memory footprint doesn't prove equivalent extraction
accuracy.

## Iterate and review

If the output contains invented field values, add a worked missing-value
example. If the response contains control-token text, check the template and
runtime pairing. Validate each field against the source note.

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
mere.run guide --model text-chat-bonsai-27b-1bit
```

To inspect the available command options, run the following command:

```bash
mere.run text chat --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `text-chat-bonsai-27b-1bit`
- `text-chat-bonsai-27b-2bit`

## Sources and validation

Provider-specific guidance remains unverified. This original mere.run recipe
follows the local command and runtime documentation.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For runtime details, see [Text runtime
documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/text.md).

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
