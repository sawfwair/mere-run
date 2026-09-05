# North Mini Code

## Purpose

This guide is for mere.run Studio and command-line users.

Use North Mini Code for a constrained code deliverable.

## Start here

Supply the language, version, available libraries, and exact behavior. Ask for a
patch or function rather than an open-ended discussion. This recipe is based on
the local code workflow although original-publisher guidance remains unverified.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Implement a Swift function that converts a byte count to a decimal display
string. Use units B, kB, and MB, with one decimal place for kB and MB.
Reject negative input with a typed error.
```

## Controls and variants

Use the managed code entry with the local code workflow. Establish a short
baseline before requesting a repository-wide change. Preserve the runtime's
template and model defaults.

## Iterate and review

If the answer uses unavailable libraries, list the dependencies explicitly. If
the API differs from the requirement, add a function signature. Run the examples
and check edge cases rather than treating compilation alone as correctness.

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
mere.run guide --model text-code-north-mini
```

To inspect the available command options, run the following command:

```bash
mere.run text code --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `text-code-north-mini` model.

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
