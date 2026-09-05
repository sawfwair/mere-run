# Ornith 1.0

## Purpose

This guide is for mere.run Studio and command-line users.

Give Ornith 1.0 a small, verifiable coding task.

## Start here

Provide a concrete function or module, the requested change, and an example of
the correct behavior. Keep this version separate from the 1.5 family. Its
provider-specific prompting instructions remain unverified.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Write a function that merges overlapping closed intervals. Adjacent
intervals that only touch at an endpoint count as overlapping. Include
empty-input and single-interval examples.
```

## Controls and variants

Select the 9B or 35B managed entry and the command supported by that entry. Keep
initial settings at local defaults. Use `model info` to distinguish the
checkpoint source and format before comparing results.

## Iterate and review

If the response solves a different interval convention, put the endpoint rule
next to the example. Test generated code before reusing it. Don't copy a 1.5
tool template into a 1.0 checkpoint with a different format or runtime.

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
mere.run guide --model text-agent-ornith-9b
```

To inspect the available command options, run the following command:

```bash
mere.run text code --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `text-agent-ornith-9b`
- `text-agent-ornith-35b`

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
