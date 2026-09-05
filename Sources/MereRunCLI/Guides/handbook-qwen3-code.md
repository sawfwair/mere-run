# Qwen3 code (Qwen)

## Purpose

This guide is for mere.run Studio and command-line users.

Make Qwen3 code requests testable.

## Start here

Specify the language and runtime, provide the function signature or relevant
files, and describe the expected behavior with examples. Keep repository
instructions and untrusted input samples clearly separated.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Implement parseDuration(text) for strings ending in ms, s, or min. Return
milliseconds as an integer. Reject negative values and unknown units.
Include examples for 250ms, 1.5s, and 2min.
```

## Controls and variants

Keep the exact managed checkpoint's template and sampling settings. Use the
runtime's tool schema for agent work; a provider repository example might
require a different tool runner. Set an output budget large enough for the
requested code and tests.

## Iterate and review

If tests mirror a wrong implementation, add independent expected outputs. If a
patch can't apply, provide the actual file contents or interface. Run generated
code and inspect failures before requesting more features.

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
mere.run guide --model text-code-qwen3
```

To inspect the available command options, run the following command:

```bash
mere.run text code --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `text-code-qwen3` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Qwen3-Coder repository](https://github.com/QwenLM/Qwen3-Coder)
- [Text runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/text.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
