# Gemma 4 (Google)

## Purpose

This guide is for mere.run Studio and command-line users.

Structure chat and image questions for Gemma 4.

## Start here

Place persistent behavior and output requirements in system instructions and the
concrete task in the user message. Let the runtime serialize roles and image
content. Don't paste tokenizer control tokens into an ordinary prompt.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Summarize this incident note as: observed facts, likely explanations, and
missing evidence. Keep each section to three bullets. Do not convert a
proposed explanation into an observed fact.
```

## Controls and variants

Choose a text or vision-capable managed entry for the task. The prompt-format
reference describes roles and modality tokens; it isn't a quality guarantee for
every size or quantization. Use the local reasoning and output-limit controls
instead of asking for a specific number of hidden reasoning tokens.

## Iterate and review

If the response ignores the format, shorten the output schema and provide one
small example. For image questions, ask about visible details and attach the
image through the interface. Validate JSON or code separately from the written
explanation.

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
mere.run guide --model text-chat-gemma4
```

To inspect the available command options, run the following command:

```bash
mere.run text chat --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `text-chat-gemma4`
- `text-chat-gemma4-turbo`
- `text-chat-gemma4-12b`
- `text-chat-gemma4-12b-4bit`
- `vision-chat-gemma4-12b`
- `text-chat-gemma4-nano`
- `text-chat-gemma4-max`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Gemma 4 prompt formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
- [Text runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/text.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
