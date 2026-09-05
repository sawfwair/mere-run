# MeBot and Psi

## Purpose

This guide is for mere.run Studio and command-line users.

Prepare task instructions for locally supplied MeBot or Psi checkpoints.

## Start here

First identify the exact local checkpoint and the supported task. These catalog
entries do not declare a managed Hub download. Their names alone don't establish
a provider chat format or a common inference interface.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Task brief:
Summarize the supplied notes in three bullets. Separate decisions from
unresolved questions. Use only information present in the notes, and
identify missing details explicitly.
```

## Controls and variants

Keep instructions, source material, and the desired response format distinct.
Use the selected runtime's message interface rather than manually inserting
special tokens. Confirm the checkpoint's context limit and tool support before
building a long agent conversation.

## Iterate and review

If output contains raw template markers or repeated role labels, check
checkpoint and runtime compatibility before changing the prose. This is a local
task-writing recipe, not verified provider guidance. Obtain the model's
provenance before assigning model-specific defaults.

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
mere.run guide --model text-chat-mebot
```

To inspect the available command options, run the following command:

```bash
mere.run model info --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `text-chat-mebot`
- `text-chat-psi-agent`

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
