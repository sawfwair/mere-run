# DiffusionGemma (Google)

## Purpose

This guide is for mere.run Studio and command-line users.

Give DiffusionGemma a bounded task and explicit output requirements.

## Start here

State the task, relevant context, and expected deliverable in separate
paragraphs. Begin with a small answer before attempting long code or
multi-document output. This local recipe doesn't substitute the Gemma 4 token
format for diffusion-specific behavior.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Write a short Python function that groups nonempty strings by their first
character. Preserve input order within each group. Return only the function,
followed by two example calls.
```

## Controls and variants

Retain the managed diffusion runtime's defaults for the first comparison. Token
budgets and diffusion generation behavior can differ from autoregressive chat.
Record the exact checkpoint and generation configuration when evaluating
revisions.

## Iterate and review

If output repeats or stops before the deliverable is complete, reduce the task
and inspect runtime compatibility and output limits. Run generated code against
meaningful examples. Provider-specific diffusion prompting guidance is
unverified.

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
mere.run guide --model text-chat-diffusiongemma-26b-optiq-4bit
```

To inspect the available command options, run the following command:

```bash
mere.run text chat --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `text-chat-diffusiongemma-26b-optiq-4bit` model.

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
