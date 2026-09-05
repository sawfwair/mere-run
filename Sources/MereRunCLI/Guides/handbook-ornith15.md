# Ornith 1.5 (Ornith)

## Purpose

This guide is for mere.run Studio and command-line users.

Write a coding brief for an agent for Ornith 1.5.

## Start here

Describe the repository task, relevant files, intended behavior, and tests.
Include enough source context for the requested change. For image-assisted work,
identify the screenshot's role and use a vision-capable entry.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Fix the parser so an empty optional field is accepted but a missing required
field returns a typed error. Keep the public API stable. Add tests for both
cases and summarize the changed behavior.
```

## Controls and variants

Choose the managed text or vision and precision variant deliberately. Keep tools
in the runtime's declared tool interface. The 1.5 model card supports this
family; it doesn't establish the prompting behavior of 1.0 entries.

## Iterate and review

If the result edits unrelated files, narrow the requested scope and acceptance
criteria. If it reports that tests passed, inspect actual command output.
Compare precision variants using the same cases, not merely a fluent
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
mere.run guide --model text-agent-ornith-35b-mlx-4bit
```

To inspect the available command options, run the following command:

```bash
mere.run text chat --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `text-agent-ornith-35b-mlx-4bit`
- `text-agent-ornith-35b-mlx-6bit`
- `text-agent-ornith-35b-mlx-8bit`
- `text-agent-ornith-35b-mlx`
- `vision-chat-ornith-35b`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Ornith 1.5 model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
- [Text runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/text.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
