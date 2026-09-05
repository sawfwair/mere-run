# Laguna 2.1 (poolside)

## Purpose

This guide is for mere.run Studio and command-line users.

Write a coding or reasoning brief for Laguna S and XS.

## Start here

Present the requested change, relevant code or evidence, constraints, and
acceptance criteria. Ask for a bounded deliverable. For repository work,
describe the files the agent can change and the checks that establish success.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Implement a bounded queue with capacity 16. Producers must wait when it is
full, and cancellation must release waiting tasks. Explain the ownership
rules and include tests for shutdown and cancellation.
```

## Controls and variants

Treat S and XS as separate capacity and performance choices. Preserve the exact
checkpoint's local settings and tool schema. The located S model card is
provenance for that release; it doesn't establish that all guidance applies
unchanged to XS.

## Iterate and review

If the answer sketches a design without completing it, specify the required
files and tests. If a long exchange loses constraints, restate a compact
acceptance checklist with the relevant code. Inspect and run changes before
relying on them.

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
mere.run guide --model text-chat-laguna-s-2-1
```

To inspect the available command options, run the following command:

```bash
mere.run text chat --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `text-chat-laguna-s-2-1`
- `text-chat-laguna-xs-2-1`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Laguna S 2.1 model card](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4-mlx)
- [Text runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/text.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
