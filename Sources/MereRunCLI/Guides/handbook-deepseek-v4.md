# DeepSeek V4 Flash (DeepSeek)

## Purpose

This guide is for mere.run Studio and command-line users.

Frame a complete reasoning task for DeepSeek V4 Flash.

## Start here

State the objective, available evidence, constraints, and requested deliverable.
Keep the output grounded in supplied material. Provider-specific V4 prompting
guidance remains unverified. This recipe follows the managed local runtime.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Compare these two cache designs for bounded memory, eviction behavior, and
cancellation. Recommend one for the supplied workload and name the
measurement that would change your recommendation.
```

## Controls and variants

Use the local V4-compatible message and reasoning interface. Don't apply a
different DeepSeek release's special tokens or system-prompt restrictions
without checking this checkpoint. Record context and output limits.

## Iterate and review

If the answer turns assumptions into facts, ask for an explicit assumptions
section. If a long request drifts, split the analysis from implementation and
preserve the agreed constraints. Test any resulting code or numerical claim.

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
mere.run guide --model text-agent-deepseek-v4-flash
```

To inspect the available command options, run the following command:

```bash
mere.run text chat --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `text-agent-deepseek-v4-flash` model.

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
