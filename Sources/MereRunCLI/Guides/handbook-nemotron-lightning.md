# Nemotron 3.5 Lightning (NVIDIA)

## Purpose

This guide is for mere.run Studio and command-line users.

Write bounded reasoning tasks for Nemotron 3.5 Lightning.

## Start here

Provide the evidence first, then the question and expected result. For agent
tasks, state the allowed tools and the completion checks through the runtime's
tool interface. Use prose to describe intent rather than fabricating tool-call
syntax.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Given the three supplied timings, compare the runs only where batch
size and prompt length match. Report the calculated ratio and list any
comparison that the evidence does not support.

Sample timings for this example:
Run A: Batch size 1; prompt length 128 tokens; elapsed time 2 seconds.
Run B: Batch size 1; prompt length 128 tokens; elapsed time 3 seconds.
Run C: Batch size 4; prompt length 128 tokens; elapsed time 5 seconds.
```

## Controls and variants

Start with the exact Lightning checkpoint's local defaults. Use supported
reasoning controls and a sufficient output budget. Sampling advice from another
Nemotron generation or a generic chat model isn't an exact substitute.

## Iterate and review

If output mixes incomparable measurements, make the comparison columns explicit.
If tools aren't called correctly, verify the server's declared capabilities and
template. Recalculate numerical conclusions independently.

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
mere.run guide --model text-chat-nemotron-35-lightning
```

To inspect the available command options, run the following command:

```bash
mere.run text chat --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `text-chat-nemotron-35-lightning` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Lightning model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
- [Text runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/text.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
