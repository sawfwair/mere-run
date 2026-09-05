# MMAudio

## Purpose

This guide is for mere.run Studio and command-line users.

Write a focused sound brief for MMAudio.

## Start here

Describe the audible action and acoustic environment. For video-to-audio, let
the supplied clip establish timing and use text to clarify the sound. Keep
unrelated music or dialogue out of the positive prompt when the task is Foley.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
A ceramic bowl is set on a stone counter with a short hard clink. A small
spoon rattles once inside it. Quiet indoor room tone, close and dry sound.
```

## Controls and variants

Use the MMAudio-specific local workflow and its supported negative-prompt
controls. Select the text-only or video-conditioned path explicitly. The managed
large 44 kHz checkpoint has a different component stack from Woosh.

## Iterate and review

If events are mistimed, inspect the source clip and conditioning before changing
adjectives. If unrelated sound appears, simplify the requested scene and use
supported negative conditioning. Listen through silence and event tails as well
as the loudest instant.

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
mere.run guide --model sfx-mmaudio-large-44k-v2
```

To inspect the available command options, run the following command:

```bash
mere.run sfx generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `sfx-mmaudio-large-44k-v2` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [MMAudio repository](https://github.com/hkchengrex/MMAudio)
- [Sound effects runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/sfx.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
