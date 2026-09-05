# Woosh generators (Sony AI)

## Purpose

This guide is for mere.run Studio and command-line users.

Describe discrete sound events for Woosh.

## Start here

Name the sound-producing object, action, material, and acoustic character.
Include a simple time sequence only when it matters. For video-conditioned
variants, supply the clip or supported visual features through the video
workflow.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
A heavy wooden drawer slides open, pauses briefly, then closes with a muted
thump in a small quiet room.

Alternative take: A light metal latch clicks twice, followed by a short
spring rattle, close microphone perspective.
```

## Controls and variants

DFlow and Flow use different schedules; the local DFlow example uses four steps
and classifier-free guidance (CFG) 4.5. VFlow and DVFlow are video-conditioned
entries with their own duration contract. Raw video conditioning needs the
Synchformer companion installed locally.

## Iterate and review

If the output contains the wrong action, simplify the event description and
emphasize the audible verb. If it is too quiet, compare phrasing before
normalizing it aggressively. Review event timing, background noise, and the
decay tail.

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
mere.run guide --model sfx-woosh-dflow
```

To inspect the available command options, run the following command:

```bash
mere.run sfx generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `sfx-woosh-dflow`
- `sfx-woosh-flow`
- `sfx-woosh-vflow-8s`
- `sfx-woosh-dvflow-8s`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Woosh repository](https://github.com/SonyResearch/Woosh)
- [Sound effects runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/sfx.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
