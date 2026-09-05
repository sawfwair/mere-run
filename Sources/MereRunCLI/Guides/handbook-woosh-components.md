# Woosh auxiliary models

## Purpose

This guide is for mere.run Studio and command-line users.

Use the Woosh scoring model and Synchformer as supporting components.

## Start here

The contrastive language-audio pretraining (CLAP) model compares text and audio.
Synchformer extracts visual conditioning for video-to-audio generation. Choose
the consuming workflow rather than trying to generate audio directly with either
component.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
CLAP comparison example:
Score a candidate clip against "a metal latch clicking" and "steady rainfall
on a roof". Listen to the candidate and confirm that the ranking matches the
audible event.
```

## Controls and variants

Keep the scoring model and preprocessing consistent across candidates. For
Synchformer, preserve video timing and use the feature shape expected by the
local workflow. Install the companion before relying on raw-video generation
offline.

## Iterate and review

If a high-scoring clip sounds wrong, treat the score as a retrieval signal and
inspect the audio. If video conditioning fails to load, check feature dimensions
and frame alignment. Component availability is distinct from a complete
generator installation.

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
mere.run guide --model sfx-woosh-clap
```

To inspect the available command options, run the following command:

```bash
mere.run sfx generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `sfx-woosh-clap`
- `sfx-woosh-synchformer`

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
