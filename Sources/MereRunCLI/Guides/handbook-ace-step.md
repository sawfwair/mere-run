# ACE-Step 1.5 (ACE-Step)

## Purpose

This guide is for mere.run Studio and command-line users.

Compose a music brief for ACE-Step 1.5.

## Start here

Keep the musical description, sung lyrics, and timing metadata separate.
Describe genre, instruments, tempo feel, vocal delivery, and the arrangement
progression in the caption. Put the actual words to sing in the lyrics input
with concise section tags.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Caption: Warm acoustic folk, steady brushed drums, fingerpicked guitar,
restrained bass, intimate lead vocal, a quiet verse opening into a fuller
chorus.

Lyrics:
[Verse]
We leave a lantern by the door
And hear the rain begin once more
[Chorus]
Carry the light along the shore
```

## Controls and variants

Select a base, supervised fine-tuning (SFT), or turbo variant. Their guidance
and step schedules differ. Independent 1.7B and 4B language model entries are
planners, not standalone audio generators. Begin with the local quality preset,
then record any caption rewrite, duration, planner, or adapter changes.

## Iterate and review

If vocals rush, reduce syllables or allow more duration. If instruments crowd
the vocal, simplify the arrangement. For covers or repainting, inspect the
source audio and edit region before rewriting the caption. Automated candidate
scores don't replace listening.

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
mere.run guide --model music-acestep
```

To inspect the available command options, run the following command:

```bash
mere.run music generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `music-acestep`
- `music-acestep-xl-base`
- `music-acestep-xl-sft`
- `music-acestep-xl-turbo`
- `music-acestep-xl-turbo-lm4b`
- `music-acestep-lm-1.7b`
- `music-acestep-lm-4b`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Provider tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md)
- [Music runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/music.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
