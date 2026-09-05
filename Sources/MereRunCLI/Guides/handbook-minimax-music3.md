# MiniMax Music3 (MiniMax)

## Purpose

This guide is for mere.run Studio and command-line users.

Write structured captions and lyrics for MiniMax Music3.

## Start here

Begin with a concise musical description. For more control, organize it into
overall musical metadata, vocal character, and arrangement. Keep stage
directions out of sung text and preserve the lyric section structure.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Global Metadata: Gentle indie pop with a moderate walking pulse and a warm,
close mix.
Vocal Details: Clear solo alto, restrained verse delivery, open sustained
chorus vowels.
Arrangement: Soft electric piano intro, bass and brushed drums enter in the
verse, layered guitar supports the chorus.

Lyrics:
[Verse]
A window glows across the bay
[Chorus]
We bring the morning home
```

## Controls and variants

The local Music3 guide maps caption and lyrics inputs to the local pipeline.
Start with the quality sampling tier and a realistic duration. ACE-Step
planners, cover controls, stem extraction, and adapter recipes aren't Music3
controls.

## Iterate and review

If lyrics are cut short, inspect the duration and acoustic-frame limits. If
arrangement details are ignored, reduce the number of conflicting demands.
Compare timing, pronunciation, vocal artifacts, and musical structure by
listening to the complete output.

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
mere.run guide --model music-minimax-music3
```

To inspect the available command options, run the following command:

```bash
mere.run music generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `music-minimax-music3` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Caption rewriter](https://github.com/MiniMax-AI/MiniMax-Music3/tree/main/skills/music-caption-rewriter)
- [model card](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
- [Music runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/music.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
