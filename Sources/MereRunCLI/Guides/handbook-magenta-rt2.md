# Magenta RealTime 2 (Google)

## Purpose

This guide is for mere.run Studio and command-line users.

Steer Magenta RealTime 2 with evolving instrumental descriptions.

## Start here

Describe the musical texture, instrumentation, groove, and energy you want next.
In a live session, change one musical dimension at a time and allow the
continuation to establish the changed musical direction.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Opening: Soft electric piano, sparse brushed percussion, warm bass, relaxed
late-night instrumental.
Next direction: Keep the gentle groove and add a light muted trumpet melody.
Later direction: Reduce percussion and let the piano carry a quiet ending.
```

## Controls and variants

Use the live music workspace for live prompt changes and the generation workflow
for an offline render. Small and base have different runtime costs. Keep
style-conditioning mode and any Musical Instrument Digital Interface (MIDI)
controls recorded with a take.

## Iterate and review

If a transition feels abrupt, make a smaller prompt change and listen across the
boundary. Don't use lyric sections as a promise of sung-word alignment. Review
the captured audio for continuity, clipping, and unwanted changes in
instrumentation.

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
mere.run guide --model music-magenta-rt2-small
```

To inspect the available command options, run the following command:

```bash
mere.run music generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `music-magenta-rt2-small`
- `music-magenta-rt2-base`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [RealTime 2 model card](https://huggingface.co/google/magenta-realtime-2)
- [Music runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/music.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
