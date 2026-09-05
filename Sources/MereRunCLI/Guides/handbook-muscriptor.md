# Muscriptor

## Purpose

This guide is for mere.run Studio and command-line users.

Prepare audio for Muscriptor transcription. Muscriptor exports notes in Musical
Instrument Digital Interface (MIDI) format.

## Start here

Supply a musical recording and choose the local transcription workflow. This is
transcription from audio to musical notes; a text prompt doesn't specify the
notes. Start with a short representative passage before processing the whole
piece.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Review example:
A piano phrase with a clear attack pattern. Compare the exported MIDI
against the recording for pitch, onset, note length, and overlapping voices.
```

## Controls and variants

Choose small, medium, or large according to the local model's availability and
resource limits. Preserve the source audio and exported Musical Instrument
Digital Interface (MIDI) files together. Keep any timing or postprocessing
settings consistent across comparisons.

## Iterate and review

If the output contains too many notes, inspect noisy transients and sustained
resonance. If timing drifts, compare a few bars against the original audio
before quantizing. A readable display of note pitch and duration doesn't
establish an accurate transcription.

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
mere.run guide --model music-muscriptor-small
```

To inspect the available command options, run the following command:

```bash
mere.run music transcribe --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `music-muscriptor-small`
- `music-muscriptor-medium`
- `music-muscriptor-large`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For runtime details, see [Music runtime
documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/music.md).

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
