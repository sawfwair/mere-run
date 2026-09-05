# Parakeet and Sortformer (NVIDIA)

## Purpose

This guide is for mere.run Studio and command-line users.

Use Parakeet and Sortformer for complementary speech tasks.

## Start here

Parakeet produces a transcript; Sortformer identifies speaker activity over
time, a task called speaker diarization. Supply an audio recording rather than a
free-form prompt. Preserve the original timing when combining transcripts and
speaker segments.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

To combine transcription and speaker segments, follow these steps:

1. Transcribe a clean copy of the recording.
2. Run speaker diarization on audio with the same timeline.
3. Review a speaker change.
4. Review a region with overlapping speech.
5. Combine the reviewed results.

## Controls and variants

Use the transcription and diarization controls independently. Avoid arbitrary
trimming between the two passes. Speaker labels are run-local labels, not
verified identities; assign names only from separate evidence.

## Iterate and review

If speaker labels switch, inspect overlapping or short turns and the diarization
settings. If the transcript is poor, diagnose noise and speech quality
separately. Review both errors before deciding that one model can fix the
other's output.

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
mere.run guide --model speech-asr-parakeet
```

To inspect the available command options, run the following command:

```bash
mere.run speech transcribe --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `speech-asr-parakeet`
- `speech-diarization-sortformer`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For runtime details, see [Speech runtime
documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/speech.md).

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
