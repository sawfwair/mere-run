# Qwen3 ASR (Qwen)

## Purpose

This guide is for mere.run Studio and command-line users.

Prepare recordings and context for Qwen3 automatic speech recognition (ASR).

## Start here

Supply the recording through the transcription input. Keep spoken content intact
and use the supported language or context controls for hints. Don't expect a
natural-language request in an unrelated field to change the transcription
contract.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Input preparation example:
A single meeting recording with a known language and an unchanged original
copy.
Review focus:
Names, numbers, speaker overlap, and words near long pauses.
```

## Controls and variants

Check the local transcription help for available language, timestamp, and
context settings. Provider examples can include conditioning that isn't exposed
by every local route. Keep sample-rate conversion and channel handling
consistent.

## Iterate and review

If the transcript includes words not spoken during silence, inspect the audio
and segmentation. If names are wrong, review the relevant phrase against the
original. Before using word or segment timestamps for editing, compare
representative timestamps with the recording.

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
mere.run guide --model speech-asr-qwen3
```

To inspect the available command options, run the following command:

```bash
mere.run speech transcribe --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `speech-asr-qwen3` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Qwen3-ASR usage](https://github.com/QwenLM/Qwen3-ASR)
- [Speech runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/speech.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
