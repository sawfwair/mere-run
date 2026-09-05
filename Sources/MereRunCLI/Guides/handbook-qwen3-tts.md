# Qwen3 TTS (Qwen)

## Purpose

This guide is for mere.run Studio and command-line users.

Keep spoken words separate from voice instructions in Qwen3 text-to-speech (TTS)
synthesis.

## Start here

Write only the words to be spoken in the synthesis text. Put voice quality,
pace, accent, and delivery into the voice and style controls. Choose the
documented style or clone workflow for the selected managed model.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Spoken text: The next stop is Harbor Square. Take your belongings
with you.
Voice description: A clear adult narrator with a warm tone, measured pace,
and neutral delivery.
```

## Controls and variants

VoiceDesign and CustomVoice are different checkpoints. Use the local speech
cookbook to select the mode and any reference profile. For cloning, use a clean
single-speaker reference with a matching transcript. Expand ambiguous
abbreviations and numbers in the spoken text.

## Iterate and review

If style instructions are read aloud, move them out of the synthesis text. If
pronunciation fails, spell the phrase to reflect the intended pronunciation or
split it into shorter sentences. Listen for reference noise, timing, and
consistency across several lines.

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
mere.run guide --model speech-tts-qwen3-nano
```

To inspect the available command options, run the following command:

```bash
mere.run speech synthesize --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `speech-tts-qwen3-nano`
- `speech-tts-qwen3-customvoice`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Qwen3-TTS usage](https://github.com/QwenLM/Qwen3-TTS)
- [Speech runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/speech.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
