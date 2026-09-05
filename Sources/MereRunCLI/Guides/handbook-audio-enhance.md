# AP-BWE and UniverSR

## Purpose

This guide is for mere.run Studio and command-line users.

Prepare audio for AP-BWE and UniverSR enhancement.

## Start here

Choose the bandwidth-extension or general-audio enhancement model that matches
the source. Use the original audio as input and preserve an unchanged copy.
Increasing sample rate can't prove that missing detail has been recovered
faithfully.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Comparison example:
A narrow-band voice recording. Render an enhanced version and compare
consonants, breaths, room tone, and silence against the original at matched
loudness.
```

## Controls and variants

The managed AP-BWE entry describes a conversion from 16 kHz to 48 kHz; use the
local command's rate and model checks. UniverSR has a separate general-audio
contract. Keep chunking and output format fixed while diagnosing artifacts.

## Iterate and review

If high frequencies hiss or ring, inspect the source and compare a short
difficult segment. If boundaries click, review chunk settings. Describe
generated high-frequency content as model-generated enhancement, not measured
original signal.

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
mere.run guide --model audio-enhance-ap-bwe-16kto48k
```

To inspect the available command options, run the following command:

```bash
mere.run audio enhance --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `audio-enhance-ap-bwe-16kto48k`
- `audio-enhance-universr-audio`

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
