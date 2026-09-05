# RoFormer separation and cleanup

## Purpose

This guide is for mere.run Studio and command-line users.

Separate or clean audio with the managed RoFormer models.

## Start here

Choose the model for the intended operation: two-stem separation, four-stem
separation, dereverberation, or denoising. Supply the audio file directly. These
tasks have no free-text prompting stage.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Workflow example:
Use a separation model to isolate the desired stem. Listen to the stem and
its complement. Apply cleanup only if needed, keeping the earlier outputs
for comparison.
```

## Controls and variants

Keep chunking, overlap, sample-rate handling, and export format recorded.
Four-stem and two-stem models separate different audio sources. Denoising and
dereverberation checkpoints aren't interchangeable.

## Iterate and review

If speech or instruments sound hollow, compare against the unprocessed input at
matched loudness. Listen to transients and quiet tails where artifacts can be
difficult to detect. Retain every stem and the processing manifest for
downstream editing.

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
mere.run guide --model music-separate-bs-roformer-viperx-1297
```

To inspect the available command options, run the following command:

```bash
mere.run music separate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `music-separate-bs-roformer-viperx-1297`
- `music-separate-bs-roformer-4stem`
- `music-separate-mel-roformer-dereverb`
- `music-separate-mel-roformer-denoise`

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
