# Cosmos3 Edge (NVIDIA)

## Purpose

This guide is for mere.run Studio and command-line users.

Separate Cosmos3 generation, reasoning, and action tasks.

## Start here

For image or video generation, describe the scene and observable change. For
reasoning, ask a question grounded in the supplied media. For action-conditioned
generation, provide the required model-space action data through the action
interface.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
The camera slowly approaches an open workshop doorway. A hanging lamp sways
slightly, revealing a workbench inside. Keep the room layout and lighting
consistent with the source image.
```

## Controls and variants

Prompt upsampling expands a short prompt into a detailed description. The
provider documents a structured JSON format for this step. The local command
also accepts direct prompts. Keep any prepared caption with the run. Choose the
correct mode and its image or video inputs. Action tensors have domain-specific
widths and normalization; prose is not a substitute.

The upstream upsampler can call an external model endpoint. It isn't required to
read or use this direct-prompt recipe. For a fully offline session, prepare any
expanded captions beforehand or use a locally served preparation model.

## Iterate and review

If generated motion differs from the requested action, inspect the action domain
and tensor values before adjusting the scene description. If generated geometry
changes, compare the source and successive frames. Reasoner answers and
simulated actions require separate review.

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
mere.run guide --model video-cosmos3-edge-mlx
```

To inspect the available command options, run the following command:

```bash
mere.run video cosmos3 --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `video-cosmos3-edge-mlx` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Prompt upsampling](https://github.com/nvidia/cosmos-framework/blob/main/docs/prompt_upsampling.md)
- [Edge model card](https://huggingface.co/nvidia/Cosmos3-Edge)
- [Video runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/video.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
