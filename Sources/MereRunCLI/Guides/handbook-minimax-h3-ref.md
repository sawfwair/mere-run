# MiniMax H3 Ref2VA (MiniMax)

## Purpose

This guide is for mere.run Studio and command-line users.

Write reference roles clearly for MiniMax H3 Ref2VA.

## Start here

Provide ordered references through the local reference input and assign each a
clear role in the prompt. Distinguish the source of appearance from the source
of motion. Describe the desired scene and sound without confusing a reference
with a mandatory endpoint.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
subject_definitions:
<Subject 1> takes the person and red jacket from <Picture 1> and the
unhurried walking movement from <Video 1>.

summary:
[reference generation] Show the referenced person crossing a quiet gray
studio.

retention_analysis:
<Subject 1>: fully_preserved - retain the jacket, hairstyle, and walking
style.

detailed_description:
Natural studio photography with soft overhead light.
[Shot 1] The referenced person enters from the left, takes three relaxed
steps across the gray floor, and pauses near the center. The camera is
fixed.

overall_soundscape:
Three soft footsteps and a faint room ambience.

non_diegetic_music:
No background music.
```

## Controls and variants

Use repeatable `--reference` values such as `image:./person.png` and
`video:./walk.mp4`. The example follows the provider's six-section reference
format with original scene content. Keep labels consistent across sections and
match them to the actual supplied assets. Write the descriptions in English;
dialogue and visible text can retain their intended language. FL2VA first-frame
and end-frame inputs and adapters aren't interchangeable with Ref2VA inputs and
adapters.

## Iterate and review

If identity drifts, reduce conflicting references and give each one a single
role. If motion copies unwanted camera movement, inspect the driving reference.
Review the entire clip for consistency and retain reference order with the saved
recipe.

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
mere.run guide --model video-minimax-h3-ref2va-mlx
```

To inspect the available command options, run the following command:

```bash
mere.run video generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `video-minimax-h3-ref2va-mlx` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Reference video prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- [Video runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/video.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
