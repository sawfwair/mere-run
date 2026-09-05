# MiniMax H3 FL2VA and FastH3 (MiniMax)

## Purpose

This guide is for mere.run Studio and command-line users.

Write an H3 audiovisual timeline with explicit keyframe intent.

## Start here

Describe visuals and action in the main timeline, summarize environmental sound
separately, and specify any background music separately. When first or last
frames are supplied, explain how the action develops between those images.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
integrated_multimodal_description: [Shot 1] A glass marble rolls from the
left side of a wooden table toward a shallow brass dish. A fixed camera
records the event in a close shot without cuts.
overall_soundscape: A soft rolling rattle grows into a brief brass ring as
the marble settles.
non_diegetic_music: No background music.
```

## Controls and variants

Use the FL2VA `--image` and `--end-image` inputs for keyframes, not Ref2VA
references. Match described event timing to the requested duration. BF16 and Q8
entries use different weight precision. Legacy and FastH3 packages also have
different runtime settings. FastH3 is a distilled model with its own schedule.

## Iterate and review

If the final frame is inconsistent with the action, simplify the path between
the keyframes. If sound competes with dialogue, narrow the soundscape. Compare
frame continuity, timing, and audible events independently of generation speed.

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
mere.run guide --model video-minimax-h3-fl2va-mlx
```

To inspect the available command options, run the following command:

```bash
mere.run video generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `video-minimax-h3-fl2va-mlx`
- `video-minimax-h3-fl2va-bf16-mlx`
- `video-minimax-h3-fl2va-8bit-mlx`
- `video-minimax-h3-fasth3-vsa-datafree-mlx`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Base video prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
- [Video runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/video.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
