# LTX 2, 2.3, and 2.5 (Lightricks)

## Purpose

This guide is for mere.run Studio and command-line users.

Write a coherent audiovisual shot for LTX.

## Start here

Describe the shot, scene, subject action, camera movement, and sound as one
sequence. Quote spoken dialogue and specify its speaker. Begin with a single
achievable action before adding cuts or several simultaneous events.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
A medium shot shows a potter lifting a small clay bowl from the wheel. The
camera slowly moves closer as water glints on the rim. Soft workshop
ambience and a faint wheel hum. The potter says, "This one is ready."
```

## Controls and variants

Keep LTX 2, 2.3, and 2.5 model settings distinct. Full, distilled, and
audio-to-video packages expose different workflows. Use local quality and
output-mode controls; hosted prompt enhancement, shot features, and service
defaults aren't automatically local features.

## Iterate and review

If motion stalls, simplify the action and describe observable change. If
dialogue is unclear, shorten the line and reduce competing sound. Review video
continuity and the actual audio track separately; a video file alone doesn't
prove synchronized sound.

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
mere.run guide --model video-ltx-av
```

To inspect the available command options, run the following command:

```bash
mere.run video generate --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `video-ltx-av`
- `video-ltx23-av-mlx`
- `video-ltx23-full-mlx`
- `video-ltx23-a2vid-mlx`
- `video-ltx25-distilled-bf16`
- `video-ltx25-full-bf16`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [LTX prompting guide](https://docs.ltx.io/api-documentation/implementation-guides/prompting-guide)
- [Video runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/video.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
