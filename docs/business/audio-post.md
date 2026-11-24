# Prepare audio for an edit

Use Audio Post to extract sound from a video, measure delivery levels, or run a
local model for noise removal, reverb reduction, or stem separation. Start with
one clip so you can compare the result with its original before processing a
production batch.

## Install and inspect the source

Follow [Set up companion workflows](./plugin-workflows.md), install FFmpeg, and
check that your Mere runtime supports the selected task:

```bash
mere.run plugin install mere-audio-post --yes
mere.run plugin doctor mere-audio-post
mere.run model capabilities
```

Confirm the recording's rights, channel layout, and intended delivery format.
Use fresh output directories. Audio Post exports the first audio stream, so
check which stream contains the material you intend to process.

## Export and measure without a model

Export a private review copy:

```bash
umask 077
mere.run plugin run mere-audio-post -- process   --input ./scene.mov --output-dir ./audio-review
```

`audio/source.wav` is a 48 kHz floating-point WAV. `audio/loudness.json` reports
integrated loudness in LUFS, loudness range in LU, and true peak in dBTP.
Unavailable or nonfinite measurements appear as `null`. This operation measures
levels and preserves floating-point headroom; it doesn't normalize gain or
certify a broadcast delivery.

## Choose a model task

The task determines which model files you need:

| Goal | Task options | Model ID |
| --- | --- | --- |
| Reduce noise | `--task denoise` | `music-separate-mel-roformer-denoise` |
| Reduce reverb | `--task dereverb` | `music-separate-mel-roformer-dereverb` |
| Separate four stems | `--task stems --stems four` | `music-separate-bs-roformer-4stem` |
| Separate vocals and accompaniment | `--task stems --stems two` | `music-separate-bs-roformer-viperx-1297` |
| Transcribe music to MIDI | `--task midi` | `music-muscriptor-medium` |
| Generate synced eight-second Foley | `--task foley` with a prompt | `sfx-woosh-dvflow-8s` and `sfx-woosh-synchformer` |

Review model terms and pull the required checkpoint separately. MuScriptor has
noncommercial terms. A plugin's availability doesn't grant production rights
for its models or inputs.

## Plan and run stem separation

Write a plan, check its input fingerprint and required model, then execute it:

```bash
mere.run model pull music-separate-bs-roformer-4stem
mere.run plugin run mere-audio-post -- plan   --input ./mix.wav --output-dir ./stem-review --task stems --stems four
mere.run plugin run mere-audio-post -- run ./stem-review/run.json
```

The output includes drums, bass, other, and vocals WAVs. `run.json` records the
outputs and hashes. To continue the recorded workflow, use:

```bash
mere.run plugin run mere-audio-post -- resume ./stem-review/run.json
```

## Listen before delivery

Compare processed and original audio at matched playback levels. Listen for
lost consonants, pumping, phase changes, residual noise, and timing drift.
Check the loudness report after your final mix; an earlier measurement doesn't
cover later edits. Keep the original and record which outputs the editor uses.
For Foley, prepare an eight-second video take and place the generated WAV at
the intended timeline position.

See the [Audio Post reference](https://plugins-docs.mere.run/plugins/audio-post)
for task outputs and [Editable media](./editable-media.md) for timeline handoff.
