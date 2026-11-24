# Retake a shot and export an editable timeline

Use Film Tools when a selected shot needs a limited change and an editor needs
the project as editable media. A regional retake creates a candidate for a
frame interval. An OpenTimelineIO export assembles selected shots, explicitly
chosen retakes, linked media copies, and separate audio tracks when available.

This guide starts with an existing Film Tools project that has a selected
clip. It doesn't generate or approve a film from an empty directory.

## Before you begin

Follow [Set up companion workflows](./plugin-workflows.md). Install Film Tools,
Pi, FFmpeg, and FFprobe, and use a compatible native LTX 2.5 runtime and
checkpoint. Check [retake requirements](../runtime/video.md#retake-a-region)
and the [Film Studio workflow](https://plugins-docs.mere.run/plugins/film-tools)
for project planning, model selection, and approval steps.

Install and inspect the project:

```bash
mere.run plugin install mere-film-tools --yes
mere.run plugin doctor mere-film-tools
mere.run plugin run mere-film-tools -- status ./film-project/run.json
```

Use a selected constant-frame-rate source with an LTX-compatible frame count:
`8n+1`, at least nine frames, with dimensions divisible by 32. The interval must
fit within the clip and scheduled shot. Confirm that model and source-media
terms cover the intended use.

## Generate a regional candidate

Replace `beacon-fails` with a shot ID from your project. This example requests
picture changes for frames 24 through 47; the end frame is exclusive:

```bash
mere.run plugin run mere-film-tools -- retake ./film-project/run.json   --shot beacon-fails --start-frame 24 --end-frame 48   --prompt "The keeper looks up as the rotating beacon slows."   --modality video --output-dir ./beacon-retake
```

The candidate remains separate from the selected clip. Its `retake.json`
receipt binds the source and candidate hashes, frame rate, interval, and
modality. Use `audio` for source audio or `both` for both modalities.

Watch `retake.mp4` and compare the region's boundaries, motion, continuity, and
sound with the selected shot. A completed generation isn't an approved take.
Use a fresh output directory for another candidate.

## Export the chosen edit

To include the reviewed candidate, explicitly pass its receipt:

```bash
mere.run plugin run mere-film-tools -- export-timeline ./film-project/run.json   --retake ./beacon-retake/retake.json --output-dir ./film-edit
```

Omit `--retake` to export the selected clips unchanged. Repeat it for additional
reviewed regions. Changed source bytes, conflicting intervals on the same
track, or incompatible frame rates stop export.

Import `film-edit/film.otio` into an editor that supports OpenTimelineIO.
Referenced media is copied into `media/`; relink that folder if you move the
export. Picture follows planned shot order and duration, with gaps where a
selected source ends early. Production audio, dialogue, effects, and score
remain separate tracks when available. Add transitions and complete the mix
and mastering in your editor.

## Check the handoff

Open the export on the receiving workstation. Check linked files, cut points,
retake intervals, gaps, track alignment, and audio playback. Keep the receipt
and source project. The editable export leaves the existing cut in place and
doesn't replace the project's human picture-lock or delivery approvals.

For sound preparation, see [Prepare audio for an edit](./audio-post.md). For
camera-generated clips, see [Hand off World captures and 3D assets](./visual-handoffs.md).
