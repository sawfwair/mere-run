# Modalities and training

Use the command cookbook for CLI controls and the selected model handbook for
provider guidance. The same command can route to different architectures with
different inputs and constraints. Do not transfer a default, flag, schedule,
coordinate format, or prompt recipe from one model family to another.

For unfamiliar work, discover the leaf command and then read its help. The
following distinctions guide that discovery; they are not a frozen model list.

## Text, code, and embeddings

Select chat versus GGUF code generation versus embedding output deliberately.
Use `text chat --require-installed` when downloads must be explicit. Inspect
support before combining reasoning controls, structured responses, tool loops,
images, audio, or video with a model. Maximum output tokens and context capacity
are separate limits. Long requests must leave room for the response.

Use a modest first generation, inspect truncation and requested format, then
adjust. Keep stdout text separate from reasoning/stats on stderr. For embeddings,
verify vector shape and downstream model compatibility; vectors from different
encoders do not become comparable because their dimensions match.

## Images, editing, and reconstruction

Choose generation, reference editing, masked editing, or reconstruction from
the actual input and desired output. `--input`, `--ref-image`, masks, strength,
CFG, steps, and LoRA behavior depend on the selected family. Inspect required
shared encoders/VAEs and any structured-prompt companion model before pulling.

Preserve source inputs. Check mask conventions, aspect ratio, output extension,
and reference count in the handbook and preflight. Use explicit seeds when
comparing changes. An annotated image, depth map, point cloud, and textured mesh
are different deliverables; inspect units, camera conventions, required view
counts, and export formats before claiming one satisfies another.

## Speech and general audio

Choose synthesis, cloning, transcription, translation, diarization, microphone
capture, or enhancement explicitly. Voice profiles and reference audio are not
interchangeable with a voice-description prompt. Read the model's requirements
for reference transcripts and sampling before cloning.

For transcription, distinguish the ASR backend from its execution provider.
Parakeet Core ML needs compatible packaged encoder/decoder assets and does not
replace Qwen's translation path. Inspect `speech transcribe --help` before
selecting `--backend`, `--provider`, or packaged asset paths. For stdin/streaming,
match the advertised input format, sample rate, channels, and JSONL mode; a raw
PCM stream is not a WAV file.

Speech commands do not universally support preflight. Check the model pull and
inputs first, run a short sample, then listen or inspect the transcript before
processing a long recording. Confirm output duration, speaker identity where
relevant, clipping, language, and timestamps.

## Music and sound effects

Separate music generation, separation, transcription, analysis, realtime
sessions, and SFX. Read the selected model handbook for caption/lyrics roles,
duration, seeds, candidate selection, required rewriters, and reference audio.
Do not assume a lyrics option or a progress callback exists in every lane.

Inspect generated audio and all declared stems, lyrics, candidates, recipes, or
DAW bundles. For a realtime session, retain its process and control identity and
stop it explicitly when the task finishes. SFX conditioned on video can have
different required inputs and preparation checks from text-to-audio generation.

## Video and world sessions

Determine whether the user wants text-to-video, image-to-video, references,
keyframes, animation, retakes, audio conditioning, or a persistent world session.
Choose the model and output mode from that requirement, then read its handbook.

Frame-count divisibility, frame rate, duration, dimensions, windowing, reference
syntax, adapter schedules, and memory use differ by model. LTX output quality
and output mode are separate choices; video-only output does not promise a
soundtrack. Distinguish generated audio from preserving source audio.
Do not mix compatibility `--variant` with `--quality` or `--output-mode` when
the command rejects that combination. Check reference assets and masks before
expensive generation, and inspect representative frames plus audio afterward.

World/session commands retain state. Discover their lifecycle and save/stop
behavior rather than treating every call as a stateless video export.

## Vision and geospatial inference

Select captioning, OCR, grounding, segmentation, tracking, identity, pose,
flow, depth, or geometry by the requested measurement. Some commands accept
local model roots only. Read coordinates, normalization, labels, thresholds,
frame ranges, and output-sidecar formats before passing prompts or boxes.

Trackers need the correct initialization frame and object identity. Masks and
bounding boxes are candidates to inspect, not automatically verified findings.
For metric outputs, inspect scale/units, coordinate systems, and required camera
or multiview inputs. Preserve detection, tracking, and mask sidecars.

Geospatial models require their documented bands, temporal inputs, resolutions,
and georeferencing; an arbitrary RGB screenshot is not equivalent. Inspect the
`geo` subcommand help, model handbook, and preflight. Verify raster alignment,
CRS, nodata handling, and requested map extent in the result.

## Training and evaluation

Training needs a prepared dataset, a compatible trainable base, a bounded
budget, and a distinct output/checkpoint location. Discover dataset candidates,
inspect captions and manifest statistics, and preflight image LoRA training.
Do not train on generated previews accidentally included in the source folder.
Read the selected recipe and the base-versus-distilled inference guidance.

Text training and some benchmarks use `--dry-run`. Verify what that mode writes,
its data schema, model binding, checkpoint/resume requirements, and expected
runtime before starting. Keep training data and outputs separate. Compare a
baseline and adapted model on held-out examples, not loss alone.

`eval` consumes external, content-addressed packs. Start with `eval pack
validate`, then `eval run --dry-run --json` using explicit model/adapter slots.
Inspect the full plan and any external scorer requirement. Executing a pinned
external scorer is an additional action represented by `--allow-external-scorer`;
apply it only within the user's authorization.

Use an explicit checkpoint for resumable evaluations; `--resume` requires the
same plan. A stopped partial report is not a passing evaluation. `eval promote`
creates a receipt for a complete, gate-passing report; it does not deploy a
model. Keep local evaluation, installed-model qualification, and published
release status as separate evidence.
