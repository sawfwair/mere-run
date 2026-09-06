# Execution and results

Run only after inspecting the request and its preparation checks. Set explicit
output paths and retain the request arguments, seed, model, version, and input
identity. Use a fresh destination when a file already exists unless replacing
it is part of the user's request; not every command prevents overwrites.

## Keep output channels distinct

| Mode | stdout | stderr |
| --- | --- | --- |
| Default command | Its documented text, path, or result | Diagnostics and progress |
| Supported `--json` | Usually one JSON document; inspect the command's schema | Diagnostics |
| `--receipt` | Normal output followed by a final result JSON line | Diagnostics and progress |
| `--progress-json` | Unchanged command output | Progress JSON lines, possibly mixed with other diagnostics |
| Remote `run watch --json-stream` | Worker NDJSON events | Diagnostics |

Do not combine the streams when parsing. `--quiet` is not a JSON mode. A
`--json` option on a discovery command does not imply that generation accepts
it. Text chat's `--response-format json_object`, where supported by the model,
controls generated content; it is not the preflight report or a receipt.

## Interpret receipts and progress

Where help advertises `--receipt`, parse the last stdout line after successful
process completion. A typical receipt is:

```json
{"event":"result","exit":0,"outputs":[{"kind":"image","path":"/abs/mug.png"}]}
```

Check `event`, `exit`, and the process exit code. `outputs[0]` is the primary
artifact when present. Sidecars follow with a `role`; preserve them instead of
assuming a run produces only one file. Output kinds include `image`, `video`,
`audio`, `text`, `json`, and `directory`. A transcription printed only to stdout
can have an empty output list. Failures do not emit a success receipt.

`--receipt` is supported by image/video/music/SFX generation, speech synthesis
and transcription, grounding, segmentation, and tracking. Confirm support in
the installed release before adding it. It conflicts with `--preflight`.

Where supported, `--progress-json` emits events such as:

```json
{"event":"progress","stage":"denoising","step":2,"total_steps":4}
```

`step` is zero-based during a stage. `step == total_steps` completes that
stage, not necessarily the whole run. A stage can restart for another window.
`total_steps: 0` means indeterminate progress and has no terminal stage event.
Use process completion and the receipt to decide whether the job finished.

Progress flags are command-level but callbacks can be model-specific. LTX
video and ACE-Step music do not emit these per-step events. Silence alone does
not justify killing a process or launching a duplicate. Inspect its status,
logs, and expected model behavior. `--progress-json` takes precedence over
`--quiet` for progress; retain non-JSON stderr lines as diagnostics.

For speech, prepare the model with a pull preflight first:

```bash
mere.run model pull speech-tts-qwen3-nano --preflight --json
```

After resolving blockers, run the selected model with a receipt:

```bash
mere.run model pull speech-tts-qwen3-nano
mere.run speech synthesize "Hello from mere.run" \
    --model speech-tts-qwen3-nano --output ./hello.wav \
    --receipt --progress-json
```

## Verify the result

Check the declared files exist, are nonempty, and can be opened as the intended
format. Then inspect the content against the user's request:

- Images: subject, composition, edits, dimensions, and unwanted artifacts.
- Audio/video: decode and play or inspect representative segments; check duration,
  audio presence, synchronization, clipping, and requested content.
- Transcripts, OCR, and model text: inspect actual content, truncation, language,
  and required structure. Validate generated JSON when the task requires it.
- Masks, geometry, and geospatial results: check coordinate conventions,
  dimensions, alignment, units, CRS where relevant, and sidecar metadata.
- Training/evaluation: inspect the saved report and checkpoint, then compare
  outputs using a reproducible evaluation. Training loss alone is not quality.

If an appropriate viewer, decoder, or playback tool is unavailable, report that
specific verification gap. Do not describe an uninspected artifact as correct.

## Recover without duplicating work

Retain the process handle, PID, output paths, or returned remote job reference.
For a local subprocess, use that process's wait/cancel mechanism. `run watch`
accepts SSH and relay job references; it is not a generic local-process monitor.
Inspect local durable directories with `run inspect` and their `events.jsonl`.

After a timeout or disconnect, inspect the existing job before resubmitting.
A request can succeed remotely even when the client loses its response. For
relay jobs, inspect placement blockers while queued and fetch verified results
after completion. Use the workflow reference for resume, cancellation, and
retry differences. Do not delete partial outputs, caches, or checkpoints as a
first response to an error.

For memory pressure, inspect other loaded models, reduce the relevant context,
resolution, duration, or batch size, or select a supported smaller model.
Avoid speculative force flags or launching more work while the first process
still holds memory. Change one variable, repeat preparation checks, then retry.

## Model-driven tool execution

Ordinary chat does not need a tool loop. When the user requests model-driven
file or shell work, inspect `text chat --help` for `--tools`, `--tool-loop`,
`--sandbox-dir`, and iteration limits. Keep writes in an explicit task directory.

`write_file` and `shell_exec` have different approval behavior.
`--auto-approve-tools` can permit supported file writes without a prompt;
`shell_exec` additionally requires `--allow-shell-exec` and **always requires
interactive approval**, even with auto-approval enabled. A noninteractive
shell tool call is denied. `--allow-absolute-tool-paths` expands file-write scope
outside the sandbox; do not enable it merely to avoid a path error.

Treat tool execution as an additional capability to configure within the user's
request. Never promise unattended shell execution from these flags, and do not
confuse a generated tool call with a tool that actually ran.
