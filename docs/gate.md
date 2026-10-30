# Quality gate

`mere.run gate` runs the real user-facing commands against your installed
models at temperature 0, hashes their outputs, checks that repeated runs match,
and compares the results with machine-local baselines. A failed correctness
check exits nonzero.

The gate exists because a stale Metal library once shipped broken decode for
prompts of 1,024 or more tokens to every clone for two months without
detection. The text checks include a long-context lane in that range, and the
determinism check would have caught the failure on the first nightly run.

## What it checks

The default run uses a fixed set of checks. Each check names the model it
needs; the following table lists them.

| Check | Operation | Failure conditions |
| --- | --- | --- |
| `text-gemma-*`, `text-ornith-*`, `text-lfm2-*` (`-short` and `-long`) | `text chat` at temperature 0 with `text-chat-gemma4-12b-4bit`, `text-agent-ornith-35b-mlx`, and `text-chat-lfm25-a1b-8bit`; the long lane uses a prompt of about 1,200 tokens; each check runs twice | Run-to-run nondeterminism, empty output, or output hash drift from the baseline |
| `tts-wav` | `speech synthesize` at temperature 0 with `speech-tts-qwen3-nano` | WAV hash drift or truncated audio |
| `stt-roundtrip` | `speech transcribe` with `speech-asr-parakeet` on the TTS output, twice | Transcript nondeterminism or less than 85% word recovery of the spoken sentence |
| `stt-live-stream` | `speech transcribe --stream` with Parakeet on the TTS output | Less than 85% word recovery of the spoken sentence |
| `ocr-page` | `vision ocr` with `vision-ocr-lighton` on a deterministically rendered text page, twice | Transcript nondeterminism or less than 90% word recovery |
| `image-klein-seed7` | Seeded `image generate` with `image-klein-nano` | PNG hash drift or truncated file |
| `embeddings` | `text embed` with `text-embed-qwen3-0.6b`, twice | Nondeterministic or drifted vector bytes |
| `video-ltx23-draft` | 9-frame native LTX 2.3 distilled render with `video-ltx23-av-mlx` | Generation failure or MP4 decode failure |
| `video-ltx23-full-av` | 9-frame full two-stage LTX 2.3 render with generated audio, using `video-ltx23-full-mlx` | Generation failure, MP4 decode failure, or missing or silent audio |
| `video-ltx23-a2vid` | 9-frame full two-stage render with `video-ltx23-a2vid-mlx`, conditioned on a generated non-silent WAV fixture | Generation failure, MP4 decode failure, or missing or silent audio |

The gate also compares performance: wall time, and `decode_tps` when the engine
reports it. Throughput below 0.75 times the baseline or wall time above 1.5
times the baseline is a warning by default and a failure with `--strict-perf`.
Use `--strict-perf` on a dedicated runner; on a shared workstation, background
load makes strict performance checks noisy.

If a check's model isn't installed, the gate reports the check as skipped.
Every check invokes the same `mere.run` binary that users run, so the gate
covers the CLI surface, model resolution, and the runtime together rather than
a parallel test path.

The video checks are semantic release smokes, not golden-output checks. They
use the minimum valid geometry and frame count, but they still load the real
checkpoints, denoise, write an MP4, decode a frame, and decode audio when the
route promises audio. The report records their hashes without comparing them
to baselines.

## Exhaustive installed-model release smoke

`--all-installed` is a separate fail-closed release mode. It reads the same
inventory as `mere.run model list`, creates one named check for every entry
whose status is `installed`, and runs the user-facing operation for that model.
Coverage includes:

- Text, code, vision-chat, omni-chat, embedding, privacy-filter,
  classification, and decision models
- Speech synthesis, recognition, and diarization models
- Every image and image-editing model
- OCR, segmentation, grounding, face detection, geometry, and depth models
- TripoSR, InstantMesh, and TRELLIS.2 image-to-3D
- Music generation, transcription, and source-separation models
- AuK audio editing and AP-BWE audio enhancement
- Woosh and MMAudio sound-effect generators, CLAP, and Synchformer
- LTX, Wan, Cosmos3, SCAIL-2, MiniMax-H3, and DreamX video and world models
- TerraMind, Tessera, and OlmoEarth geospatial models

The inventory includes managed companion checkpoints. Each companion's check
runs through an installed primary model that consumes the companion; the
report row names that inference. For example, Synchformer must load through
`sfx video generate`, and DreamX must complete a queued `world serve`
transition on its Wan base. If the required primary model is missing, or an
installed catalog entry has no explicit recipe, the command fails before
inference. A unit test also requires every catalog entry to have a plan, so
adding a model without its release smoke breaks CI.

These semantic checks validate the promised artifact instead of comparing a
golden hash: images and audio must decode, JSON must parse, geometry and mesh
directories must contain artifacts, MP4 files must decode, and audio-bearing
video must contain non-silent decoded audio.

On Linux, install FFmpeg with the `drawtext` filter and the DejaVu Sans font.
The OCR checks use them to render a text page before running inference.

## Usage

The following commands cover the common runs:

```bash
# First run on a machine: record baselines
mere.run gate --update-baselines

# Routine run: exits nonzero on any correctness failure
mere.run gate

# One suite, JSON report, hard perf gates
mere.run gate --suite text --strict-perf --json-output /tmp/gate.json

# Strict pre-release video contract: missing checkpoints are failures
mere.run gate --suite video --require-all --json-output /tmp/video-gate.json

# Inspect the exact installed-model release plan
mere.run gate --all-installed --list

# Complete packaged release matrix: every installed model must run
mere.run gate --all-installed --require-all \
  --json-output /tmp/release-gate.json

# Retry only an installed model that failed the full release matrix
mere.run gate --all-installed --require-all \
  --only-model text-agent-deepseek-v4-flash \
  --json-output /tmp/deepseek-rerun.json
```

## Release acceptance

For release acceptance, run the strict video command from the exact CLI
extracted from the candidate app, DMG, tarball, or package, not from the source
checkout. `--require-all` turns missing selected checkpoints into failures, so
a release host can't silently skip a required real-generation check. The three
LTX checks separately cover the standalone draft checkpoint, the full
generated-audio path, and the A2Vid compatibility model ID.

For the final packaged candidate, run `--all-installed` from the same extracted
CLI. The JSON report has one result per installed model ID. The gate maps
installed models to checks itself and fails closed if it can't account for an
entry; to audit the inventory before a long run, compare
`gate --all-installed --list` with `model list`. The exhaustive gate runs
DeepSeek V4 Flash last so its high-memory server can't reduce the memory
available to later checks.

To rerun a failed model, run the same packaged CLI with
`--only-model <id>[,<id>...]` and keep both reports. The option accepts only
installed IDs, can't be combined with `--skip-model`, and rejects selections
that `--suite` excludes. A targeted check validates its selected model even if
another installed model lacks a runnable companion. A targeted pass
supplements the original full-matrix report; it doesn't replace release-wide
coverage.

For a documented release quarantine, add `--skip-model <id>[,<id>...]`. The
model stays in the JSON report as an explicit `skipped` row and never counts as
a pass. The option is valid only with `--all-installed`, and unknown or
non-installed IDs fail closed.

## Baselines

Baselines live at `~/Library/Application Support/MereRun/gate/baselines.json`.
After a merge intentionally changes output, such as a sampler, model, or
scheduler change, refresh them with `--update-baselines`. When a hash drifts,
the failure message also tells you to refresh them.

## Run the gate nightly

For a per-machine workstation schedule, use `launchd`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>run.mere.gate</string>
  <key>ProgramArguments</key><array>
    <string>/usr/local/bin/mere.run</string>
    <string>gate</string>
    <string>--json-output</string>
    <string>/tmp/mere-gate-latest.json</string>
  </array>
  <key>StartCalendarInterval</key><dict>
    <key>Hour</key><integer>3</integer><key>Minute</key><integer>30</integer>
  </dict>
  <key>StandardOutPath</key><string>/tmp/mere-gate.log</string>
  <key>StandardErrorPath</key><string>/tmp/mere-gate.log</string>
</dict></plist>
```

Save the file as `~/Library/LaunchAgents/run.mere.gate.plist`, and then run:

```bash
launchctl load ~/Library/LaunchAgents/run.mere.gate.plist
```

To use GitHub Actions, configure a self-hosted Apple Silicon runner with the
models installed. Hosted runners don't have a GPU or a model store.

```yaml
name: quality-gate
on:
  schedule: [{cron: "30 6 * * *"}]
  workflow_dispatch:
jobs:
  gate:
    runs-on: [self-hosted, macOS, arm64]
    steps:
      - uses: actions/checkout@v7
        with: {lfs: true}
      - run: ./scripts/build_mlx_metallib.sh
      - run: swift build -c release
      - run: .build/release/mere.run gate --strict-perf --json-output gate-report.json
      - uses: actions/upload-artifact@v7
        if: always()
        with: {name: gate-report, path: gate-report.json}
```
