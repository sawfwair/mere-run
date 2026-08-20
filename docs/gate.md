# Quality gate

Use `mere.run gate` as the platform's flight recorder. It runs the real
user-facing commands against the installed models at temperature 0, hashes
their outputs, proves in-run determinism, and compares everything against
machine-local baselines. The gate exists because a stale Metal library once
shipped broken decode for 1,024 or more tokens to every clone for two months
without detection. Its text checks include a long-context lane in that regime,
and its determinism check would have caught the failure during the first nightly
run.

## What it checks

| Check | Operation | Failure conditions |
| --- | --- | --- |
| `text-*-short` / `text-*-long` | `text chat` at temp 0 (Gemma 4 12B, Ornith 35B, LFM2.5), long lane ≥1200 prompt tokens, each executed twice | run-to-run nondeterminism, empty output, output hash drift vs baseline |
| `tts-wav` | `speech synthesize` at temp 0 | WAV hash drift, truncated audio |
| `stt-roundtrip` | `speech transcribe` on the TTS output, twice | transcript nondeterminism, <85% word recovery of the spoken sentence |
| `ocr-page` | `vision ocr` on a deterministically rendered text page, twice | transcript nondeterminism, <90% word recovery |
| `image-klein-seed7` | seeded `image generate` | PNG hash drift, truncated file |
| `embeddings` | `text embed`, twice | vector bytes nondeterministic or drifted |
| `video-ltx23-draft` | 9-frame native LTX 2.3 distilled render | generation failure or MP4 decode failure |
| `video-ltx23-full-av` | 9-frame full two-stage LTX 2.3 render with generated audio | generation failure, MP4 decode failure, missing audio, or silent audio |
| `video-ltx23-a2vid` | 9-frame full two-stage LTX 2.3 render conditioned on a generated non-silent WAV fixture | generation failure, MP4 decode failure, missing audio, or silent audio |

The gate also compares performance, including wall time and `decode_tps` when
the engine reports it. Regressions beyond 0.75 times the tokens per second or
1.5 times the wall time are
warnings by default and failures with `--strict-perf` (suitable for a
dedicated runner). On a shared workstation, background load makes strict
performance gates noisy.

If a model is not installed, the gate skips and reports its checks. Every check
invokes the same `mere.run` binary that users run. The gate covers the
CLI surface, model resolution, and the runtime together, not a parallel test
path.

The video checks are semantic release smokes rather than golden-output checks.
They deliberately use minimum valid geometry and frame counts, but they still
load the real checkpoints, denoise, write MP4, decode a frame, and decode audio
when the route promises audio. Their hashes are recorded in the report without
being compared to machine-local baselines.

## Exhaustive installed-model release smoke

`--all-installed` is a separate fail-closed release mode. It reads the same
inventory as `mere.run model list`, creates one named check for every entry
whose status is `installed`, and runs the exact user-facing operation for that
model:

- every text, code, vision-chat, embedding, privacy, TTS, and ASR model
- every image model
- LightOn and Infinity OCR, SAM segmentation, grounding, face detection,
  MoGe geometry, VDA depth, and DA3 multi-view geometry
- TripoSR, InstantMesh, and TRELLIS.2 image-to-3D
- every ACE-Step and Magenta generator plus every installed MuScriptor model
- every Woosh/MMAudio generator, CLAP, and Synchformer
- every installed LTX, Wan, Cosmos3, SCAIL-2, and DreamX video/world model

The gate does not accept component-only entries as a family. Their report row
names the true companion inference that consumes them. For example,
Synchformer must be loaded by `sfx video generate`; DreamX must complete a
queued `world serve` transition using its Wan base. If the required companion
is missing, or an installed catalog entry has no explicit recipe, the command
fails before inference. A unit contract also requires every catalog entry to
have a plan, so adding a model without adding its release smoke breaks CI.

The semantic checks validate the promised artifact rather than comparing a
golden hash: images and audio must decode, JSON must parse, geometry and mesh
directories must contain artifacts, MP4s must decode, and audio-bearing video
must contain non-silent decoded audio.

## Usage

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
```

For release acceptance, run the strict video command from the exact extracted
CLI inside the candidate app, DMG, tarball, or package—not from the source
checkout. `--require-all` converts missing selected checkpoints into failures,
so a release host cannot silently turn a required real-generation check into a
skip. The three LTX checks cover the standalone draft checkpoint, the full
generated-audio path, and the compatibility A2Vid model ID separately.

For the final packaged candidate, run `--all-installed` from the exact
extracted CLI. The JSON report has one result per installed model ID. Compare
`gate --all-installed --list` with `model list` before starting a long run if
you need an inventory audit. The gate performs that mapping itself and
fails closed if it cannot account for an installed entry.

A documented exceptional release quarantine can add
`--skip-model <id>[,<id>...]`. The model remains in the JSON report as an
explicit `skipped` row; it is never counted as a pass. The option is valid only
with `--all-installed`, and unknown or non-installed IDs fail closed.

Baselines live at `~/Library/Application Support/MereRun/gate/baselines.json`.
After a merge intentionally changes output, such as a sampler, model, or
scheduler change, refresh the baselines with `--update-baselines`. The failure
message also directs you to refresh them when a hash drifts.

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

To use GitHub Actions, configure a self-hosted Apple silicon runner with the
models installed. Hosted runners do not have a GPU or model store.

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
