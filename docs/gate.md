# Quality gate

`mere.run gate` is the platform's flight recorder: it runs the real
user-facing commands against the installed models at temperature 0, hashes
their outputs, proves in-run determinism, and compares everything against
machine-local baselines. It exists because a stale Metal library once shipped
broken ≥1024-token decode to every clone for two months without anything
noticing — the gate's text checks deliberately include a long-context lane in
exactly that regime, and its determinism check would have caught the failure
on the first nightly run.

## What it checks

| check | what it runs | hard-fails on |
| --- | --- | --- |
| `text-*-short` / `text-*-long` | `text chat` at temp 0 (Gemma 4 12B, Ornith 35B, LFM2.5), long lane ≥1200 prompt tokens, each executed twice | run-to-run nondeterminism, empty output, output hash drift vs baseline |
| `tts-wav` | `speech synthesize` at temp 0 | WAV hash drift, truncated audio |
| `stt-roundtrip` | `speech transcribe` on the TTS output, twice | transcript nondeterminism, <85% word recovery of the spoken sentence |
| `ocr-page` | `vision ocr` on a deterministically rendered text page, twice | transcript nondeterminism, <90% word recovery |
| `image-klein-seed7` | seeded `image generate` | PNG hash drift, truncated file |
| `embeddings` | `text embed`, twice | vector bytes nondeterministic or drifted |

Performance (wall time, `decode_tps` where the engine reports it) is compared
against the baseline too: regressions beyond 0.75× tok/s or 1.5× wall are
warnings by default and failures with `--strict-perf` (suitable for a
dedicated runner; on a shared workstation, background load makes hard perf
gates noisy).

Checks whose models are not installed are skipped and reported. Every check
shells out to the same `mere.run` binary that users run — the gate covers the
CLI surface, model resolution, and the runtime together, not a parallel test
path.

## Usage

```bash
# First run on a machine: record baselines
mere.run gate --update-baselines

# Routine run: exits nonzero on any correctness failure
mere.run gate

# One suite, JSON report, hard perf gates
mere.run gate --suite text --strict-perf --json-output /tmp/gate.json
```

Baselines live at `~/Library/Application Support/MereRun/gate/baselines.json`.
After an intentionally output-changing merge (sampler changes, model updates,
scheduler changes), refresh them with `--update-baselines` — the failure
message says so when a hash drifts.

## Running it nightly

launchd (per-machine, recommended for workstations):

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

Save as `~/Library/LaunchAgents/run.mere.gate.plist`, then
`launchctl load ~/Library/LaunchAgents/run.mere.gate.plist`.

GitHub Actions (requires a self-hosted Apple-silicon runner with models
installed; the hosted runners have no GPU or model store):

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
