#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-sortformer-diarization.sh [--require-cuda]

Run native MLX Sortformer against a real A-B-A speaker fixture and verify that
the first speaker is reidentified after a different middle speaker.

Required environment:
  MERERUN_SORTFORMER_AUDIO       Real audio fixture with an A-B-A speaker order.

Optional environment:
  MERERUN_SORTFORMER_MODEL_DIR   Model id or local model directory.
                                 Default: speech-diarization-sortformer
  MERERUN_SORTFORMER_BIN         mere.run executable.
                                 Default: .build/release/mere.run
  MERERUN_SORTFORMER_OUTPUT      Preserve JSON at this path. A temporary file
                                 is used and removed when this is unset.

Options:
  --require-cuda                 Require Linux, NVIDIA visibility, and MLX GPU.
  -h, --help                     Show this help.
USAGE
}

require_cuda=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --require-cuda)
      require_cuda=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[sortformer-check] error: unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
audio="${MERERUN_SORTFORMER_AUDIO:-}"
model="${MERERUN_SORTFORMER_MODEL_DIR:-speech-diarization-sortformer}"
binary="${MERERUN_SORTFORMER_BIN:-$repo_root/.build/release/mere.run}"
output="${MERERUN_SORTFORMER_OUTPUT:-}"
temporary_output=0

if [[ -z "$audio" ]]; then
  echo "[sortformer-check] error: MERERUN_SORTFORMER_AUDIO is required." >&2
  exit 64
fi
if [[ ! -f "$audio" ]]; then
  echo "[sortformer-check] error: audio fixture not found: $audio" >&2
  exit 66
fi
if [[ ! -x "$binary" ]]; then
  echo "[sortformer-check] error: mere.run executable not found: $binary" >&2
  exit 66
fi

if [[ "$require_cuda" -eq 1 ]]; then
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "[sortformer-check] error: --require-cuda must run on Linux." >&2
    exit 65
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[sortformer-check] error: nvidia-smi is unavailable." >&2
    exit 69
  fi
  gpu_identity="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -n 1)"
  if [[ -z "$gpu_identity" ]]; then
    echo "[sortformer-check] error: no NVIDIA GPU is visible." >&2
    exit 69
  fi
  echo "[sortformer-check] NVIDIA GPU: $gpu_identity"
fi

if [[ -z "$output" ]]; then
  output="$(mktemp "${TMPDIR:-/tmp}/mere-run-sortformer-check.XXXXXX.json")"
  temporary_output=1
fi

cleanup() {
  if [[ "$temporary_output" -eq 1 ]]; then
    rm -f "$output"
  fi
}
trap cleanup EXIT

"$binary" speech diarize "$audio" \
  --model "$model" \
  --format json \
  --output "$output" \
  --quiet >/dev/null

python3 - "$output" "$require_cuda" <<'PY'
import json
import pathlib
import sys

output_path = pathlib.Path(sys.argv[1])
require_cuda = sys.argv[2] == "1"

with output_path.open("r", encoding="utf-8") as stream:
    payload = json.load(stream)

if payload.get("schema_version") != 1:
    raise SystemExit("[sortformer-check] error: expected schema_version 1")

device = payload.get("device")
if require_cuda and device != "gpu":
    raise SystemExit(
        f"[sortformer-check] error: expected MLX GPU on CUDA checkpoint, got {device!r}"
    )

segments = payload.get("segments")
if not isinstance(segments, list) or len(segments) < 3:
    raise SystemExit("[sortformer-check] error: expected at least three timed segments")

speakers = []
for index, segment in enumerate(segments):
    speaker = segment.get("speaker")
    start = segment.get("start_seconds")
    end = segment.get("end_seconds")
    if not isinstance(speaker, str) or not speaker:
        raise SystemExit(f"[sortformer-check] error: segment {index} has no speaker")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end <= start:
        raise SystemExit(f"[sortformer-check] error: segment {index} has invalid timing")
    if not speakers or speaker != speakers[-1]:
        speakers.append(speaker)

unique_speakers = set(speakers)
if len(unique_speakers) < 2:
    raise SystemExit("[sortformer-check] error: expected at least two distinct speakers")
if len(speakers) < 3 or speakers[0] != speakers[-1]:
    raise SystemExit(
        "[sortformer-check] error: first speaker was not reidentified after a speaker change"
    )
if not any(speaker != speakers[0] for speaker in speakers[1:-1]):
    raise SystemExit("[sortformer-check] error: no different middle speaker was detected")
if payload.get("speaker_count") != len({segment["speaker"] for segment in segments}):
    raise SystemExit("[sortformer-check] error: speaker_count does not match segment labels")

runtime = payload.get("runtime", "unknown")
print(
    "[sortformer-check] PASS: "
    f"device={device} speakers={payload['speaker_count']} "
    f"collapsed_sequence={' -> '.join(speakers)} runtime={runtime}"
)
PY

if [[ "$temporary_output" -eq 0 ]]; then
  echo "[sortformer-check] result: $output"
fi
