#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:-${repo_root}/.build/acestep-performance-proof}"
model="${MERERUN_ACESTEP_PERF_MODEL:-music-acestep-xl-turbo-lm4b}"
cli="${MERERUN_BIN:-${repo_root}/.build/debug/mere.run}"

if [[ ! -x "${cli}" ]]; then
  swift build --package-path "${repo_root}"
fi

mkdir -p "${output_root}"
result="${output_root}/one-second-one-step.wav"
timing="${output_root}/time.txt"

/usr/bin/time -lp -o "${timing}" \
  "${cli}" music generate \
  "tight electronic drum groove, instrumental" \
  --model "${model}" \
  --duration 1 \
  --steps 1 \
  --candidates 1 \
  --no-lm \
  --seed 4242 \
  --output "${result}"

shasum -a 256 "${result}" > "${output_root}/sha256.txt"
printf '%s\n' "${output_root}"
