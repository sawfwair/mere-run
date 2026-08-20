#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" == "Linux" ]]; then
  exec bash "$repo_root/scripts/check-linux.sh" "$@"
fi

for homebrew_bin in /opt/homebrew/bin /usr/local/bin; do
  if [[ -d "$homebrew_bin" && ":$PATH:" != *":$homebrew_bin:"* ]]; then
    PATH="$homebrew_bin:$PATH"
  fi
done
export PATH

missing_tools=()
for required_tool in swiftlint rg; do
  if ! command -v "$required_tool" >/dev/null 2>&1; then
    missing_tools+=("$required_tool")
  fi
done

if (( ${#missing_tools[@]} > 0 )); then
  cat >&2 <<'EOF'
./scripts/check.sh requires SwiftLint and ripgrep, but one or more required
tools were not found in PATH.

Install them once with:
  brew install swiftlint ripgrep
EOF
  printf 'Missing: %s\n' "${missing_tools[*]}" >&2
  exit 127
fi

swiftlint --strict --cache-path .build/swiftlint.cache
bash ./scripts/agent_readiness_check.sh
bash ./scripts/check-evaluation-boundary.sh
bash ./scripts/check-docs-examples.sh
swiftpm() {
  local subcommand="$1"
  shift
  if [[ "${MERERUN_SWIFT_DISABLE_INDEX_STORE:-0}" == "1" ]]; then
    swift "$subcommand" --disable-index-store "$@"
  else
    swift "$subcommand" "$@"
  fi
}

swiftpm build
./scripts/build_mlx_metallib.sh \
  --verify-only vendor/mlx-swift_Cmlx.bundle
swiftpm test
mere_run_bin=".build/debug/mere.run"
if [[ ! -x "$mere_run_bin" ]]; then
  echo "Expected built executable at $mere_run_bin after swift build." >&2
  exit 1
fi

"$mere_run_bin" --help >/dev/null
"$mere_run_bin" guide --help >/dev/null
"$mere_run_bin" image generate --help >/dev/null
"$mere_run_bin" image validate --help >/dev/null
"$mere_run_bin" text chat --help >/dev/null
"$mere_run_bin" text code --help >/dev/null
"$mere_run_bin" text embed --help >/dev/null
"$mere_run_bin" speech synthesize --help >/dev/null
"$mere_run_bin" speech transcribe --help >/dev/null
"$mere_run_bin" speech diarize --help >/dev/null
"$mere_run_bin" speech profile --help >/dev/null
"$mere_run_bin" vision inspect --help >/dev/null
"$mere_run_bin" vision ocr --help >/dev/null
"$mere_run_bin" music analyze --help >/dev/null
"$mere_run_bin" music generate --help >/dev/null
"$mere_run_bin" video generate --help >/dev/null
"$mere_run_bin" video animate --help >/dev/null
"$mere_run_bin" video prepare-masks --help >/dev/null
"$mere_run_bin" video export-latents --help >/dev/null
"$mere_run_bin" model --help >/dev/null
"$mere_run_bin" model storage --help >/dev/null
"$mere_run_bin" model gc --help >/dev/null
"$mere_run_bin" adapter --help >/dev/null
"$mere_run_bin" adapter list --help >/dev/null
"$mere_run_bin" adapter pull --help >/dev/null
"$mere_run_bin" model runtime --help >/dev/null
"$mere_run_bin" model runtime get --help >/dev/null
"$mere_run_bin" model runtime set --help >/dev/null
"$mere_run_bin" eval --help >/dev/null
"$mere_run_bin" eval pack validate --help >/dev/null
"$mere_run_bin" eval run --help >/dev/null
"$mere_run_bin" eval promote --help >/dev/null
"$mere_run_bin" status --help >/dev/null
"$mere_run_bin" api serve --help >/dev/null

model_list_output="$("$mere_run_bin" model list)"
rg -q '^ID +Category +Status +Referenced$' <<<"$model_list_output"
rg -q '^image-klein-max +image +' <<<"$model_list_output"

status_output="$("$mere_run_bin" status --timeout-seconds 0.1)"
rg -q '^mere\.run status$' <<<"$status_output"
rg -q '^  server: ' <<<"$status_output"
rg -q '^  model store: ' <<<"$status_output"
rg -q '^  installed models: ' <<<"$status_output"

if rg -n \
  -e 'print\("\[(Flux2KleinGenerator|ZImageTurboGenerator|QwenTextLoRAApplicator)\]' \
  Sources/MereRunCore
then
  echo "Default runtime paths still contain unconditional generator debug prints." >&2
  exit 1
fi

if rg -n -i \
  -e 'import +PythonKit' \
  -e 'maestro' \
  -e 'wangp' \
  -e 'conversion[_ -]?(script|utility)' \
  Sources/MereRunCore/SCAIL2 Sources/MereRunCLI/Commands/VideoAnimateCommand.swift Sources/MereRunCLI/Commands/VideoPrepareMasksCommand.swift
then
  echo "Native SCAIL-2 sources contain a prohibited runtime or provenance dependency." >&2
  exit 1
fi

if rg -n -i -g '*.swift' \
  -e 'python' \
  -e 'Process *\(' \
  Sources/MereRunCore/SCAIL2 Sources/MereRunCLI/Commands/VideoAnimateCommand.swift Sources/MereRunCLI/Commands/VideoPrepareMasksCommand.swift
then
  echo "Native SCAIL-2 Swift sources contain a prohibited sidecar runtime hook." >&2
  exit 1
fi

if [[ "${MERERUN_RUN_E2E:-}" == "core" ]]; then
  ./scripts/e2e_smoke.sh --core
elif [[ "${MERERUN_RUN_E2E:-}" == "installed" ]]; then
  ./scripts/e2e_smoke.sh --installed
fi
