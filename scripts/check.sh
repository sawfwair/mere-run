#!/usr/bin/env bash
set -euo pipefail

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
swift build
swift test
swift run mere.run --help >/dev/null
swift run mere.run image generate --help >/dev/null
swift run mere.run image validate --help >/dev/null
swift run mere.run text chat --help >/dev/null
swift run mere.run text code --help >/dev/null
swift run mere.run text embed --help >/dev/null
swift run mere.run speech synthesize --help >/dev/null
swift run mere.run speech transcribe --help >/dev/null
swift run mere.run speech profile --help >/dev/null
swift run mere.run vision inspect --help >/dev/null
swift run mere.run vision ocr --help >/dev/null
swift run mere.run music generate --help >/dev/null
swift run mere.run video generate --help >/dev/null
swift run mere.run video export-latents --help >/dev/null
swift run mere.run model --help >/dev/null
swift run mere.run api serve --help >/dev/null

model_list_output="$(swift run mere.run model list)"
rg -q '^ID +Category +Status +Size$' <<<"$model_list_output"
rg -q '^image-klein-max +image +' <<<"$model_list_output"

if rg -n \
  -e 'print\("\[(Flux2KleinGenerator|ZImageTurboGenerator|QwenTextLoRAApplicator)\]' \
  Sources/MereRunCore
then
  echo "Default runtime paths still contain unconditional generator debug prints." >&2
  exit 1
fi

if [[ "${MERERUN_RUN_E2E:-}" == "core" ]]; then
  ./scripts/e2e_smoke.sh --core
elif [[ "${MERERUN_RUN_E2E:-}" == "installed" ]]; then
  ./scripts/e2e_smoke.sh --installed
fi
