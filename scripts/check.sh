#!/usr/bin/env bash
set -euo pipefail

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
printf '%s\n' "$model_list_output" | rg -q '^ID +Category +Status +Size$'
printf '%s\n' "$model_list_output" | rg -q '^image-klein-max +image +'

if rg -n \
  -e 'public\.stereovoid\.com|MERERUN_R2_PUBLIC_BASE_URL|R2_PUBLIC_BASE_URL' \
  README.md docs Sources/MereRunCore Sources/MereRunCLI scripts/e2e_smoke.sh
then
  echo "Primary runtime/docs still contain private hosted defaults." >&2
  exit 1
fi

if rg -n --hidden \
  -g '!docs/.vitepress/dist/**' \
  -g '!docs/.vitepress/cache/**' \
  -e '\b(ZeroCore|ZeroCLI|ZeroModelPaths|ZeroModelManifest|ZeroModelValidator|ZeroRuntimeDebug)\b|Application Support/Zero|\bZERO_[A-Z0-9_]+\b|\bzero-oss\b|swift run zero\b|(^|[^[:alnum:]_.-])zero ([a-z<])|(^|[^[:alnum:]_.-])Zero([^[:alnum:]_.-]|$)' \
  README.md CODEBASE.md DECISIONS.md package.json docs docs/.vitepress Sources/MereRunCLI Tests/MereRunCLITests scripts/e2e_smoke.sh scripts/migrate_model_store.sh
then
  echo "Primary docs/scripts/CLI still contain pre-rename product identity." >&2
  exit 1
fi

if rg -n --hidden \
  -g '!docs/.vitepress/dist/**' \
  -g '!docs/.vitepress/cache/**' \
  -e 'swift run MereRunCLI\b|`(listen|look|talk-profile|ltxvideo|acestep)(?: [^`]*)?`|(^|[^[:alnum:]-])(zero-(nano|max|base)|zeta-(nano|max|base)|talk-nano(?:-customvoice)?|asr-parakeet|ltx-video-av)([^[:alnum:]-]|$)' \
  README.md docs docs/.vitepress Sources/MereRunCLI Tests/MereRunCLITests scripts/e2e_smoke.sh scripts/migrate_model_store.sh
then
  echo "Public docs/scripts still contain pre-rename CLI or model vocabulary." >&2
  exit 1
fi

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
