#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

allowed_fixture="Tests/MereRunEvaluationTests/Fixtures/SyntheticEvaluationPack/eval-pack.json"
unexpected_manifests=()
while IFS= read -r tracked_manifest; do
  if [[ "$tracked_manifest" != "$allowed_fixture" ]]; then
    unexpected_manifests+=("$tracked_manifest")
  fi
done < <(
  git ls-files --cached --others --exclude-standard -- '*eval-pack.json' \
    | sort -u
)

if (( ${#unexpected_manifests[@]} > 0 )); then
  printf '%s\n' "${unexpected_manifests[@]}" >&2
  cat >&2 <<'EOF'
External evaluation-pack manifests must remain in their owning repositories.
Only the repository's synthetic contract fixture may be tracked here.
EOF
  exit 1
fi

for forbidden_directory in \
  Sources/MereRunEvaluation/Packs \
  Sources/MereRunEvaluation/Fixtures \
  Sources/MereRunCLI/EvaluationPacks
do
  [[ -d "$forbidden_directory" ]] || continue
  if git ls-files --cached --others --exclude-standard -- "$forbidden_directory/**" \
    | rg -q '.'; then
    echo "Evaluation pack content cannot be bundled under $forbidden_directory." >&2
    exit 1
  fi
done

markers_file="${MERERUN_PROPRIETARY_MARKERS_FILE:-}"
if [[ -z "$markers_file" ]]; then
  exit 0
fi
if [[ ! -f "$markers_file" ]]; then
  echo "MERERUN_PROPRIETARY_MARKERS_FILE does not name a readable file: $markers_file" >&2
  exit 1
fi

boundary_paths=(
  Sources/MereRunEvaluation
  Sources/MereRunCLI/Commands/EvaluationCommand.swift
  Sources/MereRunCLI/Support/EvaluationRunTypes.swift
  Sources/MereRunCLI/Support/EvaluationScoring.swift
  Tests/MereRunEvaluationTests
  Tests/MereRunCLITests/EvaluationCommandTests.swift
  docs/evaluation-packs.md
)

found_marker=0
while IFS= read -r marker || [[ -n "$marker" ]]; do
  case "$marker" in
    ''|'#'*) continue ;;
  esac
  if rg -n -i -F -- "$marker" "${boundary_paths[@]}"; then
    echo "Private marker found in the public evaluation surface." >&2
    found_marker=1
  fi
done < "$markers_file"

if [[ "$found_marker" == "1" ]]; then
  exit 1
fi
