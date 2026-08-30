#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

matches_file="$(mktemp "${TMPDIR:-/tmp}/mere-run-doc-examples.XXXXXX")"
trap 'rm -f "$matches_file"' EXIT

scan_targets=(
  README.md
  CONTRIBUTING.md
  CODEBASE.md
  docs
)

email_pattern='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
rg --no-heading --with-filename --line-number --only-matching \
  "$email_pattern" "${scan_targets[@]}" >"$matches_file" || true

unexpected=()
while IFS=: read -r path line email; do
  [[ -n "${email:-}" ]] || continue
  domain="${email##*@}"
  domain="$(printf '%s' "$domain" | tr '[:upper:]' '[:lower:]')"

  case "$domain" in
    example.com|*.example.com|example.net|*.example.net|example.org|*.example.org)
      ;;
    *.example|*.test|*.invalid)
      ;;
    mere.run|*.mere.run)
      ;;
    *)
      unexpected+=("$path:$line:$email")
      ;;
  esac
done <"$matches_file"

if (( ${#unexpected[@]} > 0 )); then
  printf '%s\n' "Documentation contains email domains that are not reserved examples:" >&2
  printf '  %s\n' "${unexpected[@]}" >&2
  printf '%s\n' \
    "Use example.com, example.net, example.org, a special-use example TLD," \
    "or an intentional project-owned mere.run address." >&2
  exit 1
fi

if rg --line-number --ignore-case \
  '\[(click here|here|this link|link)\]\(' "${scan_targets[@]}"
then
  printf '%s\n' \
    "Documentation contains vague link text. Describe the link destination." >&2
  exit 1
fi

if rg --line-number --ignore-case \
  -e '^## +release status[[:space:]]*$' \
  -e "^#{2,3} +what'?s new([[:space:]]|$)" \
  -e '^#{2,3} +release +v?[0-9]+\.[0-9]+\.[0-9]+([[:space:]]|$)' \
  -e '\brelease +v?[0-9]+\.[0-9]+\.[0-9]+\b' \
  -e 'github\.com/sawfwair/mere-run/releases/tag/v[0-9]+\.[0-9]+\.[0-9]+' \
  README.md
then
  printf '%s\n' \
    "README.md contains version-specific release history." \
    "Keep README.md evergreen and put release chronology, qualification" \
    "results, and version-specific changes in CHANGELOG.md and GitHub releases." >&2
  exit 1
fi

if ! rg --quiet '\]\(\./CHANGELOG\.md\)' README.md || \
   ! rg --quiet 'github\.com/sawfwair/mere-run/releases' README.md
then
  printf '%s\n' \
    "README.md must link to CHANGELOG.md and the GitHub releases page." >&2
  exit 1
fi

printf '%s\n' "Documentation example-data and link-text checks passed."
