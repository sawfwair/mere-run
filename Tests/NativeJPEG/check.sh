#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
codec="$repo_root/Sources/CMereRunJPEG"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/mererun-jpeg.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT
clang -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I "$codec/include" -I "$codec/libjpeg" -I "$codec/libjpeg/src" \
  "$codec/decode.c" "$codec"/compile/*.c \
  "$repo_root/Tests/NativeJPEG/decoder.c" -o "$scratch/decoder"
"$scratch/decoder" \
  "$repo_root"/Tests/MereRunCoreTests/Fixtures/PillowJPEG/*.jpeg
