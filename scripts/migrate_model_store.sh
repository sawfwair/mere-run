#!/usr/bin/env bash
set -euo pipefail

SOURCE_MODELS_ROOT=""
DESTINATION_MODELS_ROOT="${MERERUN_MODELS_DIR:-$HOME/Library/Application Support/MereRun/models}"
DRY_RUN=0

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --source-models-root)
      SOURCE_MODELS_ROOT="$2"
      shift 2
      ;;
    --destination-models-root|--models-root)
      DESTINATION_MODELS_ROOT="$2"
      shift 2
      ;;
    *)
      echo "usage: $0 [--dry-run] [--source-models-root /old/path] [--destination-models-root /new/path]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SOURCE_MODELS_ROOT" ]]; then
  echo "usage: $0 [--dry-run] --source-models-root /legacy/path [--destination-models-root /new/path]" >&2
  exit 1
fi

python3 - "$SOURCE_MODELS_ROOT" "$DESTINATION_MODELS_ROOT" "$DRY_RUN" <<'PY'
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).expanduser().resolve()
destination_root = Path(sys.argv[2]).expanduser().resolve()
dry_run = sys.argv[3] == "1"
models_root = destination_root if not dry_run else source_root
legacy_klein_prefix = "".join(["ze", "ro"])
legacy_zimage_prefix = "".join(["ze", "ta"])
legacy_tts_prefix = "-".join(["talk", "nano"])
legacy_parakeet_id = "-".join(["asr", "parakeet"])
legacy_video_id = "-".join(["ltx", "video", "av"])
legacy_manifest_name = "_".join([legacy_klein_prefix, "model"]) + ".json"
current_manifest_name = "mererun_model.json"

top_level_map = {
    **{f"{legacy_klein_prefix}-{tier}": f"image-klein-{tier}" for tier in ("nano", "base", "max")},
    **{f"{legacy_zimage_prefix}-{tier}": f"image-zimage-{tier}" for tier in ("nano", "base", "max")},
    "shared-klein": "image-klein-shared",
    "q35": "text-chat-q35",
    "q35-nano": "text-chat-q35-nano",
    "mebot": "text-chat-mebot",
    "psi-agent": "text-chat-psi-agent",
    "qwen3-coder": "text-code-qwen3",
    "qwen3-embedding-0.6b": "text-embed-qwen3-0.6b",
    legacy_tts_prefix: "speech-tts-qwen3-nano",
    f"{legacy_tts_prefix}-customvoice": "speech-tts-qwen3-customvoice",
    "asr": "speech-asr-qwen3",
    legacy_parakeet_id: "speech-asr-parakeet",
    "ocr": "vision-ocr-lighton",
    "acestep": "music-acestep",
    legacy_video_id: "video-ltx-av",
}

family_map = {
    legacy_klein_prefix: "klein",
    legacy_zimage_prefix: "zimage",
}

music_nested_map = {
    "acestep-v15-turbo": "music-acestep-v15-turbo",
    "acestep-5Hz-lm-1.7B": "music-acestep-5hz-lm-1.7b",
    "acestep-5hz-lm-1.7b": "music-acestep-5hz-lm-1.7b",
    "acestep-5Hz-lm": "music-acestep-5hz-lm",
    "acestep-5hz-lm": "music-acestep-5hz-lm",
    "acestep-lm": "music-acestep-lm",
}

klein_shared_components = {
    "tokenizer": {
        "type": "local",
        "path": "tokenizer",
    },
    "text_encoder": {
        "type": "local",
        "path": "text_encoder",
    },
    "vae": {
        "type": "local",
        "path": "vae",
    },
    "scheduler": {
        "type": "local",
        "path": "scheduler",
    },
}

klein_hybrid_components = {
    "tokenizer": {
        "type": "anyOf",
        "candidates": [
            {"type": "local", "path": "tokenizer"},
            {"type": "model", "modelId": "image-klein-shared", "path": "tokenizer"},
        ],
    },
    "text_encoder": {
        "type": "anyOf",
        "candidates": [
            {"type": "local", "path": "text_encoder"},
            {"type": "model", "modelId": "image-klein-shared", "path": "text_encoder"},
        ],
    },
    "transformer": {
        "type": "local",
        "path": "transformer",
    },
    "vae": {
        "type": "anyOf",
        "candidates": [
            {"type": "local", "path": "vae"},
            {"type": "model", "modelId": "image-klein-shared", "path": "vae"},
        ],
    },
    "scheduler": {
        "type": "anyOf",
        "candidates": [
            {"type": "local", "path": "scheduler"},
            {"type": "model", "modelId": "image-klein-shared", "path": "scheduler"},
        ],
    },
}

if not source_root.exists():
    raise SystemExit(f"source models root not found: {source_root}")

if source_root != destination_root:
    if destination_root.exists():
        raise SystemExit(
            f"refusing to overwrite existing destination models root:\n"
            f"  source: {source_root}\n"
            f"  destination: {destination_root}"
        )
    print(f"[move-root] {source_root} -> {destination_root}")
    if not dry_run:
        destination_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_root), str(destination_root))
        models_root = destination_root

def move_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        raise SystemExit(f"refusing to overwrite existing path:\n  source: {src}\n  destination: {dst}")
    print(f"[rename] {src.name} -> {dst.name}")
    if not dry_run:
        src.rename(dst)

for old, new in top_level_map.items():
    move_dir(models_root / old, models_root / new)

music_root = models_root / "music-acestep"
for parent in [music_root, music_root / "checkpoints"]:
    if not parent.exists():
        continue
    for old, new in music_nested_map.items():
        move_dir(parent / old, parent / new)

shared_root = models_root / "image-klein-shared"
shared_manifest_path = shared_root / current_manifest_name
if shared_root.exists() and not shared_manifest_path.exists():
    shared_manifest = {
        "components": klein_shared_components,
        "engine": "flux2-klein",
        "family": "klein",
        "id": "image-klein-shared",
        "precision": "unknown",
        "schemaVersion": 2,
        "supports": ["txt2img", "reference_edit", "lora_inference"],
    }
    print(f"[write] {shared_manifest_path}")
    if not dry_run:
        shared_manifest_path.write_text(json.dumps(shared_manifest, indent=2, sort_keys=True) + "\n")

manifest_paths = []
manifest_paths.extend(models_root.glob(f"*/{legacy_manifest_name}"))
manifest_paths.extend(models_root.glob(f"*/{current_manifest_name}"))

for manifest_path in manifest_paths:
    if manifest_path.name == legacy_manifest_name:
        renamed_path = manifest_path.with_name(current_manifest_name)
        if renamed_path.exists():
            raise SystemExit(
                f"refusing to overwrite existing manifest path:\n  source: {manifest_path}\n  destination: {renamed_path}"
            )
        print(f"[rename] {manifest_path.name} -> {renamed_path.name}")
        if not dry_run:
            manifest_path.rename(renamed_path)
        manifest_path = renamed_path

    data = json.loads(manifest_path.read_text())
    changed = False

    old_id = data.get("id")
    if old_id in top_level_map:
        data["id"] = top_level_map[old_id]
        changed = True

    family = data.get("family")
    if family in family_map:
        data["family"] = family_map[family]
        changed = True

    components = data.get("components")
    if isinstance(components, dict):
        for component in components.values():
            if isinstance(component, dict) and component.get("type") == "model":
                model_id = component.get("modelId")
                if model_id in top_level_map:
                    component["modelId"] = top_level_map[model_id]
                    changed = True

    if manifest_path.parent.name in {"image-klein-base", "image-klein-max"} and shared_root.exists():
        if data.get("components") != klein_hybrid_components:
            data["components"] = klein_hybrid_components
            changed = True

    if changed:
        print(f"[rewrite] {manifest_path}")
        if not dry_run:
            manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

expected_files = {
    "image-zimage-max": [current_manifest_name],
    "image-klein-max": [current_manifest_name],
    "image-klein-nano": [current_manifest_name],
    "text-chat-q35": [current_manifest_name],
    "text-chat-q35-nano": [current_manifest_name],
}

for model_id, required in expected_files.items():
    root = models_root / model_id
    if not root.exists():
        continue
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise SystemExit(f"missing required files after migration for {model_id}: {', '.join(missing)}")

for model_id in ("image-klein-base", "image-klein-max"):
    root = models_root / model_id
    if not root.exists():
        continue

    has_local_shared_parts = all((root / name).is_dir() for name in ("tokenizer", "text_encoder", "vae", "scheduler"))
    if has_local_shared_parts:
        continue

    if not shared_root.exists():
        raise SystemExit(
            f"{model_id} is missing shared components locally and image-klein-shared is not installed."
        )

    missing_shared = [
        name for name in ("tokenizer", "text_encoder", "vae", "scheduler")
        if not (shared_root / name).is_dir()
    ]
    if missing_shared:
        raise SystemExit(
            f"image-klein-shared is missing required directories for {model_id}: {', '.join(missing_shared)}"
        )

print(f"[done] destination models root: {destination_root}")
if dry_run:
    print("[done] dry run only; no files were changed")
PY
