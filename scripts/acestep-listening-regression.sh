#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture="${1:-${repo_root}/Tests/MereRunCLITests/Fixtures/ACEStep/listening-regression.json}"
output_root="${2:-${repo_root}/.build/acestep-listening-regression}"
cli="${MERERUN_BIN:-${repo_root}/.build/debug/mere.run}"

if [[ ! -x "${cli}" ]]; then
  swift build --package-path "${repo_root}"
fi

mkdir -p "${output_root}"

python3 - "${fixture}" "${output_root}" "${cli}" <<'PY'
import csv
import hashlib
import json
import pathlib
import subprocess
import sys

fixture_path = pathlib.Path(sys.argv[1]).resolve()
output_root = pathlib.Path(sys.argv[2]).resolve()
cli = pathlib.Path(sys.argv[3]).resolve()
fixture = json.loads(fixture_path.read_text())
model = fixture["model"]
duration = fixture["duration_seconds"]
quality = fixture["quality"]
candidates = fixture["candidates"]
manifest = {
    "schema_version": 1,
    "fixture": str(fixture_path),
    "model": model,
    "cases": [],
}
playlist = ["#EXTM3U"]
review_rows = []

for case in fixture["cases"]:
    case_id = case["id"]
    output = output_root / f"{case_id}.wav"
    command = [
        str(cli),
        "music",
        "generate",
        case["caption"],
        "--model",
        model,
        "--duration",
        str(duration),
        "--quality",
        quality,
        "--candidates",
        str(candidates),
        "--seed",
        str(case["seed"]),
        "--output",
        str(output),
    ]
    if case.get("lyrics"):
        command.extend(["--lyrics", case["lyrics"]])
    subprocess.run(command, check=True)
    recipe = output.with_suffix(".recipe.json")
    recipe_payload = json.loads(recipe.read_text())
    selected_candidate = next(
        candidate
        for candidate in recipe_payload["candidates"]
        if candidate["selected"]
    )
    metrics = selected_candidate["metrics"]
    temporal_variation = metrics.get("temporalSpectralVariation", 0)
    tail_energy_ratio = metrics.get("tailEnergyRatio", 0)
    if temporal_variation < 0.85:
        raise RuntimeError(
            f"{case_id} failed musical-structure gate: "
            f"temporalSpectralVariation={temporal_variation:.3f} < 0.850"
        )
    if tail_energy_ratio < 0.05:
        raise RuntimeError(
            f"{case_id} failed tail-continuity gate: "
            f"tailEnergyRatio={tail_energy_ratio:.3f} < 0.050"
        )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest["cases"].append(
        {
            "id": case_id,
            "audio": output.name,
            "recipe": recipe.name,
            "sha256": digest,
            "candidate_score": selected_candidate["score"],
            "candidate_metrics": metrics,
            "listen_for": case["listen_for"],
        }
    )
    playlist.append(output.name)
    review_rows.append(
        [
            case_id,
            case["listen_for"],
            "",
            "",
            "",
            "",
            "",
            "",
        ]
    )

(output_root / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
(output_root / "playlist.m3u8").write_text("\n".join(playlist) + "\n")
with (output_root / "review.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "case",
            "listen_for",
            "structure_1_to_5",
            "prompt_adherence_1_to_5",
            "audio_quality_1_to_5",
            "vocal_alignment_1_to_5",
            "regression_yes_no",
            "notes",
        ]
    )
    writer.writerows(review_rows)
PY

printf '%s\n' "${output_root}/playlist.m3u8"
