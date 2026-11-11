#!/usr/bin/env python3
"""Measure one real native H3-Image run and preserve logs, progress, and output identity.

The inference process is the supplied release binary. Python only records
measurements and inspects the completed PNG. Requires Pillow and numpy.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    image = output / (args.case + ".png")
    record_path = output / (args.case + ".json")
    stdout_path = output / (args.case + ".stdout.log")
    stderr_path = output / (args.case + ".stderr.log")
    if any(path.exists() for path in [image, record_path, stdout_path, stderr_path]):
        raise SystemExit("Choose a fresh case ID; existing evidence is never overwritten.")
    binary = args.binary.resolve()
    command = [str(binary), "image", "generate", "--model", str(args.model.resolve()),
               "--prompt", args.prompt, "--width", str(args.size), "--height", str(args.size),
               "--steps", "6", "--cfg", "1", "--seed", str(args.seed), "--output", str(image),
               "--run-dir", str(output / (args.case + ".run")), "--progress-json", "--receipt"]
    preflight = subprocess.run(command[:-2] + ["--preflight", "--json"], capture_output=True, text=True)
    (output / (args.case + ".preflight.json")).write_text(preflight.stdout)
    (output / (args.case + ".preflight.stderr")).write_text(preflight.stderr)
    if preflight.returncode != 0 or json.loads(preflight.stdout)["status"] == "blocked":
        raise SystemExit("Preflight blocked; inspect the retained diagnostics.")
    record = {
        "case": args.case, "prompt": args.prompt, "command": command,
        "checkpoint_revision": "492fd8c4eca666f1085f57e30738c1e0d86b3439",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "binary_sha256": hashlib.file_digest(binary.open("rb"), "sha256").hexdigest(),
        "machine": {"system": platform.platform(),
                    "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                    "physical_memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))},
        "measurement": "/usr/bin/time -l over the native child; wall time includes admission and component loading",
        "visual_inspection": "pending",
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "unset (MLX default enabled)"),
    }
    source_root = Path(__file__).resolve().parents[2]
    source_files = sorted((source_root / "Sources/MereRunH3Model").glob("*.swift"))
    source_files += sorted((source_root / "Sources/MereRunCore/MiniMaxH3Image").glob("*.swift"))
    record["source_base_commit"] = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    record["source_sha256"] = {str(path.relative_to(source_root)):
                              hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    record["model_manifest_sha256"] = hashlib.sha256(
        (args.model / "mererun_model.json").read_bytes()).hexdigest()
    manifest = json.loads((args.model / "mererun_model.json").read_text())
    record["precision"] = manifest.get("precision")
    record["quantization"] = manifest.get("quantization")
    record["h3_image_profile_enabled"] = os.environ.get("MERERUN_H3_IMAGE_PROFILE") == "1"
    events = []
    profiles = []
    started = time.monotonic()
    with stdout_path.open("w") as stdout, stderr_path.open("w", buffering=1) as stderr:
        process = subprocess.Popen(["/usr/bin/time", "-l", *command], stdout=stdout,
                                   stderr=subprocess.PIPE, text=True, bufsize=1)
        print(json.dumps({"event": "benchmark_started", "pid": process.pid, "case": args.case}), flush=True)
        for line in process.stderr:
            stderr.write(line)
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") == "progress":
                event = {"elapsed_seconds": time.monotonic() - started, "payload": payload}
                events.append(event)
                print(json.dumps(event), flush=True)
            elif payload.get("event") == "h3_image_profile":
                profiles.append(payload)
        exit_code = process.wait()
    record.update(exit_code=exit_code, wall_seconds=time.monotonic() - started, progress_events=events)
    if profiles:
        record["profile_events"] = profiles
        for name, selected in [("profile_totals_seconds", profiles),
                               ("profile_after_first_step_seconds", [p for p in profiles if p["step"] > 0])]:
            totals = {}
            for item in selected:
                totals[item["phase"]] = totals.get(item["phase"], 0) + item["seconds"]
            record[name] = totals
    log = stderr_path.read_text()
    for label, key in [("maximum resident set size", "maximum_resident_bytes"),
                       ("peak memory footprint", "peak_memory_footprint_bytes")]:
        match = re.search(r"(\d+)\s+" + label, log)
        record[key] = int(match[1]) if match else None
    if exit_code == 0:
        record["receipt"] = json.loads(stdout_path.read_text().splitlines()[-1])
        with Image.open(image) as result:
            result.load()
            rgb = np.asarray(result.convert("RGB"))
            record["image"] = {"path": str(image), "size": list(result.size), "format": result.format,
                               "file_bytes": image.stat().st_size,
                               "sha256": hashlib.file_digest(image.open("rb"), "sha256").hexdigest(),
                               "rgb_mean": rgb.mean(axis=(0, 1)).tolist(),
                               "rgb_standard_deviation": float(rgb.std()),
                               "black_fraction": float(np.mean(np.all(rgb == 0, axis=-1))),
                               "white_fraction": float(np.mean(np.all(rgb == 255, axis=-1)))}
            if result.size != (args.size, args.size) or result.format != "PNG":
                raise ValueError("Output geometry or format differs from the request.")
    denoising = [event for event in events if event["payload"].get("stage") == "denoising"]
    if denoising:
        record["denoising_progress_span_seconds"] = denoising[-1]["elapsed_seconds"] - denoising[0]["elapsed_seconds"]
        record["denoising_progress_first_step"] = denoising[0]["payload"]["step"]
        if exit_code == 0 and record["denoising_progress_first_step"] == 0 and denoising[-1]["payload"]["step"] == 6:
            record["denoising_seconds"] = record["denoising_progress_span_seconds"]
            record["mean_step_seconds"] = record["denoising_seconds"] / 6
    first_stages = {}
    for event in events:
        first_stages.setdefault(event["payload"]["stage"], event["elapsed_seconds"])
    record["first_stage_elapsed_seconds"] = first_stages
    if exit_code == 0 and "loadingEncoder" in first_stages and events:
        record["active_generation_seconds"] = events[-1]["elapsed_seconds"] - first_stages["loadingEncoder"]
    intervals = {
        "conditioner_load_seconds": ("loadingEncoder", "encodingText"),
        "transformer_load_and_text_refinement_seconds": ("loadingTransformer", "denoising"),
        "vae_load_seconds": ("loadingVAE", "decoding"),
        "decode_and_image_preparation_seconds": ("decoding", "saving"),
    }
    for key, (before, after) in intervals.items():
        if before in first_stages and after in first_stages:
            record[key] = first_stages[after] - first_stages[before]
    encoding = [event for event in events if event["payload"].get("stage") == "encodingText"]
    if len(encoding) > 1:
        record["prompt_encoding_seconds"] = encoding[-1]["elapsed_seconds"] - encoding[0]["elapsed_seconds"]
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"event": "benchmark_finished", "record": str(record_path), "exit_code": exit_code,
                      "wall_seconds": record["wall_seconds"],
                      "peak_memory_footprint_bytes": record["peak_memory_footprint_bytes"]}), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
