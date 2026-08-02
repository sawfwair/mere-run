#!/usr/bin/env python3
"""Run paper-aligned DreamX duration and revisit scenarios through mere.run.

The harness records generation truth, media truth, global-pose loop closure, and
scene-memory telemetry. It never substitutes geometry success for visual score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(base: str, method: str, path: str, token: str | None, body=None, content_type=None):
    headers = {}
    if token:
        headers["authorization"] = f"Bearer {token}"
    if content_type:
        headers["content-type"] = content_type
    data = body if isinstance(body, bytes) else None
    if body is not None and not isinstance(body, bytes):
        data = json.dumps(body).encode()
        headers["content-type"] = "application/json"
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=120) as response:
        payload = response.read()
        return response.status, response.headers, payload


def request_json(*args, **kwargs):
    status, _, payload = request(*args, **kwargs)
    return status, json.loads(payload)


def download(base: str, path: str, token: str | None, output: Path):
    _, _, payload = request(base, "GET", path, token)
    output.write_bytes(payload)


def wait_for_job(base: str, job_id: str, token: str | None, timeout: float, poll: float):
    started = time.monotonic()
    first_chunk = None
    peak_chunk_count = 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, job = request_json(base, "GET", f"/v1/world/jobs/{job_id}", token)
        peak_chunk_count = max(peak_chunk_count, len(job.get("chunks", [])))
        if first_chunk is None and peak_chunk_count:
            first_chunk = time.monotonic() - started
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job, {
                "time_to_first_chunk_seconds": first_chunk,
                "total_seconds": time.monotonic() - started,
                "observed_chunk_count": peak_chunk_count,
            }
        time.sleep(poll)
    raise TimeoutError(f"DreamX job {job_id} did not finish within {timeout:g}s")


def ffprobe(path: Path):
    command = [
        "ffprobe", "-v", "error", "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
        "-of", "json", str(path),
    ]
    try:
        return json.loads(subprocess.check_output(command, text=True))
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {"unavailable": str(error)}


def media_contract(probe, scenario):
    errors = []
    streams = probe.get("streams", [])
    formats = probe.get("format", {})
    if not streams:
        return {"status": "failed", "errors": ["ffprobe did not return a video stream"]}
    stream = streams[0]
    if "expected_pixel_frames" in scenario:
        actual = int(stream.get("nb_read_frames", -1))
        expected = scenario["expected_pixel_frames"]
        if actual != expected:
            errors.append(f"encoded frame count {actual}, expected {expected}")
    if "expected_duration_seconds" in scenario:
        actual = float(formats.get("duration", "nan"))
        expected = scenario["expected_duration_seconds"]
        if not math.isclose(actual, expected, abs_tol=1e-3):
            errors.append(f"encoded duration {actual}, expected {expected}")
    return {"status": "passed" if not errors else "failed", "errors": errors}


def camera_position(world_to_camera):
    rotation = [
        [world_to_camera[0], world_to_camera[1], world_to_camera[2]],
        [world_to_camera[4], world_to_camera[5], world_to_camera[6]],
        [world_to_camera[8], world_to_camera[9], world_to_camera[10]],
    ]
    translation = [world_to_camera[3], world_to_camera[7], world_to_camera[11]]
    return [
        -sum(rotation[row][column] * translation[row] for row in range(3))
        for column in range(3)
    ]


def pose_metrics(pose):
    if not pose:
        return None
    matrix = pose["world_to_camera"]
    position = camera_position(matrix)
    # Camera-to-world forward is the third row of world-to-camera.
    forward = [matrix[8], matrix[9], matrix[10]]
    yaw = math.degrees(math.atan2(forward[0], forward[2]))
    yaw = abs((yaw + 180) % 360 - 180)
    distance = math.sqrt(sum(value * value for value in position))
    return {
        "translation_distance_from_origin": distance,
        "yaw_distance_degrees_from_origin": yaw,
        "paper_revisit_gate_pass": distance <= 0.1 and yaw <= 2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8791")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--suite", type=Path, default=Path(__file__).with_name("dreamx_eval_suite.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", action="append", help="Run only the named scenario; repeatable")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--poll", type=float, default=0.1)
    parser.add_argument("--token", default=os.environ.get("MERE_RUN_API_KEY"))
    args = parser.parse_args()

    suite = json.loads(args.suite.read_text())
    selected = set(args.scenario or [])
    scenarios = [item for item in suite["scenarios"] if not selected or item["id"] in selected]
    missing = selected - {item["id"] for item in scenarios}
    if missing:
        raise SystemExit(f"Unknown scenarios: {', '.join(sorted(missing))}")
    if not args.source.is_file():
        raise SystemExit(f"Source image not found: {args.source}")

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "status": "running",
        "runtime": args.base_url,
        "source": str(args.source.resolve()),
        "prompt": args.prompt,
        "geometry": {"width": args.width, "height": args.height},
        "suite": str(args.suite.resolve()),
        "scenarios": [],
    }
    _, runtime_snapshot = request_json(args.base_url, "GET", "/v1/world/session", args.token)
    report["runtime_snapshot"] = runtime_snapshot

    for scenario in scenarios:
        scenario_dir = args.output / scenario["id"]
        scenario_dir.mkdir(parents=True, exist_ok=True)
        request(
            args.base_url, "POST", "/v1/world/session/source", args.token,
            args.source.read_bytes(), "application/octet-stream",
        )
        scenario_report = {
            "id": scenario["id"],
            "kind": scenario["kind"],
            "baseline_scenario_id": scenario.get("baseline_scenario_id"),
            "periodic_revisit_stride": scenario.get("periodic_revisit_stride"),
            "minimum_periodic_revisits": scenario.get("minimum_periodic_revisits"),
            "steps": [],
        }
        steps = scenario["steps"] * scenario.get("repeat", 1)
        for index, step in enumerate(steps):
            payload = {
                "prompt": args.prompt,
                "action_seq": step["action_seq"],
                "action_speed_list": step["action_speed_list"],
                "width": args.width,
                "height": args.height,
                "num_output_frames": step["latent_frame_count"],
                "speed": suite["speed"],
                "seed": args.seed,
                "fps": suite["fps"],
            }
            _, accepted = request_json(
                args.base_url, "POST", "/v1/world/session/rollouts", args.token, payload
            )
            job_id = accepted["job_id"]
            job, performance = wait_for_job(
                args.base_url, job_id, args.token, args.timeout, args.poll
            )
            (scenario_dir / f"step-{index + 1:02d}-job.json").write_text(json.dumps(job, indent=2) + "\n")
            if job["status"] != "completed":
                raise RuntimeError(f"{scenario['id']} step {index + 1} failed: {job.get('error')}")
            receipt = job["rollout_receipt"]
            video = scenario_dir / f"step-{index + 1:02d}.mp4"
            terminal = scenario_dir / f"step-{index + 1:02d}-terminal.png"
            download(args.base_url, receipt["output_media_path"], args.token, video)
            download(args.base_url, receipt["terminal_frame_media_path"], args.token, terminal)
            scenario_report["steps"].append({
                "request": payload,
                "job_id": job_id,
                "receipt": receipt,
                "performance": performance,
                "media_probe": ffprobe(video),
                "video": str(video.resolve()),
                "terminal_frame": str(terminal.resolve()),
            })

        _, session = request_json(args.base_url, "GET", "/v1/world/session", args.token)
        snapshot = session["session"]
        scenario_report["terminal_session"] = snapshot
        scenario_report["pose_metrics"] = pose_metrics(snapshot.get("current_world_pose"))
        errors = []
        if "expected_pixel_frames" in scenario or "expected_duration_seconds" in scenario:
            contract = media_contract(scenario_report["steps"][-1]["media_probe"], scenario)
            scenario_report["media_contract"] = contract
            errors.extend(contract["errors"])
        if scenario.get("expected_origin_recovery") is True:
            pose = scenario_report["pose_metrics"]
            if not pose or not pose["paper_revisit_gate_pass"]:
                errors.append("terminal pose did not satisfy the paper revisit gate")
        if "expected_action_count" in scenario and len(steps) != scenario["expected_action_count"]:
            errors.append(
                f"executed {len(steps)} action requests, expected {scenario['expected_action_count']}"
            )
        scenario_report["status"] = "passed" if not errors else "failed"
        scenario_report["errors"] = errors
        completed_steps = [
            step["performance"] for step in scenario_report["steps"]
            if step["performance"].get("total_seconds") is not None
        ]
        scenario_report["performance"] = {
            "step_count": len(completed_steps),
            "mean_total_seconds": (
                sum(step["total_seconds"] for step in completed_steps) / len(completed_steps)
                if completed_steps else None
            ),
            "mean_time_to_first_chunk_seconds": (
                sum(step["time_to_first_chunk_seconds"] for step in completed_steps) / len(completed_steps)
                if completed_steps and all(
                    step.get("time_to_first_chunk_seconds") is not None for step in completed_steps
                ) else None
            ),
        }
        scenario_report["visual_metrics"] = {
            "status": "not_scored",
            "required": ["PSNR", "SSIM", "LPIPS", "DINO-Sim", "VPR-Sim", "SP-Match", "CLIP-Video"],
            "note": "Run a pinned metric environment against the captured media; geometry is not visual parity.",
        }
        report["scenarios"].append(scenario_report)
        (scenario_dir / "report.json").write_text(json.dumps(scenario_report, indent=2) + "\n")

    failed = [scenario for scenario in report["scenarios"] if scenario["status"] != "passed"]
    report["status"] = "passed" if not failed else "failed"
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
