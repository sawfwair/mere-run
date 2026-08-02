#!/usr/bin/env python3
"""Issue a machine-readable DreamX product-experience gate receipt."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = min(len(ordered) - 1, math.ceil(proportion * len(ordered)) - 1)
    return ordered[index]


def process_receipt(pid: int | None) -> dict:
    if pid is None:
        return {"status": "not_requested"}
    try:
        output = subprocess.check_output(
            ["ps", "-o", "pid=,rss=,%cpu=,%mem=,etime=", "-p", str(pid)], text=True
        ).strip()
        fields = output.split()
        return {
            "status": "captured",
            "pid": int(fields[0]),
            "rss_kib": int(fields[1]),
            "cpu_percent": float(fields[2]),
            "memory_percent": float(fields[3]),
            "elapsed": fields[4],
        }
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError) as error:
        return {"status": "unavailable", "error": str(error)}


def thermal_receipt() -> dict:
    try:
        output = subprocess.check_output(["pmset", "-g", "therm"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        return {"status": "unavailable", "error": str(error)}
    lowered = output.lower()
    warning = "warning level has been recorded" in lowered and "no thermal warning" not in lowered
    performance_warning = (
        "performance warning level has been recorded" in lowered
        and "no performance warning" not in lowered
    )
    return {
        "status": "failed" if warning or performance_warning else "passed",
        "thermal_warning": warning,
        "performance_warning": performance_warning,
        "raw": output,
    }


def check(condition: bool, name: str, detail: dict, failures: list[str]) -> None:
    detail[name] = "passed" if condition else "failed"
    if not condition:
        failures.append(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-report", type=Path, required=True)
    parser.add_argument("--soak-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-pid", type=int)
    parser.add_argument("--minimum-actions", type=int, default=100)
    parser.add_argument("--minimum-periodic-revisits", type=int, default=13)
    parser.add_argument("--maximum-mean-first-block-seconds", type=float, default=7.5)
    parser.add_argument("--maximum-p95-first-block-seconds", type=float, default=8.5)
    parser.add_argument("--maximum-mean-move-seconds", type=float, default=8.0)
    parser.add_argument("--maximum-rss-gib", type=float, default=32.0)
    args = parser.parse_args()

    paired = json.loads(args.paired_report.read_text())
    soak = json.loads(args.soak_report.read_text())
    soak_scenarios = [
        scenario for scenario in soak["scenarios"] if scenario["kind"] == "soak"
    ]
    if len(soak_scenarios) != 1:
        raise SystemExit(f"Expected one soak scenario, found {len(soak_scenarios)}")
    scenario = soak_scenarios[0]
    performance = [step["performance"] for step in scenario["steps"]]
    first_blocks = [
        item["time_to_first_chunk_seconds"] for item in performance
        if item.get("time_to_first_chunk_seconds") is not None
    ]
    totals = [item["total_seconds"] for item in performance]
    periodic = scenario.get("visual_metrics", {}).get("periodic_revisits", [])
    session = scenario["terminal_session"]
    runtime_policy = soak.get("runtime_snapshot", {}).get("session", {}).get(
        "scene_memory_policy", {}
    )
    process = process_receipt(args.runtime_pid)
    thermal = thermal_receipt()
    failures: list[str] = []
    checks: dict[str, str] = {}

    check(paired.get("learned_visual_gate", {}).get("status") == "passed",
          "paired_learned_visual_gate", checks, failures)
    check(soak.get("status") == "passed", "soak_capture_status", checks, failures)
    check(scenario.get("status") == "passed", "soak_scenario_status", checks, failures)
    check(len(scenario["steps"]) >= args.minimum_actions,
          "minimum_action_count", checks, failures)
    check(scenario.get("pose_metrics", {}).get("paper_revisit_gate_pass") is True,
          "terminal_pose_closure", checks, failures)
    check(scenario.get("visual_metrics", {}).get("status") == "passed",
          "periodic_learned_visual_gate", checks, failures)
    check(len(periodic) >= args.minimum_periodic_revisits,
          "minimum_periodic_revisits", checks, failures)
    check(len(first_blocks) == len(scenario["steps"]),
          "all_first_blocks_observed", checks, failures)
    check(statistics.fmean(first_blocks) <= args.maximum_mean_first_block_seconds,
          "mean_first_block_latency", checks, failures)
    check(percentile(first_blocks, 0.95) <= args.maximum_p95_first_block_seconds,
          "p95_first_block_latency", checks, failures)
    check(statistics.fmean(totals) <= args.maximum_mean_move_seconds,
          "mean_move_latency", checks, failures)
    check(session.get("scene_memory_retrieval_count", 0) > 0,
          "scene_memory_exercised", checks, failures)
    check(session.get("scene_memory_frame_count", 0) <= runtime_policy.get("maximum_frame_count", -1),
          "scene_memory_bounded", checks, failures)
    check(runtime_policy.get("mode") == "paper_reconstructed_revisit_anchor",
          "runtime_policy_receipted", checks, failures)
    check("third-person" in soak.get("prompt", "").lower(),
          "third_person_conditioning", checks, failures)
    check(thermal.get("status") == "passed", "thermal_gate", checks, failures)
    if process.get("status") == "captured":
        check(process["rss_kib"] / 1024 / 1024 <= args.maximum_rss_gib,
              "resident_memory_gate", checks, failures)
    else:
        failures.append("resident_memory_gate")
        checks["resident_memory_gate"] = "failed"

    receipt = {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "inputs": {
            "paired_report": str(args.paired_report.resolve()),
            "soak_report": str(args.soak_report.resolve()),
        },
        "checks": checks,
        "failures": failures,
        "measurements": {
            "actions": len(scenario["steps"]),
            "periodic_revisits": len(periodic),
            "mean_first_block_seconds": statistics.fmean(first_blocks),
            "p95_first_block_seconds": percentile(first_blocks, 0.95),
            "mean_move_seconds": statistics.fmean(totals),
            "p95_move_seconds": percentile(totals, 0.95),
            "scene_memory_frame_count": session.get("scene_memory_frame_count"),
            "scene_memory_retrieval_count": session.get("scene_memory_retrieval_count"),
            "scene_memory_recycled_frame_count": session.get("scene_memory_recycled_frame_count"),
            "runtime_policy": runtime_policy,
        },
        "process": process,
        "thermal": thermal,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
