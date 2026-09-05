#!/usr/bin/env python3
"""Collect warmed Qwen/Ornith generation receipts from an optimized CLI.

Run one process at a time. Keep synchronized profiles separate from throughput
measurements. The receipt includes every measured request, including outliers.
"""

import argparse
import hashlib
import json
import os
import plistlib
from pathlib import Path
import re
import subprocess
import time


MODELS = {
    "qwen": "vision-chat-q38-27b-4bit",
    "ornith": "text-agent-ornith-35b-mlx-4bit",
}
PROMPTS = {
    "code": "Write a complete Python implementation of an LRU cache using a dictionary and a doubly linked list, with get and put operations, type hints, and a short usage example. Return the code directly without introductory prose.",
    "chat": "I'm trying to get back into reading after a busy year. I have about twenty minutes most evenings, enjoy mysteries and science fiction, and keep abandoning books when I miss a few days. Help me make a gentle, realistic four-week plan. Talk to me conversationally, suggest ways to choose books without buying a stack, and explain what to do when I fall behind. Give enough practical detail that I can start tonight.",
    "prose": "Explain how a small community could turn an abandoned railway station into a public library. Discuss the building, accessibility, staffing, the collection, and a realistic opening-day scene. Write approximately 400 words of clear, connected prose.",
    "math": "Derive the sum of the first n squares using induction. Explain the base case and inductive step, then check your formula for n=10. Show each algebraic step.",
    "summary": "Summarize this fictional incident report for a manager. Explain the timeline, user impact, cause, recovery, and three concrete follow-up actions. Distinguish confirmed facts from open questions. Use clear paragraphs and a short action list.\n\n09:02: The team deployed version 2.8 of the document search service. 09:07: Monitoring showed the 95th percentile search latency rising from 180 milliseconds to 4.2 seconds. 09:10: Support received six reports of slow searches; document uploads and downloads remained available. 09:13: The on-call engineer paused the rollout at 40 percent of traffic. 09:17: Logs showed repeated cache misses for queries containing a language filter. A new cache key included a randomly generated request identifier. 09:21: The engineer rolled traffic back to version 2.7. 09:26: Latency returned below 250 milliseconds. 09:32: The team confirmed no documents were lost or modified. Approximately 1,800 search requests experienced elevated latency; 73 timed out. The exact number of affected users is still being calculated. The release tests covered result correctness but did not assert cache reuse. A separate warning about database connection count appeared at 09:15; the team has not established whether it was a cause or an effect. Proposed follow-ups include a deterministic cache-key unit test, a staging workload with repeated filtered queries, and an automatic rollback threshold for latency.",
}


def command_text(*args):
    return subprocess.check_output(args, text=True).strip()


def gpu_utilization():
    devices = plistlib.loads(subprocess.check_output(["ioreg", "-r", "-c", "AGXAccelerator", "-a"]))
    return max(device["PerformanceStatistics"]["Device Utilization %"] for device in devices)


def snapshot():
    swap = command_text("sysctl", "-n", "vm.swapusage")
    return {
        "swap": swap,
        "swapUsedMiB": float(re.search(r"used = ([0-9.]+)M", swap).group(1)),
        "pressureLevel": int(command_text("sysctl", "-n", "kern.memorystatus_vm_pressure_level")),
        "gpuDeviceUtilizationPercent": gpu_utilization(),
    }


def is_metadata_command(arguments):
    if arguments and arguments[-1] in {"--help", "-h", "--version"}:
        return True
    if arguments[:1] == ["--models-root"]:
        arguments = arguments[2:]
    elif arguments and arguments[0].startswith("--models-root="):
        arguments = arguments[1:]
    if arguments[:1] and arguments[0] in {"status", "catalog", "guide"}:
        return True
    if arguments[:1] == ["model"]:
        return len(arguments) > 1 and arguments[1] in {"list", "location", "info", "capabilities"}
    return False


def inference_processes():
    rows = command_text("ps", "-axo", "pid,comm").splitlines()[1:]
    active = set()
    for row in rows:
        fields = row.split(None, 1)
        if len(fields) != 2 or Path(fields[1]).name != "mere.run":
            continue
        pid, executable = int(fields[0]), fields[1]
        result = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
        command = result.stdout.strip()
        if not command:
            continue
        arguments = command[len(executable):].split() if command.startswith(executable) else []
        if not is_metadata_command(arguments):
            active.add(pid)
    return active


def rendering_processes():
    # HyperFrames can contend for Metal without launching a model process.
    rows = command_text("ps", "-axo", "pid,pcpu,comm").splitlines()[1:]
    candidates = [
        int(fields[0]) for row in rows
        if len(fields := row.split(None, 2)) == 3
        and Path(fields[2]).name == "chrome-headless-shell" and float(fields[1]) > 1
    ]
    active = set()
    for pid in candidates:
        process = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
        if "--type=gpu-process" in process.stdout:
            active.add(pid)
    return active


def run_case(args, root, binary, name, prompt, environment):
    destination = args.output / f"{args.model}-{name}.json"
    if destination.exists():
        raise SystemExit(f"Receipt already exists: {destination}; use a new output directory.")
    others = inference_processes()
    if others:
        raise SystemExit(f"Another inference process is active: {sorted(others)}. Leave it untouched and retry later.")
    renderers = rendering_processes()
    if renderers:
        raise SystemExit(f"A GPU rendering worker is active: {sorted(renderers)}. Leave it untouched and retry later.")
    before = snapshot()
    if before["gpuDeviceUtilizationPercent"] > 10:
        raise SystemExit("GPU is already busy; wait for an idle measurement window.")
    if before["pressureLevel"] != 1:
        raise SystemExit("Memory pressure is elevated; defer this benchmark.")
    command = [
        str(binary), "model", "benchmark", "q36-mtp", "--model", MODELS[args.model],
        "--prompt", prompt, "--decode-tokens", str(args.tokens), "--context-size", "8192",
        "--temperature", str(args.temperature), "--top-p", str(args.top_p), "--json", "--variants", args.variants,
        "--warmups", str(args.warmups), "--warmup-tokens", str(args.warmup_tokens),
        "--repetitions", str(args.repetitions),
    ]
    if args.block_size:
        command.extend(["--mtp-block-size", str(args.block_size)])
    started = time.time()
    print(f"start {args.model} {name}", flush=True)
    peak_rss = 0
    competing = set()
    max_pressure = before["pressureLevel"]
    with destination.with_suffix(".stdout").open("w") as output, destination.with_suffix(".stderr").open("w") as error:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=output, stderr=error)
        while process.poll() is None:
            competing.update(inference_processes() - {process.pid})
            renderers.update(rendering_processes())
            state = snapshot()
            max_pressure = max(max_pressure, state["pressureLevel"])
            rss = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, text=True)
            if rss.stdout.strip():
                peak_rss = max(peak_rss, int(rss.stdout.strip()) * 1024)
            if state["pressureLevel"] >= 4 or state["swapUsedMiB"] - before["swapUsedMiB"] > 512:
                process.terminate()
                process.wait()
                raise SystemExit("Stopped the owned benchmark after critical pressure or more than 512 MiB swap growth.")
            time.sleep(2)
    after = snapshot()
    if process.returncode:
        raise SystemExit(f"Benchmark failed ({process.returncode}); see {destination.with_suffix('.stderr')}")
    report = json.loads(destination.with_suffix(".stdout").read_text())
    report["receipt"] = {
        "workload": name, "prompt": prompt, "command": command,
        "binarySHA256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "sourceHead": command_text("git", "-C", str(root), "rev-parse", "HEAD"),
        "runtimeOverrides": {key: value for key, value in environment.items() if key.startswith("MERERUN_Q35_")},
        "profiled": args.profile, "startedUnixSeconds": started,
        "processSeconds": time.time() - started, "peakResidentBytes": peak_rss,
        "before": before, "after": after, "maxPressureLevel": max_pressure,
        "competingInferencePIDs": sorted(competing),
        "knownRenderingWorkerPIDs": sorted(renderers),
        "uncontended": not competing and not renderers and max_pressure == 1,
    }
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"done {args.model} {name}: {time.time() - started:.1f}s", flush=True)
    if competing or renderers or max_pressure != 1:
        raise SystemExit("Retained a contended receipt; exclude it from throughput conclusions and repeat separately.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, help="Checkout used to build the binary.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workloads", nargs="+", choices=PROMPTS, default=list(PROMPTS))
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--warmup-tokens", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--variants", default="baseline,adaptive")
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--cost-ratio", type=float)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--async-blocks", choices=["0", "1"])
    parser.add_argument("--pipelined-fallback", choices=["0", "1"])
    parser.add_argument("--scoped-compile", choices=["0", "1"])
    args = parser.parse_args()
    root = (args.source_root or Path(__file__).resolve().parents[1]).resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve(strict=True)
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(("MERERUN_Q35_", "MERERUN_Q38_")) or key == "MLX_SDPA_BLOCKS":
            environment.pop(key)
    if args.async_blocks is not None:
        environment["MERERUN_Q35_ASYNC_DECODE_BLOCKS"] = args.async_blocks
    if args.scoped_compile is not None:
        environment["MERERUN_Q35_SCOPED_COMPILE"] = args.scoped_compile
    if args.pipelined_fallback is not None:
        environment["MERERUN_Q35_MTP_PIPELINED_FALLBACK"] = args.pipelined_fallback
    if args.profile:
        environment["MERERUN_Q35_MTP_PROFILE"] = "1"
    if args.cost_ratio is not None:
        environment["MERERUN_Q35_MTP_HEAD_COST_RATIO"] = str(args.cost_ratio)
    for name in args.workloads:
        run_case(args, root, binary, name, PROMPTS[name], environment)


if __name__ == "__main__":
    main()
