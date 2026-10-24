#!/usr/bin/env python3
"""Summarize saved Bolt checkpoint diagnostics without re-running inference."""
import argparse
import json
import re
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("records", type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--release-results", type=Path, help="Directory with release-text-1/2 and release-chart logs")
args = parser.parse_args()
records = []
for path in sorted(args.records.glob("*.json")):
    record = json.loads(path.read_text())
    if "generatedTokens" in record:
        records.append(record)
if not records:
    parser.error("No checkpoint measurement records found")


def group(items):
    scored = [item for item in items if "passed" in item]
    rates = [item["generatedTokens"] / item["decodeSeconds"] for item in items
             if item.get("decodeSeconds", 0) > 0]
    prefill = [item["promptTokens"] / item["prefillSeconds"] for item in items
               if item.get("prefillSeconds", 0) > 0 and item.get("promptTokens")]
    return {
        "cases": len(items),
        "scored_cases": len(scored),
        "exact_answer_passes": sum(item["passed"] for item in scored),
        "median_decode_tokens_per_second": statistics.median(rates) if rates else None,
        "median_prefill_tokens_per_second": statistics.median(prefill) if prefill else None,
        "maximum_mlx_peak_bytes": max((item["mlxPeakBytes"] for item in items), default=None),
        "maximum_process_peak_rss_bytes": max((item["processPeakRSSBytes"] for item in items), default=None),
        "maximum_physical_footprint_bytes": max((item.get("peakPhysicalFootprintBytes", 0) for item in items), default=0) or None,
    }


result = {
    "scope": "Small synthetic diagnostic suite; not a general capability or upstream parity benchmark",
    "sampling": {"temperature": 0, "top_p": 1, "seed": 17, "context_limit": 4096},
    "measurement_build": "SwiftPM debug harness, native Metal kernels",
    "memory_scope": "MLX peak reset per case including resident model; RSS and kernel physical-footprint peaks are cumulative for the entire test process",
    "all": group(records),
    "text": group([r for r in records if not r.get("image") and "expected" in r]),
    "images": group([r for r in records if r.get("image")]),
    "speed_repeats": group([r for r in records if r["id"].startswith("decode-speed")]),
    "cases": records,
}
if args.release_results:
    release = []
    for name in ["release-text-1", "release-text-2", "release-chart"]:
        log = (args.release_results / (name + ".log")).read_text()
        text = (args.release_results / (name + ".txt")).read_text().strip()
        stats = next(line for line in log.splitlines() if line.startswith("time="))
        fields = {key: float(value) for key, value in re.findall(r"([a-z_]+)=([0-9.]+)", stats)}
        physical = re.search(r"(\d+)\s+peak memory footprint", log)
        rss = re.search(r"(\d+)\s+maximum resident set size", log)
        release.append({"id": name, "output": text, "timing": fields,
                        "peak_physical_footprint_bytes": int(physical[1]) if physical else None,
                        "peak_rss_bytes": int(rss[1]) if rss else None})
    result["release_cli"] = {
        "build": "SwiftPM release (-O), native Metal kernels",
        "sampling": {"temperature": 0, "top_p": 1, "context_limit": 4096, "thinking": False},
        "text_decode_tokens_per_second": [r["timing"]["decode_tps"] for r in release[:2]],
        "median_text_decode_tokens_per_second": statistics.median(r["timing"]["decode_tps"] for r in release[:2]),
        "text_outputs_identical": release[0]["output"] == release[1]["output"],
        "chart_exact_answer_passed": release[2]["output"] == "B",
        "maximum_physical_footprint_bytes": max(r["peak_physical_footprint_bytes"] or 0 for r in release),
        "runs": release,
    }

args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
