#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["coremltools==9.0", "numpy==2.3.5"]
# ///
"""Inspect device assignment for both compiled Parakeet models on this Mac.

The compute plan describes anticipated placement, not measured utilization.
Pair this receipt with an Instruments Core ML trace of the native workload.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def inspect_model(path: Path) -> dict:
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(
        str(path), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    if plan.model_structure.program is None:
        raise ValueError(f"Expected an ML Program: {path}")
    operations = []

    def visit(block, location):
        for index, operation in enumerate(block.operations):
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            cost = plan.get_estimated_cost_for_mlprogram_operation(operation)
            operations.append({
                "path": f"{location}/{index}",
                "operator": operation.operator_name,
                "outputs": [output.name for output in operation.outputs],
                "preferredDevice": type(usage.preferred_compute_device).__name__ if usage else None,
                "supportedDevices": [type(device).__name__ for device in usage.supported_compute_devices] if usage else [],
                "estimatedCostWeight": cost.weight if cost else None,
            })
            for child_index, child in enumerate(operation.blocks):
                visit(child, f"{location}/{index}/block{child_index}")

    for name, function in plan.model_structure.program.functions.items():
        visit(function.block, name)
    active = [operation for operation in operations if operation["operator"] != "const"]
    costs = defaultdict(float)
    for operation in active:
        costs[str(operation["preferredDevice"])] += operation["estimatedCostWeight"] or 0
    return {
        "nonconstantOperationCount": len(active),
        "constantOperationCount": len(operations) - len(active),
        "preferredDeviceCounts": dict(Counter(operation["preferredDevice"] for operation in active)),
        "estimatedCostWeights": dict(costs),
        "allNonconstantOperationsPreferANE": bool(active) and all(
            operation["preferredDevice"] == "MLNeuralEngineComputeDevice" for operation in active
        ),
        "operations": operations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, help="Write the JSON receipt to a file")
    parser.add_argument("--require-ane", action="store_true", help="Fail if any nonconstant operation prefers another device or has unknown placement")
    args = parser.parse_args()
    import coremltools as ct

    root = args.artifact.resolve(strict=True)
    manifest = root / "parakeet-coreml.json"
    receipt = {
        "schemaVersion": 1,
        "evidenceKind": "anticipated-device-placement",
        "artifact": str(root),
        "manifestSHA256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "macOS": platform.mac_ver()[0],
        "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "coremltools": ct.__version__,
        "computeUnits": "CPU_AND_NE",
        "models": {
            name: inspect_model(root / f"{name}.mlmodelc")
            for name in ("encoder", "decoder")
        },
    }
    passed = all(model["allNonconstantOperationsPreferANE"] for model in receipt["models"].values())
    receipt["allNonconstantOperationsPreferANE"] = passed
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    if args.require_ane and not passed:
        print("ANE placement check failed; inspect the per-operation receipt.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
