#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["coremltools==9.0", "numpy==2.3.5"]
# ///
"""Record anticipated SOV compute placement; pair with an Instruments trace.

This receipt alone cannot establish actual Neural Engine execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from collections import Counter, defaultdict


def inspect_model(path, include_costs=False):
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    operations = []
    constants = 0
    def visit(block, location):
        nonlocal constants
        for index, operation in enumerate(block.operations):
            operator = operation.operator_name
            # Constants have no execution-device assignment. Avoid expensive
            # Core ML lookups for them, especially in a large tiled graph.
            if operator == "const" or operator.rsplit(".", 1)[-1].startswith("constexpr_"):
                constants += 1
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            cost = plan.get_estimated_cost_for_mlprogram_operation(operation) if include_costs else None
            operations.append(dict(path=f"{location}/{index}", operator=operator,
                outputs=[output.name for output in operation.outputs],
                preferredDevice=type(usage.preferred_compute_device).__name__ if usage else None,
                supportedDevices=[type(device).__name__ for device in usage.supported_compute_devices] if usage else [],
                estimatedCostWeight=cost.weight if cost else None))
            for child_index, child in enumerate(operation.blocks):
                visit(child, f"{location}/{index}/block{child_index}")
    for name, function in plan.model_structure.program.functions.items():
        visit(function.block, name)
    costs = defaultdict(float)
    for operation in operations:
        costs[str(operation["preferredDevice"])] += operation["estimatedCostWeight"] or 0
    return dict(nonconstantOperationCount=len(operations), constantOperationCount=constants,
        preferredDeviceCounts=dict(Counter(op["preferredDevice"] for op in operations)),
        estimatedCostWeights=dict(costs) if include_costs else None,
        allNonconstantOperationsPreferANE=bool(operations) and all(op["preferredDevice"] == "MLNeuralEngineComputeDevice" for op in operations),
        operations=operations)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-ane", action="store_true")
    parser.add_argument("--include-costs", action="store_true", help="Query estimated operation costs as well as placement")
    args = parser.parse_args()
    model = inspect_model(args.artifact / "encoder.mlmodelc", include_costs=args.include_costs)
    receipt = dict(schemaVersion=1, evidenceKind="anticipated-device-placement", computeUnits="CPU_AND_NE",
                   manifestSHA256=hashlib.sha256((args.artifact / "sov-coreml.json").read_bytes()).hexdigest(),
                   macOS=platform.mac_ver()[0], hardware=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                   model=model)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k:v for k,v in model.items() if k != "operations"}, indent=2))
    if args.require_ane and not model["allNonconstantOperationsPreferANE"]:
        raise SystemExit("Not all nonconstant operations prefer ANE; inspect the receipt")


if __name__ == "__main__":
    main()
