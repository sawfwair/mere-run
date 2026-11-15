#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["coremltools==9.0", "numpy==2.3.5", "safetensors==0.8.0", "torch==2.7.0", "scikit-learn==1.9.1"]
# ///
"""Measure isolated original-weight SOV components and exact layout alternatives.

Compute-plan costs are estimates. This tool separately converts real stem,
normalization, attention, feed-forward, projector, and block computations and
measures warmed Core ML prediction latency. Isolated timings include a model
call; their sum is diagnostic and is not a full-model latency measurement.
"""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
import statistics
import time

from convert_sov_coreml import build_graph, conversion_precision, palettize_model

VARIANTS = {
    "stem": {"stage": "stem"},
    "projector": {"stage": "projector"},
    "norm-baseline": {"stage": "norm"},
    "norm-channel": {"stage": "norm", "channel_norm": True},
    "mlp-baseline": {"stage": "mlp"},
    "mlp-channel": {"stage": "mlp", "channel_norm": True},
    "mlp-width1": {"stage": "mlp", "spatial_width": 1},
    "mlp-width100": {"stage": "mlp", "spatial_width": 100},
    "mlp-width50": {"stage": "mlp", "spatial_width": 50},
    "attention-baseline": {"stage": "attention"},
    "attention-channel": {"stage": "attention", "attention_layout": "channel"},
    "attention-channel20": {"stage": "attention", "attention_layout": "channel", "head_group_size": 20},
    "attention-fused": {"stage": "attention", "fuse_qkv": True},
    "attention-sdpa": {"stage": "attention", "attention_layout": "sdpa"},
    "mlp-chunk32": {"stage": "mlp", "mlp_chunk_size": 32},
    "mlp-chunk64": {"stage": "mlp", "mlp_chunk_size": 64},
    "attention-original-query128": {"stage": "attention", "query_chunk_size": 128},
    "attention-original-query256": {"stage": "attention", "query_chunk_size": 256},
    "attention-chunk32": {"stage": "attention", "attention_layout": "channel", "query_chunk_size": 32},
    "attention-chunk64": {"stage": "attention", "attention_layout": "channel", "query_chunk_size": 64},
    "attention-chunk64-head1": {"stage": "attention", "attention_layout": "channel", "query_chunk_size": 64, "head_group_size": 1},
    "attention-chunk64-head20": {"stage": "attention", "attention_layout": "channel", "query_chunk_size": 64, "head_group_size": 20},
    "block-tiled": {"stage": "block", "channel_norm": True, "attention_layout": "channel", "query_chunk_size": 128, "head_group_size": 1, "mlp_chunk_size": 128},
    "mlp-linear": {"stage": "mlp", "linear_operator": True},
    "mlp-chunk128": {"stage": "mlp", "mlp_chunk_size": 128},
    "mlp-chunk256": {"stage": "mlp", "mlp_chunk_size": 256},
    "mlp-chunk500": {"stage": "mlp", "mlp_chunk_size": 500},
    "attention-chunk128": {"stage": "attention", "attention_layout": "channel", "query_chunk_size": 128},
    "attention-chunk256": {"stage": "attention", "attention_layout": "channel", "query_chunk_size": 256},
    "attention-chunk500": {"stage": "attention", "attention_layout": "channel", "query_chunk_size": 500},
    "attention-chunk128-head1": {"stage": "attention", "attention_layout": "channel", "query_chunk_size": 128, "head_group_size": 1},
    "mlp-w8a16": {"stage": "mlp", "quantization": "w8a16"},
    "mlp-w4a16": {"stage": "mlp", "quantization": "w4a16"},
    "block-w8a16": {"stage": "block", "quantization": "w8a16"},
    "block-w4a16": {"stage": "block", "quantization": "w4a16"},
    "mlp-w8a8": {"stage": "mlp", "quantization": "w8a8"},
    "block-w8a8": {"stage": "block", "quantization": "w8a8"},
    "block-baseline": {"stage": "block"},
    "attention-einsum128": {"stage": "attention", "attention_layout": "einsum", "query_chunk_size": 128},
    "attention-einsum500": {"stage": "attention", "attention_layout": "einsum", "query_chunk_size": 500},
    "block-einsum": {"stage": "block", "attention_layout": "einsum", "query_chunk_size": 128, "mlp_chunk_size": 128},
    "block-sensitive-fp32": {"stage": "block", "attention_layout": "channel", "head_group_size": 1, "query_chunk_size": 128, "mlp_chunk_size": 128, "sensitive_fp32": True},
    "block-p8": {"stage": "block", "palettization_bits": 8},
    "block-p6": {"stage": "block", "palettization_bits": 6},
    "block-p6g16": {"stage": "block", "palettization_bits": 6, "palette_group_size": 16},
    "block-tiled-p8": {"stage": "block", "query_chunk_size": 128, "mlp_chunk_size": 128, "palettization_bits": 8},
    "block-tiled-p6": {"stage": "block", "query_chunk_size": 128, "mlp_chunk_size": 128, "palettization_bits": 6},
    "block-tiled-p8g16": {"stage": "block", "query_chunk_size": 128, "mlp_chunk_size": 128, "palettization_bits": 8, "palette_group_size": 16},
    "block-scaled-einsum": {"stage": "block", "channel_norm": True, "normalization_scale": 8.0, "attention_layout": "einsum", "query_chunk_size": 128, "mlp_chunk_size": 128},
    "block-scaled-einsum500": {"stage": "block", "channel_norm": True, "normalization_scale": 8.0, "attention_layout": "einsum", "query_chunk_size": 500, "mlp_chunk_size": 128},
    "block-einsum500-p8": {"stage": "block", "channel_norm": True, "normalization_scale": 8.0, "attention_layout": "einsum", "query_chunk_size": 500, "mlp_chunk_size": 128, "palettization_bits": 8},
    "block-einsum500-p6": {"stage": "block", "channel_norm": True, "normalization_scale": 8.0, "attention_layout": "einsum", "query_chunk_size": 500, "mlp_chunk_size": 128, "palettization_bits": 6},
}


def placement(model_path):
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(str(model_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts = Counter()
    for function in plan.model_structure.program.functions.values():
        for operation in function.block.operations:
            if operation.operator_name != "const" and not operation.operator_name.rsplit(".", 1)[-1].startswith("constexpr_"):
                usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
                counts[type(usage.preferred_compute_device).__name__ if usage else "unknown"] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--layer-index", type=int, choices=range(32), default=0)
    args = parser.parse_args()
    import coremltools as ct
    import numpy as np
    from safetensors.numpy import load_file
    import torch
    torch.set_num_threads(8)
    graph = build_graph(args.checkpoint)
    mel = torch.from_numpy(load_file(args.features)["mel"][0:1, :, None, :])
    with torch.inference_mode():
        hidden = graph.stem(mel)
        for layer in range(args.layer_index):
            hidden = graph.block(hidden, layer)
    defaults = dict(stage="full", channel_norm=False, spatial_width=1500, attention_layout="qk", fuse_qkv=False, head_group_size=4, query_chunk_size=1500, mlp_chunk_size=1500, linear_operator=False, quantization=None, palettization_bits=None, palette_group_size=0, sensitive_fp32=False, layer_index=args.layer_index, normalization_scale=1.0)
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(Path(__file__).with_name("convert_sov_coreml.py").read_bytes()).hexdigest()
    features_hash = hashlib.sha256(args.features.read_bytes()).hexdigest()
    reports = []
    for name in args.variants:
        options = dict(VARIANTS[name])
        if options.get("attention_layout") == "einsum":
            options["head_group_size"] = 1
        for key, value in (defaults | options).items():
            setattr(graph, key, value)
        example = mel if options["stage"] == "stem" else hidden
        with torch.inference_mode():
            reference = graph(example).numpy()
            traced = torch.jit.trace(graph, example, check_trace=False)
        artifact = args.output / (name + ".mlpackage")
        fingerprint = dict(options=defaults | options, converterSHA256=source_hash)
        fingerprint_path = args.output / (name + ".conversion.json")
        if artifact.exists() and (not fingerprint_path.exists() or json.loads(fingerprint_path.read_text()) != fingerprint):
            raise ValueError(f"Cached graph differs from current conversion code: {artifact}; use a new output directory")
        if not artifact.exists():
            converted = ct.convert(traced, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                inputs=[ct.TensorType(name="x", shape=example.shape, dtype=np.float16)],
                outputs=[ct.TensorType(name="y", dtype=np.float16)], compute_precision=conversion_precision(options.get("sensitive_fp32", False)),
                compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
            if options.get("quantization"):
                import coremltools.optimize.coreml as optimization
                config = optimization.OptimizationConfig(global_config=optimization.OpLinearQuantizerConfig(mode="linear_symmetric"))
                samples = [{"x": np.ascontiguousarray(example.numpy(), dtype=np.float16)}]
                if options["quantization"] == "w8a8":
                    converted = optimization.linear_quantize_activations(converted, config, samples)
                if options["quantization"] == "w4a16":
                    config = optimization.OptimizationConfig(global_config=optimization.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4", block_size=32))
                converted = optimization.linear_quantize_weights(converted, config)
            if options.get("palettization_bits"):
                converted = palettize_model(converted, options["palettization_bits"], options.get("palette_group_size", 0))
            converted.save(str(artifact))
            fingerprint_path.write_text(json.dumps(fingerprint, indent=2) + "\n")
        model = ct.models.MLModel(str(artifact), compute_units=ct.ComputeUnit.CPU_AND_NE)
        inputs = {"x": np.ascontiguousarray(example.numpy(), dtype=np.float16)}
        for _ in range(2):
            model.predict(inputs)
        seconds = []
        for _ in range(args.repetitions):
            start = time.perf_counter()
            predicted = model.predict(inputs)["y"]
            seconds.append(time.perf_counter() - start)
        a, b = reference.astype(np.float64), predicted.astype(np.float64)
        report = dict(name=name, options=options, converterSHA256=source_hash, featuresSHA256=features_hash, predictionSeconds=seconds, medianSeconds=statistics.median(seconds),
                      relativeRMSE=float(np.linalg.norm(a-b) / np.linalg.norm(a)),
                      cosine=float(np.sum(a*b) / np.linalg.norm(a) / np.linalg.norm(b)),
                      finite=bool(np.isfinite(b).all()), anticipatedDevices=placement(model.get_compiled_model_path()))
        reports.append(report)
        print(json.dumps(report), flush=True)
        (args.output / "profile.json").write_text(json.dumps(reports, indent=2) + "\n")
        del model


if __name__ == "__main__":
    main()
