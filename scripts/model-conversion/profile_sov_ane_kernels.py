#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["coremltools==9.0", "numpy==2.3.5", "torch==2.7.0", "safetensors==0.8.0", "scikit-learn==1.9.1"]
# ///
"""Measure calibrated quantization and bounded-memory SOV kernels on ANE.

Calibration uses independent training utterances. Numerical errors are measured
on the disjoint held-out activations, against the original FP32 checkpoint.
Input-only quantization avoids an extra rounding before floating-point GELU.
Core ML's graph passes used for that experiment are pinned to version 9.0.
Q/DQ nodes establish the requested quantization, not the hardware MAC precision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from convert_sov_coreml import build_graph
from profile_sov_coreml import placement

VARIANTS = {
    "block-fp16": {},
    "block-delayed": {"attention_layout": "streaming"},
    "block-delayed-ch1024": {"attention_layout": "streaming", "mlp_channel_chunk_size": 1024},
    "block-delayed-q250": {"attention_layout": "streaming", "query_chunk_size": 250},
    "block-delayed-q750": {"attention_layout": "streaming", "query_chunk_size": 750},
    "block-delayed-a8": {"attention_layout": "streaming", "scope": "mlp", "smooth": 0.5, "mlp_chunk_size": 1500},
    "block-delayed-a8-fc1": {"attention_layout": "streaming", "scope": "fc1", "smooth": 0.5, "mlp_chunk_size": 1500},
    "block-delayed-a8-smooth75": {"attention_layout": "streaming", "scope": "mlp", "smooth": 0.75, "mlp_chunk_size": 1500},
    "block-aligned": {"attention_layout": "streaming", "encoder_frames": 1536, "query_chunk_size": 512, "key_chunk_size": 1536},
    "block-aligned-ch1024": {"attention_layout": "streaming", "encoder_frames": 1536, "query_chunk_size": 512, "key_chunk_size": 1536, "mlp_channel_chunk_size": 1024},
    "block-outlier64": {"attention_layout": "streaming", "scope": "regular", "smooth": 0.5, "outliers": 64, "mlp_chunk_size": 1500},
    "block-outlier128": {"attention_layout": "streaming", "scope": "regular", "smooth": 0.5, "outliers": 128, "mlp_chunk_size": 1500},
    "block-outlier256": {"attention_layout": "streaming", "scope": "regular", "smooth": 0.5, "outliers": 256, "mlp_chunk_size": 1500},
    "block-outlier512": {"attention_layout": "streaming", "scope": "regular", "smooth": 0.5, "outliers": 512, "mlp_chunk_size": 1500},
    "block-outlier1024": {"attention_layout": "streaming", "scope": "regular", "smooth": 0.5, "outliers": 1024, "mlp_chunk_size": 1500},
    "block-outlier128-output": {"attention_layout": "streaming", "scope": "regular", "smooth": 0.5, "outliers": 128, "mlp_chunk_size": 1500, "regular_output": True},
    "block-qkv-head1": {"attention_layout": "streaming", "qkv_head_chunk_size": 1},
    "block-qkv-head2": {"attention_layout": "streaming", "qkv_head_chunk_size": 2},
    "block-ffn-channel-schedule512": {"attention_layout": "streaming", "mlp_channel_chunk_size": 512, "mlp_schedule": "channels"},
    "block-ffn-channel-schedule1024": {"attention_layout": "streaming", "mlp_channel_chunk_size": 1024, "mlp_schedule": "channels"},
    "block-ffn-channel-schedule2048": {"attention_layout": "streaming", "mlp_channel_chunk_size": 2048, "mlp_schedule": "channels"},
    "block-mlp-a8-symmetric": {"scope": "mlp", "mode": "linear_symmetric"},
    "block-mlp-a8": {"scope": "mlp"},
    "block-mlp-a8-smooth50": {"scope": "mlp", "smooth": 0.5},
    "block-mlp-a8-smooth75": {"scope": "mlp", "smooth": 0.75},
    "block-fc1-a8-smooth50": {"scope": "fc1", "smooth": 0.5},
    "block-fc2-a8-smooth50": {"scope": "fc2", "smooth": 0.5},
    "block-conv-a8-smooth50": {"scope": "conv", "smooth": 0.5},
    "block-mlp-a8-smooth50-output": {"scope": "mlp", "smooth": 0.5, "both_sides": True},
    "block-conv-a8-smooth50-output": {"scope": "conv", "smooth": 0.5, "both_sides": True},
    "mlp-a8-smooth50": {"stage": "mlp", "scope": "mlp", "smooth": 0.5},
    "mlp-a8-smooth50-output": {"stage": "mlp", "scope": "mlp", "smooth": 0.5, "both_sides": True},
    "block-mlp-a8-channel-output": {"scope": "mlp", "smooth": 0.5, "channel_outputs": True},
    "block-conv-a8-channel-output": {"scope": "conv", "smooth": 0.5, "channel_outputs": True},
    "mlp-a8-channel-output": {"stage": "mlp", "scope": "mlp", "smooth": 0.5, "channel_outputs": True},
    "block-mlp-a8-full": {"scope": "mlp", "smooth": 0.5, "mlp_chunk_size": 1500},
    "block-mlp-a8-tile500": {"scope": "mlp", "smooth": 0.5, "mlp_chunk_size": 500},
    "norm-channel": {"stage": "norm"},
    "norm-instance": {"stage": "norm", "normalization_layout": "instance"},
    "norm-moments": {"stage": "norm", "normalization_layout": "moments"},
    "attention-einsum": {"stage": "attention"},
    "attention-direct": {"stage": "attention", "attention_layout": "direct"},
    "attention-direct-stream": {"stage": "attention", "attention_layout": "direct-streaming"},
    "attention-batch2": {"stage": "attention", "attention_layout": "batch-streaming", "head_group_size": 2},
    "attention-batch4": {"stage": "attention", "attention_layout": "batch-streaming", "head_group_size": 4},
    "attention-packed4": {"stage": "attention", "attention_layout": "packed", "head_group_size": 4},
    "attention-packed20": {"stage": "attention", "attention_layout": "packed", "head_group_size": 20},
    "attention-packed-stream4": {"stage": "attention", "attention_layout": "packed-streaming", "head_group_size": 4},
    "attention-packed-stream20": {"stage": "attention", "attention_layout": "packed-streaming", "head_group_size": 20},
    "attention-unnormalized": {"stage": "attention", "attention_layout": "streaming", "key_chunk_size": 1500},
    "attention-stream500": {"stage": "attention", "attention_layout": "streaming", "key_chunk_size": 500},
    "attention-stream250": {"stage": "attention", "attention_layout": "streaming", "key_chunk_size": 250},
    "attention-stream750q250": {"stage": "attention", "attention_layout": "streaming", "key_chunk_size": 750, "query_chunk_size": 250},
    "mlp-token128": {"stage": "mlp"},
    "mlp-channel1024": {"stage": "mlp", "mlp_channel_chunk_size": 1024},
    "mlp-channel2048": {"stage": "mlp", "mlp_channel_chunk_size": 2048},
    "mlp-channel512": {"stage": "mlp", "mlp_channel_chunk_size": 512},
}


def balanced_scale(activation_maximum, weight, alpha):
    """Positive channel scales; multiplying both branches preserves the linear map."""
    import torch
    scale = activation_maximum.clamp_min(1e-5).pow(alpha) / weight.abs().amax(dim=0).clamp_min(1e-5).pow(1 - alpha)
    return scale / scale.median()


def rescale_norm_projection(weight, bias, projection, scale):
    return weight / scale, bias / scale, projection * scale[None, :]


def partition_feedforward_weights(first_weight, first_bias, second_weight, second_bias, outliers):
    """Partition complete hidden neurons; place the second bias in one branch."""
    import torch
    selected = torch.zeros(first_weight.shape[0], dtype=torch.bool)
    selected[outliers] = True
    regular = torch.where(~selected)[0]
    outliers = torch.where(selected)[0]
    return {
        "fc1_regular.weight": first_weight[regular], "fc1_regular.bias": first_bias[regular],
        "fc2_regular.weight": second_weight[:, regular], "fc2_regular.bias": second_bias,
        "fc1_outlier.weight": first_weight[outliers], "fc1_outlier.bias": first_bias[outliers],
        "fc2_outlier.weight": second_weight[:, outliers], "fc2_outlier.bias": torch.zeros_like(second_bias),
    }, regular


def split_ffn(graph, maxima, ranges, count, layers, scales):
    import numpy as np
    import torch
    # Training-only post-GELU maxima identify channels that widen FC2's range.
    for layer in layers:
        prefix = f"audio_tower.layers.{layer}"
        outliers = torch.from_numpy(np.argsort(maxima[prefix + ".fc2"])[-count:].copy())
        partition, regular = partition_feedforward_weights(*[graph.weight(prefix + name) for name in
            [".fc1.weight", ".fc1.bias", ".fc2.weight", ".fc2.bias"]], outliers)
        for name, values in partition.items():
            buffer = "partition_" + str(len(graph.names))
            graph.register_buffer(buffer, values.clone())
            graph.names[prefix + "." + name] = buffer
        for boundary in ["min/", "max/"]:
            ranges[boundary + prefix + ".fc1_regular"] = ranges[boundary + prefix + ".fc1"]
            ranges[boundary + prefix + ".fc2_regular"] = ranges[boundary + prefix + ".fc2"][regular.numpy()]
        if prefix + ".fc1" in scales:
            scales[prefix + ".fc1_regular"] = scales[prefix + ".fc1"]
        graph.split_ffn_layers.add(layer)


def selected_linear(source, scope):
    return source is not None and (scope == "conv" or source.endswith("." + scope)
        or scope == "mlp" and source.endswith((".fc1", ".fc2"))
        or scope == "regular" and source.endswith((".fc1_regular", ".fc2_regular")))


def smooth_graph(graph, maxima, alpha, scope, layers):
    import torch
    scales = {}
    with torch.inference_mode():
        for layer in layers:
            prefix = f"audio_tower.layers.{layer}"
            if scope in {"mlp", "fc1", "conv"}:
                name = prefix + ".fc1"
                scale = balanced_scale(torch.from_numpy(maxima[name]), graph.weight(name + ".weight"), alpha)
                norm = prefix + ".final_layer_norm"
                w, b, projection = rescale_norm_projection(graph.weight(norm + ".weight"), graph.weight(norm + ".bias"), graph.weight(name + ".weight"), scale)
                graph.weight(norm + ".weight").copy_(w)
                graph.weight(norm + ".bias").copy_(b)
                graph.weight(name + ".weight").copy_(projection)
                scales[name] = scale
            if scope in {"mlp", "fc2", "conv"}:
                name = prefix + ".fc2"
                scale = balanced_scale(torch.from_numpy(maxima[name]), graph.weight(name + ".weight"), alpha)
                graph.weight(name + ".weight").mul_(scale[None, :])
                buffer = f"balance_fc2_{layer}"
                graph.register_buffer(buffer, scale)
                graph.linear_input_scales[name] = buffer
                scales[name] = scale
            if scope == "conv":
                names = [prefix + f".self_attn.{key}_proj" for key in ["q", "k", "v"]]
                combined_weight = torch.cat([graph.weight(name + ".weight") for name in names])
                scale = balanced_scale(torch.from_numpy(maxima[names[0]]), combined_weight, alpha)
                norm = prefix + ".self_attn_layer_norm"
                graph.weight(norm + ".weight").div_(scale)
                graph.weight(norm + ".bias").div_(scale)
                for name in names:
                    graph.weight(name + ".weight").mul_(scale[None, :])
                    scales[name] = scale
                name = prefix + ".self_attn.out_proj"
                scale = balanced_scale(torch.from_numpy(maxima[name]), graph.weight(name + ".weight"), alpha)
                graph.weight(name + ".weight").mul_(scale[None, :])
                graph.weight(prefix + ".self_attn.v_proj.weight").div_(scale[:, None])
                graph.weight(prefix + ".self_attn.v_proj.bias").div_(scale)
                scales[name] = scale
    return {name: scale.numpy() for name, scale in scales.items()}


def quantize_inputs(model, graph, ranges, scales, scope, mode="linear", samples=None, channel_outputs=False, regular_output=False):
    import numpy as np
    import coremltools.optimize.coreml as optimization
    from coremltools.optimize.coreml import _post_training_quantization as api
    # Content identifies constant-folded convolution weights even when tracing
    # renames the original registered buffer. Cache repeated tiled weights.
    names = {}
    for name in graph.names:
        if name.endswith(".weight") and graph.weight(name).ndim == 2:
            values = np.ascontiguousarray(graph.weight(name).numpy(), dtype=np.float16)
            names[hashlib.sha256(values.tobytes()).hexdigest()] = name.removesuffix(".weight")
    weights, selected, stats = {}, [], {}
    for op in model._mil_program.functions["main"].operations:
        if op.op_type != "conv":
            continue
        if op.weight.name not in weights:
            weights[op.weight.name] = names.get(hashlib.sha256(np.ascontiguousarray(op.weight.val).tobytes()).hexdigest())
        source = weights[op.weight.name]
        if not selected_linear(source, scope):
            continue
        scale = scales.get(source, 1)
        stats[op.x.name] = {"rmin": float(np.min(ranges["min/" + source] / scale)), "rmax": float(np.max(ranges["max/" + source] / scale))}
        selected.append(op.name)
    if not selected:
        raise ValueError("No requested convolution weights matched the checkpoint")
    activation_config = optimization.OptimizationConfig(op_name_configs={name: optimization.OpLinearQuantizerConfig(mode=mode) for name in selected})
    if samples is not None:
        model = optimization.linear_quantize_activations(model, activation_config, samples)
    else:
        prefix = api.PASS_REGISTRY["compression::insert_prefix_quantize_dequantize_pair"]
        prefix.set_options([api.PassOption("config", activation_config), api.PassOption("activation_stats", stats)])
        model = api._model_utils._apply_graph_pass(model, [prefix], spec_version=api._SPECIFICATION_VERSION_IOS_17,
            pymil_load_func=api._milproto_to_pymil.load, skip_model_load=True)
    if channel_outputs or regular_output:
        from coremltools.converters.mil.mil import Builder as mb
        program = model._mil_program
        block = program.functions["main"]
        for op in list(block.operations):
            if op.op_type != "conv":
                continue
            source = names.get(hashlib.sha256(np.ascontiguousarray(op.weight.val).tobytes()).hexdigest())
            if not selected_linear(source, scope) or regular_output and not source.endswith(".fc1_regular"):
                continue
            lower = np.minimum(graph.output_min[source].numpy(), 0)
            upper = np.maximum(graph.output_max[source].numpy(), 0)
            if regular_output:
                # Saturated negative preactivations map to zero under FP16 GELU.
                lower, upper = np.maximum(np.min(lower), -6.0), np.max(upper)
            scale = np.maximum((upper-lower)/255, np.finfo(np.float16).smallest_subnormal).astype(np.float16)
            zero = np.clip(np.rint(-128-lower/scale.astype(np.float32)), -128, 127).astype(np.int8)
            next_op = list(block.operations)[list(block.operations).index(op)+1]
            with block:
                quantized = mb.quantize(input=op.outputs[0], scale=scale, zero_point=zero, output_dtype="int8", axis=None if regular_output else 1, before_op=next_op)
                dequantized = mb.dequantize(input=quantized, scale=scale, zero_point=zero, axis=None if regular_output else 1, before_op=next_op)
                block.replace_uses_of_var_after_op(anchor_op=dequantized.op, old_var=op.outputs[0], new_var=dequantized)
        model = api._model_utils._apply_graph_pass(model, [], spec_version=api._SPECIFICATION_VERSION_IOS_17,
            pymil_load_func=lambda *unused, **kw: program, skip_model_load=True)
    weight_config = optimization.OptimizationConfig(op_name_configs={name: optimization.OpLinearQuantizerConfig(mode="linear_symmetric", granularity="per_channel") for name in selected})
    model = optimization.linear_quantize_weights(model, weight_config)
    return model, dict(quantizedConvolutions=len(selected), inputOnly=samples is None and not channel_outputs and not regular_output,
        channelOutputs=channel_outputs, regularOutput=regular_output, mode=mode, ranges=stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variants", choices=list(VARIANTS), nargs="+", default=list(VARIANTS))
    parser.add_argument("--layer-index", type=int, choices=[0, 16, 31], default=0)
    args = parser.parse_args()
    import coremltools as ct
    import numpy as np
    from safetensors.numpy import load_file
    import torch
    torch.set_num_threads(8)
    activations = load_file(args.calibration / "activations.safetensors")[f"layer{args.layer_index}/heldout"]
    training = load_file(args.calibration / "activations.safetensors")[f"layer{args.layer_index}/train"]
    maxima = load_file(args.calibration / "activation-max.safetensors")
    ranges = load_file(args.calibration / "activation-ranges.safetensors")
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in args.variants:
        options = VARIANTS[name]
        graph = build_graph(args.checkpoint)
        defaults = dict(stage="block", channel_norm=True, normalization_scale=8.0, attention_layout="einsum", head_group_size=1,
            query_chunk_size=500, key_chunk_size=1500, mlp_chunk_size=128, mlp_channel_chunk_size=5120, layer_index=args.layer_index, normalization_layout="centered")
        for key, value in (defaults | options).items():
            setattr(graph, key, value)
        with torch.inference_mode():
            canonical = dict(attention_layout="sdpa", channel_norm=False, linear_operator=True,
                encoder_frames=1500, qkv_head_chunk_size=0, mlp_channel_chunk_size=5120, mlp_chunk_size=1500)
            configured = {key: getattr(graph, key) for key in canonical}
            for key, value in canonical.items():
                setattr(graph, key, value)
            reference = [graph(torch.from_numpy(x[None].astype(np.float32))).numpy() for x in activations]
            for key, value in configured.items():
                setattr(graph, key, value)
            scales = smooth_graph(graph, maxima, options["smooth"], "fc1" if options["scope"] == "regular" else options["scope"], [args.layer_index]) if "smooth" in options else {}
            if "outliers" in options:
                split_ffn(graph, maxima, ranges, options["outliers"], [args.layer_index], scales)
            smoothed = graph(torch.from_numpy(activations[:1].astype(np.float32))).numpy()
            identity_error = float(np.linalg.norm(smoothed-reference[0])/np.linalg.norm(reference[0]))
            if not np.isfinite(identity_error) or identity_error > 1e-4:
                raise ValueError(f"Channel rescaling changed the FP32 function: {identity_error}")
            if options.get("channel_outputs") or options.get("regular_output"):
                graph.collect_output_ranges = True
                for x in training:
                    graph(torch.from_numpy(x[None].astype(np.float32)))
                graph.collect_output_ranges = False
            example = torch.from_numpy(activations[:1].astype(np.float32))
            traced = torch.jit.trace(graph, example, check_trace=False)
        model = ct.convert(traced, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
            inputs=[ct.TensorType(name="x", shape=example.shape, dtype=np.float16)], outputs=[ct.TensorType(name="y", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
        quantization = None
        if "scope" in options:
            calibration_samples = [{"x": x[None]} for x in training] if options.get("both_sides") else None
            model, quantization = quantize_inputs(model, graph, ranges, scales, options["scope"], options.get("mode", "linear"), calibration_samples,
                options.get("channel_outputs", False), options.get("regular_output", False))
        artifact = args.output / (name + ".mlpackage")
        model.save(str(artifact))
        runtime = ct.models.MLModel(str(artifact), compute_units=ct.ComputeUnit.CPU_AND_NE)
        inputs = {"x": activations[:1]}
        for _ in range(2):
            runtime.predict(inputs)
        seconds = []
        for _ in range(9):
            started = time.perf_counter()
            runtime.predict(inputs)
            seconds.append(time.perf_counter()-started)
        errors = []
        for x, expected in zip(activations, reference):
            predicted = runtime.predict({"x": x[None]})["y"].astype(np.float64)
            expected = expected.astype(np.float64)
            errors.append(dict(relativeRMSE=float(np.linalg.norm(expected-predicted)/np.linalg.norm(expected)),
                cosine=float(np.sum(expected*predicted)/np.linalg.norm(expected)/np.linalg.norm(predicted)), finite=bool(np.isfinite(predicted).all())))
        report = dict(name=name, options=defaults | options, float32RescalingRelativeError=identity_error,
            referenceKind="original-fp32-sdpa",
            quantization=quantization, predictionSeconds=seconds, medianSeconds=statistics.median(seconds), heldOutErrors=errors,
            anticipatedDevices=placement(runtime.get_compiled_model_path()))
        reports.append(report)
        (args.output / "profile.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps({k:v for k,v in report.items() if k != "quantization"}), flush=True)
        del runtime, model, graph


if __name__ == "__main__":
    main()
