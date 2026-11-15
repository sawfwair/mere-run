#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["aneforge==0.4.0", "numpy==2.3.5", "torch==2.7.0", "safetensors==0.8.0"]
# ///
"""Research native ANE fused attention with full SOV key/value context.

Uses ANEForge's pinned private bridge, outside the shipping runtime. Query tiles
avoid the bridge's documented 512-by-512 reliability cliff. Its standard
persistent worker assumes equal Q/K lengths; this experiment applies the same
asymmetric-shape adjustment as its subprocess bridge to a persistent worker.
Native-bridge timings include Python, pipe transfer, transposes, and execution.
The e5rt comparison includes its in-process dispatch and complete attention
sublayer, including LayerNorm and the four original-weight projections.
"""
import argparse
import json
from pathlib import Path
import plistlib
import statistics
import tempfile
import time

import numpy as np
import torch
from safetensors.numpy import load_file

from convert_sov_coreml import build_graph


def worker(heads, queries, keys, dimension):
    from aneforge._bridges import ane_sdpa_fused
    from aneforge._netplist_worker import _Worker
    directory = tempfile.TemporaryDirectory(prefix="sov_native_sdpa_")
    netplist, weights = ane_sdpa_fused._write_netplist(
        Path(directory.name), channels=keys, sequence=heads, dim=dimension,
        scale=dimension ** -0.5, constant_flag_spelling="Constants_array", subtract_max=True)
    specification = plistlib.loads(netplist.read_bytes())
    for procedure in specification["ProcedureList"]:
        for entry in procedure.get("InputList", []):
            if entry["InputName"] == "query":
                entry["InputChannels"] = queries
        for entry in procedure.get("OutputList", []):
            entry["OutputChannels"] = queries
    netplist.write_bytes(plistlib.dumps(specification))
    return _Worker(netplist, list(weights), ["query", "key", "value"], ["y"], directory, op="sov-sdpa")


def e5rt_attention(graph, prefix, chunk, delayed):
    import aneforge as af
    x = af.input([1, 1280, 1, 1500])

    def weight(name):
        return graph.weight(prefix + name).numpy()

    def linear(x, name, bias=True):
        return af.conv(x, weight(name + ".weight")[:, :, None, None],
            bias=weight(name + ".bias") if bias else None)

    normalized = (x / 8).channel_layer_norm(weight(".self_attn_layer_norm.weight"),
        weight(".self_attn_layer_norm.bias"), eps=1e-5/64)
    q, k, v = [af.split(linear(normalized, f".self_attn.{name}_proj", name != "k"), 20, axis=1) for name in ["q", "k", "v"]]
    heads = []
    for head in range(20):
        pieces = []
        for query in af.split(q[head], 1500//chunk, axis=3):
            scores = af.einsum("bchq,bchk->bkhq", query, k[head]) * 0.125
            if delayed:
                exponential = (scores - scores.amax(1)).exp()
                value = af.einsum("bkhq,bchk->bchq", exponential, v[head]) / exponential.sum(1)
            else:
                value = af.einsum("bkhq,bchk->bchq", scores.softmax(1), v[head])
            pieces.append(value)
        heads.append(af.concat(pieces, axis=3))
    output = linear(af.concat(heads, axis=1), ".self_attn.out_proj")
    return af.compile(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--query-chunk", type=int, choices=[125, 250, 500], default=500)
    parser.add_argument("--head-group", type=int, choices=[1, 4, 20], default=20)
    parser.add_argument("--key-frames", type=int, choices=[250, 1500], default=1500, help="250 is a runtime control only; SOV qualification requires 1500")
    parser.add_argument("--layer-index", type=int, choices=[0, 16, 31], default=0)
    parser.add_argument("--backend", choices=["native", "e5rt", "e5rt-delayed"], default="native")
    args = parser.parse_args()
    if args.key_frames != 1500 and args.backend != "native":
        parser.error("--key-frames 250 is only a native bridge runtime control")
    torch.set_num_threads(8)
    graph = build_graph(args.checkpoint)
    prefix = f"audio_tower.layers.{args.layer_index}"
    activations = load_file(args.calibration / "activations.safetensors")[f"layer{args.layer_index}/heldout"]
    samples = []
    with torch.inference_mode():
        for activation in activations:
            x = graph.norm(torch.from_numpy(activation[None].astype(np.float32)), prefix + ".self_attn_layer_norm")
            q, k, v = [graph.linear(x, prefix + f".self_attn.{name}_proj", name != "k").reshape(1, 20, 64, 1500).transpose(-1, -2) for name in ["q", "k", "v"]]
            k, v = k[:, :, :args.key_frames], v[:, :, :args.key_frames]
            expected = torch.nn.functional.scaled_dot_product_attention(q, k, v).numpy()
            samples.append(([a.numpy().astype(np.float16) for a in [q, k, v]], expected))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        runtime = worker(args.head_group, args.query_chunk, args.key_frames, 64) if args.backend == "native" else e5rt_attention(graph, prefix, args.query_chunk, args.backend == "e5rt-delayed")
    except RuntimeError as error:
        args.output.write_text(json.dumps(dict(backend=args.backend, queryChunk=args.query_chunk,
            headGroup=args.head_group, keyFrames=args.key_frames, layerIndex=args.layer_index,
            status="compile_failed", error=str(error)), indent=2) + "\n")
        raise

    def predict(arrays):
        q, k, v = arrays
        heads = []
        for head in range(0, 20, args.head_group):
            key = np.ascontiguousarray(k[:, head:head + args.head_group].transpose(0, 2, 1, 3))
            value = np.ascontiguousarray(v[:, head:head + args.head_group].transpose(0, 2, 1, 3))
            pieces = []
            for start in range(0, 1500, args.query_chunk):
                query = np.ascontiguousarray(q[:, head:head + args.head_group, start:start + args.query_chunk].transpose(0, 2, 1, 3))
                y = runtime.eval([query, key, value])[0].reshape(1, args.query_chunk, args.head_group, 64)
                pieces.append(y.transpose(0, 2, 1, 3))
            heads.append(np.concatenate(pieces, axis=2))
        return np.concatenate(heads, axis=1)

    if args.backend != "native":
        with torch.inference_mode():
            samples = [(x[None], graph.attention(torch.from_numpy(x[None].astype(np.float32)), args.layer_index).numpy()) for x in activations]
        predict = runtime

    try:
        for _ in range(2):
            predict(samples[0][0])
        seconds = []
        for _ in range(9):
            started = time.perf_counter()
            predict(samples[0][0])
            seconds.append(time.perf_counter() - started)
        errors = []
        for arrays, expected in samples:
            actual = predict(arrays).astype(np.float64)
            expected = expected.astype(np.float64)
            errors.append(dict(relativeRMSE=float(np.linalg.norm(expected-actual)/np.linalg.norm(expected)),
                cosine=float(np.sum(expected*actual)/np.linalg.norm(expected)/np.linalg.norm(actual)), finite=bool(np.isfinite(actual).all())))
        report = dict(backend="ANEForge 0.4.0 " + args.backend, queryChunk=args.query_chunk,
            headGroup=args.head_group, keyFrames=args.key_frames, layerIndex=args.layer_index,
            stage="attention-kernel" if args.backend == "native" else "attention-sublayer",
            predictionSeconds=seconds, medianSeconds=statistics.median(seconds), heldOutErrors=errors)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        runtime.release()


if __name__ == "__main__":
    main()
