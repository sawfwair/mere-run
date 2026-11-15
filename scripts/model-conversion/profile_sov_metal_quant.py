#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mlx==0.32.4", "numpy==2.3.5", "torch==2.7.0", "safetensors==0.8.0"]
# ///
"""Prototype custom M5 int8 matrix arithmetic with live activation scales.

No checkpoint parameters are changed. Offline weight packing is timed separately;
prediction includes activation scaling, packing, TensorOps, and output conversion.
GPU tests compare against BF16 MLX addmm, not a dequantized reference alone.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import mlx.core as mx
import numpy as np


KERNEL_PATH = Path(__file__).resolve().parents[2]/"Sources/AudioSOVTranscribeModel/Kernels/SOVDynamicInt8.metal"


class DynamicInt8:
    def __init__(self, block=128, exponent=2, integer=True, tile_major=False, rotate=False, fused=False, columns=32):
        if columns not in (32,64):
            raise ValueError("TensorOps columns must be 32 or 64.")
        self.columns = columns
        self.block, self.exponent, self.integer = block, exponent, integer
        self.tile_major = tile_major
        self.rotate = rotate
        self.fused = fused
        text = KERNEL_PATH.read_text()
        header, bodies = text.split("// ROTATE_BODY\n")
        rotate_body, bodies = bodies.split("// FUSED_PACK_BODY\n")
        fused_body, bodies = bodies.split("// PACK_BODY\n")
        pack, mm = bodies.split("// MATMUL_BODY\n")
        if columns == 64:
            mm = mm.replace('N / 64', 'N / 128').replace('(tile % (N / 128)) * 64 + (sg % 2) * 32', '(tile % (N / 128)) * 128 + (sg % 2) * 64')
            mm = mm.replace('32, 32, BK,', '32, 64, BK,').replace('float totals[32]', 'float totals[64]').replace('int previous[32]', 'int previous[64]')
            mm = mm.replace('extents<int, BK, 32>{}, array<int, 2>{1, TileMajor ? BK : K});\nconstexpr', 'extents<int, BK, 64>{}, array<int, 2>{1, TileMajor ? BK : K});\nconstexpr')
            mm = mm.replace('extents<int, BK, 32>{}, array<int, 2>{1, TileMajor ? BK : K});\n    operation.run', 'extents<int, BK, 64>{}, array<int, 2>{1, TileMajor ? BK : K});\n    operation.run')
        self.effective_matmul_sha256 = hashlib.sha256((header+mm).encode()).hexdigest()
        self.rotate_kernel = mx.fast.metal_kernel("sov_dynamic_hadamard", ["input"], ["rotated"], rotate_body, header=header)
        self.fused_kernel = mx.fast.metal_kernel("sov_dynamic_fused_pack", ["input"], ["codes", "row_scale", "exponents"], fused_body, header=header)
        self.pack_kernel = mx.fast.metal_kernel("sov_dynamic_int8_pack", ["input", "row_scale"], ["codes", "exponents"], pack, header=header)
        self.mm_kernel = mx.fast.metal_kernel("sov_dynamic_int8_mpp", ["aq", "a_scale", "a_exp", "bq", "b_scale", "b_exp", "bias"], ["output"], mm, header=header)

    def pack(self, x):
        k = x.shape[-1]
        rows = x.size//k
        if k % self.block or (self.tile_major and rows % 64):
            raise ValueError("Packing requires complete K blocks and 64-row tiles.")
        if self.fused and k//self.block > 32:
            raise ValueError("Fused packing supports at most 32 SIMDgroups per row.")
        if self.integer and (self.exponent > 2 or k > 8192):
            raise ValueError("Integer accumulator requires exponent <= 2 and K <= 8192.")
        if self.fused:
            q,s,e = self.fused_kernel(inputs=[x],template=[("K",k),("BK",self.block),("Rotate",self.rotate),("TileMajor",self.tile_major),("MaxExponent",self.exponent)],
                grid=(rows*(k//self.block)*32,1,1),threadgroup=((k//self.block)*32,1,1),
                output_shapes=[x.shape,(rows,),(rows,k//self.block)],output_dtypes=[mx.int8,mx.float32,mx.uint8])
            return q,s,e
        if self.rotate:
            x, = self.rotate_kernel(inputs=[x], template=[("BK", self.block)],
                grid=(rows*k//self.block*32,1,1),threadgroup=(32,1,1),output_shapes=[x.shape],output_dtypes=[mx.float32])
        scale = mx.maximum(mx.max(mx.abs(x), axis=-1).astype(mx.float32)/127, 1e-12)
        q, e = self.pack_kernel(inputs=[x, scale], template=[("K", k), ("BK", self.block), ("MaxExponent", self.exponent), ("TileMajor", self.tile_major)],
            grid=(rows*(k//self.block)*32, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[x.shape, (rows, k//self.block)], output_dtypes=[mx.int8, mx.uint8])
        return q, scale, e

    def matmul(self, x, packed, bias):
        m, k = x.shape
        n = packed[0].shape[0]
        if n % (2*self.columns):
            raise ValueError("Output width must contain complete column tiles.")
        padded = ((m+63)//64)*64
        if padded != m:
            x = mx.concatenate([x, mx.zeros((padded-m, k), dtype=x.dtype)])
        aq, a_scale, a_exp = self.pack(x)
        bq, b_scale, b_exp = packed
        out, = self.mm_kernel(inputs=[aq, a_scale, a_exp, bq, b_scale, b_exp, bias],
            template=[("M", padded), ("N", n), ("K", k), ("BK", self.block), ("IntegerAccumulation", self.integer), ("MaxExponent", self.exponent), ("TileMajor", self.tile_major), ("Out", x.dtype)],
            grid=((padded//64)*(n//(2*self.columns))*128, 1, 1), threadgroup=(128, 1, 1),
            output_shapes=[(padded, n)], output_dtypes=[x.dtype])
        return out[:m]


def paired(baseline, candidate, x, repetitions):
    functions = [mx.compile(baseline), mx.compile(candidate)]
    end = time.perf_counter()+1.0
    while time.perf_counter() < end:
        for fn in functions: mx.eval(fn(x))
    samples = [[], []]
    results = [None, None]
    for iteration in range(repetitions):
        for index in ([0, 1] if iteration%2 == 0 else [1, 0]):
            start = time.perf_counter()
            results[index] = functions[index](x)
            mx.eval(results[index])
            samples[index].append(time.perf_counter()-start)
    return results[0], results[1], samples[0], samples[1]


def metrics(actual, expected):
    a, b = np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32))
    a, b = a.astype(np.float64), b.astype(np.float64)
    return dict(relativeRMSE=float(np.linalg.norm(a-b)/np.linalg.norm(b)), cosine=float(np.sum(a*b)/np.linalg.norm(a)/np.linalg.norm(b)), finite=bool(np.isfinite(a).all()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--stage", choices=["fc1", "fc2"], default="fc1")
    parser.add_argument("--columns",type=int,choices=[32,64],default=32)
    parser.add_argument("--block", type=int, choices=[64, 128, 256], default=128)
    parser.add_argument("--exponent", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--float-accumulation", action="store_true")
    parser.add_argument("--tile-major", action="store_true")
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--fused-pack", action="store_true")
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    if args.checkpoint:
        from safetensors import safe_open
        from safetensors.numpy import load_file
        import torch
        torch.set_num_threads(4)
        prefix = f"model.audio_tower.layers.{args.layer}."
        with safe_open(args.checkpoint/"model-00016-of-00016.safetensors", framework="pt") as f:
            def get(name): return mx.array(f.get_tensor(prefix+name).float().numpy()).astype(mx.bfloat16)
            w, bias = get(args.stage+".weight"), get(args.stage+".bias")
            a = load_file(args.calibration/"activations.safetensors")[f"layer{args.layer}/heldout"][0].reshape(1280, 1500).T.copy()
            x = mx.fast.layer_norm(mx.array(a).astype(mx.bfloat16), get("final_layer_norm.weight"), get("final_layer_norm.bias"), eps=1e-5)
            if args.stage == "fc2":
                z = mx.addmm(get("fc1.bias"), x, get("fc1.weight").T)
                x = mx.compile(lambda t: t*(1+mx.erf(t/2**0.5))/2)(z)
    else:
        m,k,n = (67,256,128) if args.small else (1500,1280,5120)
        mx.random.seed(17)
        x = mx.random.normal((m,k)).astype(mx.bfloat16)
        w = (mx.random.normal((n,k))/k**0.5).astype(mx.bfloat16)
        bias = mx.random.normal((n,)).astype(mx.bfloat16)*0.1
    mx.eval(x,w,bias)
    operator = DynamicInt8(args.block, args.exponent, not args.float_accumulation, args.tile_major, args.rotate, args.fused_pack, args.columns)
    start = time.perf_counter(); packed = operator.pack(w); mx.eval(packed)
    packing_seconds = time.perf_counter()-start
    expected, actual, baseline, measured = paired(lambda a: mx.addmm(bias,a,w.T), lambda a: operator.matmul(a,packed,bias), x, args.repetitions)
    receipt = dict(columns=args.columns,effectiveMatmulSHA256=operator.effective_matmul_sha256,mlx=mx.__version__, shape=[x.shape[0],w.shape[0],x.shape[1]], block=args.block, exponent=args.exponent,
        integerAccumulation=not args.float_accumulation, layer=args.layer,stage=args.stage,
        tileMajor=args.tile_major,
        hadamardRotation=args.rotate,
        fusedPacking=args.fused_pack,
        kernelSHA256=hashlib.sha256(KERNEL_PATH.read_bytes()).hexdigest(),
        packingSeconds=packing_seconds, baselineSeconds=baseline, customSeconds=measured,
        baselineMedianSeconds=statistics.median(baseline), customMedianSeconds=statistics.median(measured),
        speedup=statistics.median(baseline)/statistics.median(measured), errors=metrics(actual,expected))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True)


if __name__ == "__main__": main()
