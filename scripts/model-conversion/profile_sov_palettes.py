#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "coremltools==9.0", "faiss-cpu==1.16.0", "scipy==1.18.1"]
# ///
"""Compare scalar palette training on SOV weights, including official e-means.

Timings cover palette training and final nearest-centroid assignment. They are
offline compression costs, not model prediction latency. Core ML uses weighted
optimal one-dimensional clustering over unique FP16 values when kmeans1d is
available; compare that path as well as conventional Lloyd iterations.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import time
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-values", type=int, default=1000000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 31, 43])
    args = parser.parse_args()
    import faiss
    from faiss.contrib.e_means import NumpyEmeans
    import numpy as np
    # Load Core ML's weighted kmeans1d module without its top-level imports.
    # Importing coremltools imports PyTorch and its conflicting OpenMP runtime.
    package = importlib.metadata.distribution("coremltools").locate_file("coremltools/_deps/kmeans1d/__init__.py")
    spec = importlib.util.spec_from_file_location("sov_weighted_kmeans1d", package)
    kmeans1d = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kmeans1d
    spec.loader.exec_module(kmeans1d)
    software = {name: importlib.metadata.version(name) for name in ["faiss-cpu", "coremltools", "numpy", "scipy"]}
    implementation_hashes = {str(path.relative_to(Path(faiss.__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(faiss.__file__).parent / "contrib/e_means/e_means.py", Path(faiss.__file__).parent / "contrib/e_means/base.py"]}
    faiss.omp_set_num_threads(8)
    shard = args.checkpoint / "model-00016-of-00016.safetensors"
    with shard.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != "438cd592fe6fc231d4f10dad756cfed5ab60964ea0021155ca136792885f6b5d":
        raise ValueError("Source checkpoint checksum differs from the pinned SOV release")
    # Decode the two pinned BF16 arrays without loading PyTorch: its separate
    # OpenMP runtime conflicts with Faiss on macOS.
    with shard.open("rb") as source:
        header_size = int.from_bytes(source.read(8), "little")
        header = json.loads(source.read(header_size))
    reports = []
    for layer in [0, 16]:
        tensor_name = f"model.audio_tower.layers.{layer}.fc1.weight"
        tensor = header[tensor_name]
        if tensor["dtype"] != "BF16":
            raise ValueError(f"Expected BF16 source: {tensor_name}")
        start, end = tensor["data_offsets"]
        encoded = np.memmap(shard, mode="r", dtype="<u2", offset=8 + header_size + start, shape=((end - start) // 2,))
        weights = (encoded.astype(np.uint32) << 16).view(np.float32)
        rng = np.random.default_rng(17)
        sampled = weights[rng.choice(len(weights), min(args.max_values, len(weights)), replace=False)]
        train = np.ascontiguousarray(sampled[:int(len(sampled) * 0.8), None])
        test = np.ascontiguousarray(sampled[int(len(sampled) * 0.8):, None])
        for bits in [8, 6]:
            for algorithm in ["weighted-kmeans1d", "faiss-lloyd25", "faiss-emeans3", "faiss-emeans5"]:
                for seed in args.seeds:
                    started = time.perf_counter()
                    if algorithm == "weighted-kmeans1d":
                        values, counts = np.unique(train, return_counts=True)
                        result = kmeans1d.cluster(values, 1 << bits, weights=counts)
                        centroids = np.asarray(result.centroids, dtype=np.float32)[:, None]
                    elif algorithm == "faiss-lloyd25":
                        model = faiss.Kmeans(1, 1 << bits, niter=25, seed=seed, max_points_per_centroid=len(train))
                        model.train(train)
                        centroids = model.centroids
                    else:
                        model = NumpyEmeans(1, 1 << bits, nepochs=3 if algorithm.endswith("3") else 5, seed=seed)
                        centroids, _ = model.train(train)
                    # Score the actual FP16 palette Core ML would store.
                    centroids = np.sort(centroids.ravel().astype(np.float16).astype(np.float32))
                    def errors(data):
                        indices = np.searchsorted((centroids[:-1] + centroids[1:]) * 0.5, data.ravel())
                        return float(np.mean((data.ravel().astype(np.float64) - centroids[indices]) ** 2))
                    train_mse, test_mse = errors(train), errors(test)
                    seconds = time.perf_counter() - started
                    report = dict(tensor=tensor_name, bits=bits, algorithm=algorithm, seed=seed,
                        trainCount=len(train), testCount=len(test), seconds=seconds,
                        trainMSE=train_mse, heldOutMSE=test_mse,
                        heldOutRelativeRMSE=float(np.sqrt(test_mse / np.mean(test.astype(np.float64) ** 2))))
                    reports.append(report)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(dict(software=software, implementationSHA256=implementation_hashes,
                        benchmarkSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), sourceSHA256=digest,
                        features="scalar BF16 weights represented exactly as FP16", measurements=reports), indent=2) + "\n")
                    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
