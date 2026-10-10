#!/usr/bin/env python3
"""Export independent Whistle FP32 parity fixtures; never downloads assets.

Requires numpy, onnx, onnxruntime. Graphs are validation inputs only.
See Tests/SpeechRuntimeTests/Fixtures/Whistle/README.md for provenance.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


def verified_bytes(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Unexpected asset SHA-256: {path}")
    return data


def session(path, digest, weights):
    graph = onnx.load_model_from_string(verified_bytes(path, digest))
    initializers = []
    for value in graph.graph.initializer:
        if value.name in ("perm1", "perm2"):
            seed = 11 if value.name == "perm1" else 13
            array = np.random.RandomState(seed).permutation(512).astype(np.int64)
        else:
            name = value.name[2:].replace("__", "/")
            array = weights[name]
            if name.endswith("/kernel") or "/mhc_phi_" in name:
                array = array.swapaxes(-1, -2)
        if list(array.shape) != list(value.dims):
            raise ValueError(f"Shape mismatch: {value.name}")
        initializers.append(numpy_helper.from_array(array, value.name))
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(initializers)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(graph.SerializeToString(), options, providers=["CPUExecutionProvider"])


def reference(weights, encoder, decoder):
    mel = (np.sin(np.arange(32 * 80) * .037) * .8).astype(np.float32).reshape(32, 80)
    keys, values = encoder.run(None, {"mel": mel})
    previous_shape = next(item.shape for item in decoder.get_inputs() if item.name == "prev")
    feeds = dict(efeat=np.zeros((2, 4, 512), np.float32),
                 prev=np.zeros(previous_shape, np.float32),
                 past_k=np.zeros((8, 2, 0, 48), np.float32),
                 past_v=np.zeros((8, 2, 0, 64), np.float32), cross_k=keys, cross_v=values)
    results, tokens = [], []
    for position, token in enumerate([2, 8192, 234, 456]):
        tokens.append(token)
        features = np.zeros((2, 4, 512), np.float32)
        for tap in range(4):
            p = position - 3 * tap
            if p < 0:
                continue
            for order_index, order in enumerate([2, 3]):
                for head in range(2):
                    table = order_index * 2 + head
                    acc = (0x9e3779b9 * (table + 1)) & 0xffffffff
                    for index in range(order):
                        old_token = tokens[p - index] if p >= index else 0
                        acc = ((acc ^ old_token) * 0x01000193) & 0xffffffff
                    acc ^= acc >> 15
                    if p >= order - 1:
                        for site in range(2):
                            features[site, tap, table * 128:(table + 1) * 128] = \
                                weights[f"engrams_{site}/embedding"][table, acc % 18432]
        feeds.update(pos=np.array([position], np.int64), token=np.array([token], np.int64), efeat=features)
        logits, previous, cached_keys, cached_values, _ = decoder.run(None, feeds)
        feeds.update(prev=previous, past_k=cached_keys, past_v=cached_values)
        results.append(logits.ravel())
    return dict(mel=mel, keys=keys, values=values, logits=np.stack(results))


def write_safetensors(path, arrays, layout):
    header = {"__metadata__": {"layout": json.dumps(layout, separators=(",", ":"))}}
    payload, offset = [], 0
    for name, array in arrays.items():
        data = array.astype("<f4").tobytes()
        header[name] = dict(dtype="F32", shape=list(array.shape), data_offsets=[offset, offset + len(data)])
        payload.append(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--cact", type=Path, help="Compare the packed archive after independent NumPy dequantization")
    args = parser.parse_args()
    data = verified_bytes(args.checkpoint, "5fd58c246f522ecee2b568b306598a3befdc274c225e3a7376abce277ab3d107")
    length, = struct.unpack_from("<Q", data)
    header = json.loads(data[8:8 + length])
    weights = {}
    for name, item in header.items():
        if name == "__metadata__":
            continue
        if item["dtype"] != "F32":
            raise ValueError(f"Expected FP32: {name}")
        start, end = item["data_offsets"]
        weights[name] = np.frombuffer(data, dtype="<f4", offset=8 + length + start,
                                     count=(end - start) // 4).reshape(item["shape"])
    layout = {name: list(array.shape) for name, array in weights.items()}
    if args.synthetic:
        for index, name in enumerate(sorted(weights)):
            pattern = (np.sin(np.arange(97, dtype=np.float64) * .1 + index * .3) * .02).astype(np.float32)
            weights[name] = np.resize(pattern, weights[name].shape)
    if args.cact:
        if args.synthetic:
            raise ValueError("Use either synthetic weights or the released cact")
        cact = verified_bytes(args.cact, "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb")
        mapping_path = Path(__file__).resolve().parents[2] / "Sources/AudioWhistleModel/Resources/cact-layout.json"
        mapping = json.loads(mapping_path.read_text())
        decoded = {}
        for name, item in mapping.items():
            pieces = []
            for piece in item["pieces"]:
                record = piece["record"]
                if record not in decoded:
                    r = struct.unpack_from("<BBHIIIIQQII", cact, 308 + record * 44)
                    dtype, ndim = r[:2]
                    shape, offset, size, group, bits = r[3:3 + ndim], *r[7:]
                    if dtype == 3:
                        rows, cols = shape
                        packed_count = rows * cols * bits // 8
                        packed = np.frombuffer(cact, np.uint8, count=packed_count, offset=offset)
                        shifts = np.arange(8 // bits, dtype=np.uint8) * bits
                        indices = ((packed[:, None] >> shifts) & ((1 << bits) - 1)).reshape(rows, cols)
                        book_offset = 196 + (0 if bits == 2 else 12) * 4
                        book = np.frombuffer(cact, "<f4", count=1 << bits, offset=book_offset)
                        norms = np.frombuffer(cact, "<f2", count=rows * cols // group,
                                              offset=offset + packed_count).astype(np.float32).reshape(rows, -1, 1)
                        values = book[indices].reshape(rows, -1, group).astype(np.float64) * norms
                        width = 1
                        while width < group:
                            pairs = values.reshape(-1, 2, width)
                            a, b = pairs[:, 0].copy(), pairs[:, 1].copy()
                            pairs[:, 0], pairs[:, 1] = a + b, a - b
                            width *= 2
                        decoded[record] = (values.reshape(rows, cols) / np.sqrt(group)).astype(np.float32)
                    else:
                        decoded[record] = np.frombuffer(cact, "<f2" if dtype == 1 else "<f4",
                                                       count=int(np.prod(shape)), offset=offset).reshape(shape).astype(np.float32)
                pieces.append(decoded[record])
            value = np.stack(pieces) if len(pieces) > 1 else pieces[0]
            if item["transpose"]:
                if value.ndim == 2 and name.count("/mhc_phi_"):
                    value = value.reshape(8, layout[name][-1], layout[name][-2])
                value = value.swapaxes(-1, -2)
            weights[name] = value.reshape(layout[name])
    encoder = session(args.encoder, "feda0a9192f65921a30f94d37bcfce533a2545a1fc3415639ae83aeee4800ae4", weights)
    decoder = session(args.decoder, "d2309759f3db92a23cefb8265132b744dd7ca602cb9c30902451a8b2587d3a75", weights)
    arrays = reference(weights, encoder, decoder)
    if args.synthetic:
        write_safetensors(args.output, arrays, layout)
    else:
        result = {name: array.ravel().tolist() for name, array in arrays.items() if name != "logits"}
        result["logits"] = arrays["logits"].tolist()
        args.output.write_text(json.dumps(result))
    print(args.output)


if __name__ == "__main__":
    main()
