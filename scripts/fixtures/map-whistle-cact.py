#!/usr/bin/env python3
"""Regenerate the pinned .cact name map from byte-identical public WebGPU segments.

Requires only Python's standard library. The pack is a validation/provenance input,
never a runtime dependency. Source: huggingface.co/spaces/mrfakename/whistle-webgpu.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def checked(path, digest):
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != digest:
        raise ValueError(f"Unexpected SHA-256: {path}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cact", required=True, type=Path)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cact = checked(args.cact, "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb")
    pack = checked(args.pack, "1edf33dcf5772dad142d6f19571371f98999053a7a3b4397364bea0d8f4679d0")
    length, = struct.unpack_from("<I", pack, 4)
    manifest = json.loads(pack[8:8 + length])
    base = (8 + length + 15) // 16 * 16
    records = [struct.unpack_from("<BBHIIIIQQII", cact, 308 + i * 44) for i in range(681)]
    blobs = {}
    for index, record in enumerate(records):
        blobs.setdefault(cact[record[7]:record[7] + record[8]], []).append(index)
    layout_path = Path(__file__).resolve().parents[2] / "Sources/AudioWhistleModel/Resources/layout.json"
    layout = json.loads(layout_path.read_text())
    output = {}
    for model in manifest["models"].values():
        for tensor in model["inits"]:
            name = tensor["name"][2:].replace("__", "/")
            if not tensor["name"].startswith("p.") or name not in layout:
                continue
            pieces = []
            for segment in tensor["segs"]:
                start = base + segment["src"]
                if segment["k"] == "cq":
                    ids = blobs[pack[start:start + segment["n"]]]
                    pieces.append(dict(record=ids[0], rows=segment["rows"]))
                else:
                    size = 2 if segment["dt"] == "f16" else 4
                    data = pack[start:start + segment["n"] * size]
                    count = 8 if name.startswith(("encoder/layers/", "stack/layers/")) else 1
                    assert len(data) % count == 0
                    chunk = len(data) // count
                    for index in range(count):
                        pieces.append(dict(record=blobs[data[index * chunk:(index + 1) * chunk]][0]))
            entry = dict(pieces=pieces, transpose=name.endswith("/kernel") or "/mhc_phi_" in name)
            if name in output:
                assert output[name] == entry, name
            output[name] = entry
    for site, segment in enumerate(manifest["engram"]):
        start = base + segment["src"]
        record = blobs[pack[start:start + segment["n"]]][0]
        output[f"engrams_{site}/embedding"] = dict(pieces=[dict(record=record, rows=segment["rows"])], transpose=False)
    assert set(output) == set(layout)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
