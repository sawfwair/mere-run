#!/usr/bin/env python3
"""Export an existing native Q8 cache as a portable, standalone Hub bundle.

Uses only the Python standard library. Tensor payloads are copied in bounded
chunks without importing upstream Python, MLX, NumPy, or Transformers.
"""

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
from pathlib import Path

SOURCE = "salesforce-sovereign-intelligence/SOV-Transcribe-1"
REVISION = "505d8fae243924c0d8acdc622f7f0abab344eba3"
DRAFT = SOURCE + "-DFlash"
DRAFT_REVISION = "8c0b3f3a98507d2e3dd35b1bee0f147fa3bded05"
PACKER = "affine-fused-gate-up-v1"
DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "U32": "uint32"}
ASSETS = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "special_tokens_map.json", "generation_config.json", "processor_config.json", "preprocessor_config.json")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def header(path):
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        if not 0 < length <= min(path.stat().st_size - 8, 100_000_000):
            raise ValueError(f"Invalid safetensors header: {path.name}")
        return json.loads(handle.read(length)), 8 + length


def metadata_hash(path):
    tensors, offset = header(path)
    lines = []
    for name, tensor in sorted(tensors.items()):
        if name == "__metadata__":
            continue
        start, end = tensor["data_offsets"]
        if not 0 <= start <= end <= path.stat().st_size - offset:
            raise ValueError(f"Invalid tensor range: {name}")
        lines.append(f'{name}:{tensor["shape"]}:{DTYPES[tensor["dtype"]]}:{offset + start}:{offset + end}')
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def named(root, name):
    if not name or Path(name).name != name or "\\" in name:
        raise ValueError(f"Invalid shard filename: {name}")
    return root / name


def validate_cached(root, record):
    path = named(root, record["name"])
    stat = path.stat()
    if (stat.st_size != record["bytes"] or
            abs(stat.st_mtime - 978307200 - record["modified"]) > 0.00001 or
            metadata_hash(path) != record["headerSHA256"]):
        raise ValueError(f"Stale or changed cache/source shard: {path.name}")


def portable_record(path):
    return {"name": path.name, "bytes": path.stat().st_size,
            "headerSHA256": metadata_hash(path), "sha256": sha256(path)}


def copy_audio(source, files, destination):
    """Keep original audio tensor names, layouts, dtypes and bytes."""
    selected = {}
    output_header = {}
    output_offset = 0
    for record in files:
        path = named(source, record["name"])
        tensors, offset = header(path)
        for name, tensor in sorted(tensors.items()):
            if not name.startswith(("model.audio_tower.", "model.embed_audio.")):
                continue
            if name in selected:
                raise ValueError(f"Duplicate audio tensor: {name}")
            start, end = tensor["data_offsets"]
            selected[name] = (path, offset + start, end - start)
            output_header[name] = dict(tensor, data_offsets=[output_offset, output_offset + end - start])
            output_offset += end - start
    if not all(any(name.startswith(prefix) for name in selected)
               for prefix in ("model.audio_tower.", "model.embed_audio.")):
        raise ValueError("Missing audio encoder or projector tensors")
    encoded = json.dumps(output_header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with destination.open("xb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for path, offset, count in selected.values():
            with path.open("rb") as handle:
                handle.seek(offset)
                while count:
                    chunk = handle.read(min(count, 8 * 1024 * 1024))
                    if not chunk:
                        raise ValueError(f"Truncated audio tensor in {path.name}")
                    output.write(chunk)
                    count -= len(chunk)
    return len(selected)


def export(source, packed, draft, destination, runtime_revision):
    if destination.exists():
        raise ValueError("Output already exists; select a new bundle directory")
    if len(runtime_revision) != 40 or any(c not in "0123456789abcdef" for c in runtime_revision):
        raise ValueError("Runtime revision must be a full lowercase Git commit hash")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                 "processor_config.json", "preprocessor_config.json"):
        if not (source / name).is_file():
            raise ValueError(f"Missing source asset: {name}")
    manifest = json.loads((packed / "sov-packed.json").read_text())
    config = json.loads((source / "config.json").read_text())
    layers = config["text_config"]["num_hidden_layers"]
    if (manifest["formatVersion"] != 1 or manifest["packerVersion"] != PACKER or
            manifest["groupSize"] != 64 or manifest["layerBits"] != [8] * layers):
        raise ValueError("This distribution format requires native Q8 group-64 experts in every layer")
    identity = manifest["source"]
    index = source / "model.safetensors.index.json"
    if (sha256(source / "config.json") != identity["configSHA256"] or
            (sha256(index) if index.exists() else None) != identity.get("indexSHA256")):
        raise ValueError("Source configuration/index does not match the packed cache")
    for record in identity["files"]:
        validate_cached(source, record)
    for record in manifest["shards"]:
        validate_cached(packed, record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sov-bundle-", dir=destination.parent) as temporary:
        output = Path(temporary) / "bundle"
        output.mkdir()
        for name in ASSETS:
            if (source / name).exists():
                shutil.copyfile(source / name, output / name)
        count = copy_audio(source, identity["files"], output / "model.safetensors")
        shards = []
        for record in manifest["shards"]:
            path = named(output, record["name"])
            shutil.copyfile(named(packed, record["name"]), path)
            validate_cached(packed, record)
            portable = portable_record(path)
            if portable["headerSHA256"] != record["headerSHA256"] or portable["bytes"] != record["bytes"]:
                raise ValueError(f"Copied packed shard changed: {path.name}")
            shards.append(portable)
            print(f"Verified {path.name}", flush=True)
        for record in identity["files"]:
            validate_cached(source, record)
        bundle = {"formatVersion": 1, "packerVersion": PACKER, "sourceRepository": SOURCE,
                  "sourceRevision": REVISION, "configSHA256": identity["configSHA256"],
                  "groupSize": 64, "layerBits": manifest["layerBits"],
                  "audio": portable_record(output / "model.safetensors"), "shards": shards}
        (output / "sov-bundle.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
        if draft:
            (output / "draft").mkdir()
            for name in ("config.json", "model.safetensors"):
                shutil.copyfile(draft / name, output / "draft" / name)
        repo = Path(__file__).resolve().parents[2]
        shutil.copyfile(repo / "licenses/clef-APACHE-2.0.txt", output / "LICENSE")
        shutil.copyfile(repo / "scripts/model-conversion/sov-bundle-NOTICE.txt", output / "NOTICE")
        card = (repo / "scripts/model-conversion/sov-bundle-card.md").read_text()
        (output / "README.md").write_text(card.replace("@RUNTIME_REVISION@", runtime_revision))
        provenance = {"target": {"repository": SOURCE, "revision": REVISION},
                      "draft": {"repository": DRAFT, "revision": DRAFT_REVISION} if draft else None,
                      "runtimeRevision": runtime_revision, "audioTensorCount": count,
                      "changes": "Affine Q8 group-64 MoE experts; fused gate/up; remaining tensors retain source bytes."}
        (output / "conversion.json").write_text(json.dumps(provenance, indent=2) + "\n")
        files = sorted(path for path in output.rglob("*") if path.is_file())
        (output / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in files))
        output.rename(destination)
    print(f"Published local bundle: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-revision", required=True, help="mere-run commit supporting this format")
    args = parser.parse_args()
    export(args.source, args.packed, args.draft, args.output, args.runtime_revision)


if __name__ == "__main__":
    main()
