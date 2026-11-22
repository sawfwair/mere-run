#!/usr/bin/env python3
"""Package the benchmarked Inkling 2 Mini checkpoint with a shared MLX Q8 head."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx


UPSTREAM = "thinkingmachines/Inkling-2-Mini"
BASE_REVISION = "6e7c8248f75265bdd4fbab5273304243b5d8c2cb"
DRAFT_REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"
BASE_SHA256 = "65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1"
DRAFT_SHA256 = "7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2"
BASE_SHARD = "model-00001-of-00001.safetensors"
HEAD = "model.llm.unembed"
DRAFT = "mererun.dspark.markov"
CHUNK = 8 * 1024**2
MAX_SHARD_BYTES = 4 * 1024**3
QUANTIZATION = {"group_size": 64, "bits": 8, "mode": "affine"}
CHANGES = "Sawfwair replaced the BF16 output head with MLX affine Q8 and precomputed the DSpark output projections."
ASSETS = (
    "config.json", "hf_quant_config.json", "processor_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
)
MATERIALS = ("LICENSE", "generation_config.json", "tiktoken/tokenizer.model")


def verify_digest(path, expected):
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"Unexpected SHA-256 for {path}: {digest}")


def read_header(path):
    with path.open("rb") as source:
        length = int.from_bytes(source.read(8), "little")
        header = json.loads(source.read(length))
    return 8 + length, header


def tensor_size(entry):
    start, end = entry["data_offsets"]
    return end - start


def write_shard(source, destination, header_offset, entries, metadata):
    header = {"__metadata__": metadata}
    output_offset = 0
    for name, entry in entries.items():
        end = output_offset + tensor_size(entry)
        header[name] = {**entry, "data_offsets": [output_offset, end]}
        output_offset = end
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with source.open("rb") as original, destination.open("xb") as shard:
        shard.write(len(encoded).to_bytes(8, "little"))
        shard.write(encoded)
        for entry in entries.values():
            start, end = entry["data_offsets"]
            original.seek(header_offset + start)
            remaining = end - start
            while remaining:
                data = original.read(min(remaining, CHUNK))
                if not data:
                    raise EOFError(f"Truncated tensor in {source}")
                shard.write(data)
                remaining -= len(data)
    return output_offset


def shard_entries(header):
    shard, size = {}, 0
    for name, entry in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
        length = tensor_size(entry)
        if shard and size + length > MAX_SHARD_BYTES:
            yield shard
            shard, size = {}, 0
        shard[name] = entry
        size += length
    if shard:
        yield shard


def pack(weight, key):
    packed, scales, biases = mx.quantize(weight, **QUANTIZATION)
    return {f"{key}.q8_weight": packed, f"{key}.scales": scales, f"{key}.biases": biases}


def project(tensors, key, x):
    return mx.quantized_matmul(
        x, tensors[f"{key}.q8_weight"], tensors[f"{key}.scales"], tensors[f"{key}.biases"], **QUANTIZATION
    )


def export_head(base, draft, vocab, output, scratch, head_metadata):
    inputs = {}
    for label, source, names in [
        ("base", base, (f"{HEAD}.weight",)),
        ("draft", draft, ("markov_head.hidden_out.weight", "markov_head.hidden_out.bias")),
    ]:
        offset, header = read_header(source)
        extracted = scratch / f"{label}.safetensors"
        write_shard(source, extracted, offset, {name: header[name] for name in names}, {"format": "pt"})
        inputs.update(mx.load(str(extracted)))
    unembed = inputs[f"{HEAD}.weight"][:vocab]
    correction = unembed @ inputs["markov_head.hidden_out.weight"]
    bias = unembed @ inputs["markov_head.hidden_out.bias"]
    packed = pack(unembed, HEAD) | pack(correction, DRAFT) | {f"{DRAFT}.output_bias": bias}
    mx.save_safetensors(str(output), packed, metadata=head_metadata)
    restored = mx.load(str(output))
    for name, value in packed.items():
        if not mx.array_equal(value, restored[name]).item():
            raise ValueError(f"Q8 tensor did not round-trip: {name}")
    mx.random.seed(2026)
    for key, width in [(HEAD, unembed.shape[1]), (DRAFT, correction.shape[1])]:
        for rows in (1, 4, 32):
            x = mx.random.normal((1, rows, width)).astype(mx.bfloat16)
            if not mx.array_equal(project(packed, key, x), project(restored, key, x)).item():
                raise ValueError(f"Q8 projection logits changed after reload: {key}, {rows} rows")
    print("Q8 tensors and target/draft projection logits round-tripped exactly", flush=True)


def model_card():
    return f"""---
license: apache-2.0
base_model: {UPSTREAM}
base_model_relation: quantized
tags:
  - mere-run
  - mlx
  - apple-silicon
  - safetensors
  - dspark
---

# Inkling 2 Mini Q8 Head

This is Sawfwair's self-contained mere.run checkpoint of Thinking Machines'
Inkling 2 Mini. It keeps the original mixed NVFP4/MXFP8 experts, vision weights,
tokenizer, image processor, chat template, and official DSpark draft in one repository.

{CHANGES} The packed, group-size-64 vocabulary projection is shared by the
target and draft. The packed head and draft projections use mere.run's native
Inkling 2 Q8 checkpoint layout. Use a mere.run build with Q8 head support.

```sh
hf download Sawfwair/Inkling-2-Mini-Q8-Head --local-dir ./inkling2-mini-q8-head
mere.run text chat --model ./inkling2-mini-q8-head --prompt 'Explain why the sky is blue.' --stats
```

Add `--image ./photo.png` for image chat. Mini uses adaptive DSpark by default
on eligible Metal chat requests.

The original weights and model assets come from
[{UPSTREAM}@{BASE_REVISION[:8]}](https://huggingface.co/{UPSTREAM}/tree/{BASE_REVISION}).
`mtp.safetensors` is unchanged from
[{UPSTREAM}@{DRAFT_REVISION[:8]}](https://huggingface.co/{UPSTREAM}/tree/{DRAFT_REVISION}).
The original model card is included as `UPSTREAM_README.md`. Apache 2.0 and the
[Thinking Machines model acceptable use policy](https://thinkingmachines.ai/model-acceptable-use-policy/)
are linked in the upstream documentation. A copy of the license is included.
"""


def export(root, materials, output):
    base, draft = root / BASE_SHARD, root / "mtp.safetensors"
    verify_digest(base, BASE_SHA256)
    verify_digest(draft, DRAFT_SHA256)
    for path in [*(root / name for name in ASSETS), root / "README.md", *(materials / name for name in MATERIALS)]:
        if not path.is_file():
            raise FileNotFoundError(path)
    offset, header = read_header(base)
    source_metadata = header.pop("__metadata__")
    source_index = json.loads((root / "model.safetensors.index.json").read_text())
    if source_index["weight_map"] != dict.fromkeys(header, BASE_SHARD):
        raise ValueError("The upstream tensor index does not match the original shard")
    del header[f"{HEAD}.weight"]
    shards = list(shard_entries(header))
    count = len(shards) + 1
    metadata = {**source_metadata, "upstream_revision": BASE_REVISION, "changes": "Resharded original tensors byte for byte."}

    output.mkdir(parents=True)
    head = output / f"model-{count:05d}-of-{count:05d}.safetensors"
    vocab = json.loads((root / "config.json").read_text())["text_config"]["unpadded_vocab_size"]
    with TemporaryDirectory(dir=output.parent) as scratch:
        export_head(base, draft, vocab, head, Path(scratch), {
            "format": "mlx", "upstream_revision": BASE_REVISION, "draft_revision": DRAFT_REVISION, "changes": CHANGES,
        })
    _, head_header = read_header(head)
    del head_header["__metadata__"]
    files = dict.fromkeys(head_header, head.name)
    total_bytes = sum(map(tensor_size, head_header.values()))
    for index, entries in enumerate(shards, 1):
        filename = f"model-{index:05d}-of-{count:05d}.safetensors"
        total_bytes += write_shard(base, output / filename, offset, entries, metadata)
        files.update(dict.fromkeys(entries, filename))
        print(f"Wrote {filename}", flush=True)
    for name in ASSETS:
        shutil.copyfile(root / name, output / name)
    for name in MATERIALS:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(materials / name, destination)
    shutil.copyfile(root / "README.md", output / "UPSTREAM_README.md")
    shutil.copyfile(draft, output / draft.name)
    index = {"metadata": {"total_size": total_bytes}, "weight_map": files}
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    (output / "README.md").write_text(model_card())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Original Mini directory with official mtp.safetensors")
    parser.add_argument("--upstream-materials", type=Path, required=True, help="Directory with Apache LICENSE and extra upstream assets")
    parser.add_argument("--output", type=Path, required=True, help="New checkpoint directory")
    args = parser.parse_args()
    export(args.checkpoint, args.upstream_materials, args.output)


if __name__ == "__main__":
    main()
