#!/usr/bin/env python3
"""Exercise the released EmbeddingGemma 2 checkpoint through the public CLI.

Uses stdlib only unless --reference is selected (torch and transformers required).
Writes inputs, vectors, timings, and assertions to a receipt directory. The optional
reference uses Transformers' text encoder with the original checkpoint tensors.
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(
        sum(x * x for x in a) * sum(y * y for y in b)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--long-context", action="store_true")
    parser.add_argument("--reuse-native", action="store_true", help="Reuse matching native receipts when running reference checks separately.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    queries = ["What causes the northern lights?", "How do I bake sourdough bread?"]
    documents = [
        "Charged particles from the sun collide with gases in Earth's atmosphere, producing the northern lights.",
        "Sourdough bread uses a fermented flour-and-water starter; knead, proof, and bake the dough.",
        "The Toronto subway uses rail vehicles to carry passengers around the city.",
    ]
    cases = []
    for dimensions in (128, 256, 512, 768):
        cases.extend([
            dict(name=f"queries-{dimensions}", texts=queries, task="query", dimensions=dimensions),
            dict(name=f"documents-{dimensions}", texts=documents, task="document", dimensions=dimensions),
        ])
    cases.extend([
        dict(name="solo", texts=queries[:1], task="query", dimensions=768),
        dict(name="unicode-and-empty", texts=["", "雪が降る静かな夜。", "Bonjour, où est la bibliothèque?", "🦉 Swift & MLX"], task="raw", dimensions=768),
        dict(name="code-query", texts=["find a function that sorts a list of numbers"], task="code-retrieval", dimensions=768),
        dict(name="code-documents", texts=["def sort_numbers(values):\n    return sorted(values)", "def add(a, b):\n    return a + b"], task="document", dimensions=768, title="utilities.py"),
        dict(name="truncation", texts=["The northern lights glow over the snow. " * 100], task="similarity", dimensions=768, max_tokens=32),
        dict(name="sliding-window", texts=["Auroras glow over snow. " * 120], task="raw", dimensions=768),
    ])
    if args.long_context:
        cases.append(dict(name="context-8192", texts=["Auroras glow over snow. " * 2000], task="raw", dimensions=128, max_tokens=8192))
    receipt = {
        "model_path": str(args.model.resolve()), "cases": [], "checks": {},
        "hardware": {
            "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
            "macos": subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip(),
        },
        "cli_sha256": hashlib.sha256(args.cli.read_bytes()).hexdigest(),
    }
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        digest = hashlib.sha256()
        with (args.model / name).open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        receipt.setdefault("sha256", {})[name] = digest.hexdigest()
    cached = {}
    if args.reuse_native:
        previous = json.loads((args.out / "native.json").read_text())
        assert previous["sha256"] == receipt["sha256"], "Checkpoint changed since native run"
        if "cli_sha256" in previous:
            assert previous["cli_sha256"] == receipt["cli_sha256"], "CLI changed since native run"
        cached = {case["name"]: case for case in previous["cases"]}
    for case in cases:
        if case["name"] in cached:
            previous = cached[case["name"]]
            assert all(previous[key] == value for key, value in case.items()), "Qualification inputs changed"
            case.update(previous)
            footprint = re.search(r"(\d+)\s+peak memory footprint", (args.out / (case["name"] + ".stderr.log")).read_text())
            case["peak_footprint_bytes"] = int(footprint[1]) if footprint else None
            receipt["cases"].append(case)
            continue
        command = [str(args.cli.resolve()), "text", "embed", *case["texts"], "--model", str(args.model.resolve()), "--task", case["task"], "--dimensions", str(case["dimensions"])]
        for key, flag in (("title", "--title"), ("max_tokens", "--max-tokens")):
            if key in case:
                command.extend([flag, str(case[key])])
        start = time.monotonic()
        result = subprocess.run(["/usr/bin/time", "-l", *command], capture_output=True, text=True)
        (args.out / (case["name"] + ".stderr.log")).write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"{case['name']} failed: {result.stderr}")
        payload = json.loads(result.stdout)
        vectors = [item["embedding"] for item in payload["data"]]
        assert payload["model"] == "text-embed-embeddinggemma2"
        assert [item["index"] for item in payload["data"]] == list(range(len(case["texts"])))
        assert len(vectors) == len(case["texts"])
        for vector in vectors:
            assert len(vector) == case["dimensions"] and all(math.isfinite(x) for x in vector)
            assert abs(math.sqrt(sum(x * x for x in vector)) - 1) < 1e-5
        rss = re.search(r"(\d+)\s+maximum resident set size", result.stderr)
        footprint = re.search(r"(\d+)\s+peak memory footprint", result.stderr)
        case.update(payload=payload, elapsed_seconds=time.monotonic() - start, max_rss_bytes=int(rss[1]) if rss else None,
                    peak_footprint_bytes=int(footprint[1]) if footprint else None)
        receipt["cases"].append(case)
        (args.out / "native.json").write_text(json.dumps(receipt, indent=2))
        print(f"{case['name']}: {case['elapsed_seconds']:.2f}s, peak footprint {case['peak_footprint_bytes'] / 2**30:.2f} GiB, tokens {payload['usage']['total_tokens']}", flush=True)
    by_name = {case["name"]: case for case in cases}
    for dimensions in (128, 256, 512, 768):
        query = by_name[f"queries-{dimensions}"]["payload"]["data"]
        docs = by_name[f"documents-{dimensions}"]["payload"]["data"]
        scores = [[cosine(q["embedding"], d["embedding"]) for d in docs] for q in query]
        assert [max(range(len(row)), key=row.__getitem__) for row in scores] == [0, 1]
        receipt["checks"][f"retrieval-{dimensions}"] = scores
    batch = by_name["queries-768"]["payload"]["data"][0]["embedding"]
    solo = by_name["solo"]["payload"]["data"][0]["embedding"]
    receipt["checks"]["batch_solo_cosine"] = cosine(batch, solo)
    assert cosine(batch, solo) > .999
    code = by_name["code-query"]["payload"]["data"][0]["embedding"]
    code_scores = [cosine(code, d["embedding"]) for d in by_name["code-documents"]["payload"]["data"]]
    assert code_scores[0] > code_scores[1]
    receipt["checks"]["code_retrieval"] = code_scores
    assert by_name["truncation"]["payload"]["usage"]["total_tokens"] == 32
    if args.long_context:
        assert by_name["context-8192"]["payload"]["usage"]["total_tokens"] == 8192
    if args.reference:
        run_reference(args, cases, receipt)
    receipt["status"] = "passed"
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print("All checkpoint qualification assertions passed.", flush=True)


def run_reference(args, cases, receipt):
    import torch
    import transformers
    from safetensors import safe_open
    from transformers import AutoTokenizer, EmbeddingGemma2TextConfig, EmbeddingGemma2TextModel

    torch.set_num_threads(6)
    config = EmbeddingGemma2TextConfig(**json.loads((args.model / "config.json").read_text())["text_config"])
    config._attn_implementation = "sdpa"
    model = EmbeddingGemma2TextModel(config)
    with safe_open(str(args.model / "model.safetensors"), framework="pt", device="cpu") as checkpoint:
        tensors = {key.removeprefix("language_model."): checkpoint.get_tensor(key).float() for key in checkpoint.keys() if key.startswith("language_model.")}
    model.load_state_dict(tensors, strict=True, assign=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    prefixes = {"raw": "", "query": "task: search result | query: ", "code-retrieval": "task: code retrieval | query: ", "similarity": "task: sentence similarity | query: "}
    receipt["reference"] = {"torch": torch.__version__, "transformers": transformers.__version__, "dtype": "float32", "device": "cpu", "cases": []}
    for case in cases:
        if case["name"] == "context-8192":
            continue  # Native context stress is separate from short/medium CPU parity.
        prefix = f"title: {case.get('title', 'none')} | text: " if case["task"] == "document" else prefixes[case["task"]]
        tokens = tokenizer([prefix + text for text in case["texts"]], padding=True, truncation=True, max_length=case.get("max_tokens", 8192), return_tensors="pt")
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state.float()
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            vectors = torch.nn.functional.normalize(pooled[:, :case["dimensions"]], dim=-1).tolist()
        actual = [item["embedding"] for item in case["payload"]["data"]]
        similarities = [cosine(a, b) for a, b in zip(actual, vectors)]
        errors = [max(abs(a - b) for a, b in zip(native, reference)) for native, reference in zip(actual, vectors)]
        counts = tokens["attention_mask"].sum(1).tolist()
        assert sum(counts) == case["payload"]["usage"]["total_tokens"], f"Tokenizer count mismatch: {case['name']}"
        entry = {"name": case["name"], "token_counts": counts, "cosines": similarities, "max_absolute_errors": errors, "vectors": vectors}
        receipt["reference"]["cases"].append(entry)
        (args.out / "reference.json").write_text(json.dumps(receipt["reference"], indent=2))
        assert min(similarities) > .999, f"Embedding parity failed: {case['name']}: {similarities}"
        print(f"reference {case['name']}: min cosine {min(similarities):.8f}, max error {max(errors):.6f}", flush=True)


if __name__ == "__main__":
    main()
