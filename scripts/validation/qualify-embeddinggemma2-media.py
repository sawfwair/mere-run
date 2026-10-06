#!/usr/bin/env python3
"""Qualify native EmbeddingGemma 2 media inputs with deterministic local assets.

Requires numpy, Pillow, soundfile, and ffmpeg to author probes. --reference also
requires torch/transformers and runs the optional Swift preprocessing export.
The video reference consumes the same Apple-decoded frames as the native CLI;
codec/color conversion differences between decoders are not encoder errors.
Run from the repository root. No model weights or receipts are committed.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--cli-model", help="CLI model ID when testing an installed checkpoint; defaults to --model path.")
    parser.add_argument("--stress", action="store_true", help="Native-only duration/frame limits, stereo resampling, and an 8192-token mixed record (requires tokenizers).")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:96, :192]
    image = np.stack([(x * 3 + y) % 256, (x + y * 5) % 256, (y * 2 + x // 2) % 256], axis=-1).astype(np.uint8)
    Image.fromarray(image).save(args.out / "pattern.png")
    Image.fromarray(image[:, ::-1]).save(args.out / "reverse.png")
    t = np.arange(16_337, dtype=np.float64) / 16_000
    wave = (.2 * np.sin(2 * np.pi * (220 * t + 120 * t * t)) + .03 * np.sin(2 * np.pi * 930 * t)).astype(np.float32)
    sf.write(args.out / "tone.wav", wave, 16_000, subtype="FLOAT")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", "2", "-i", str(args.out / "pattern.png"),
                    "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(args.out / "clip.mp4")], check=True, timeout=60)
    shutil.copyfile(args.model / "processor_config.json", args.out / "processor_config.json")
    image_part = {"type": "image", "path": "pattern.png"}
    audio_part = {"type": "audio", "path": "tone.wav"}
    video_part = {"type": "video", "path": "clip.mp4"}
    cases = [dict(name=f"image-{dim}", content=[image_part], dimensions=dim) for dim in (128, 256, 512, 768)]
    cases += [
        dict(name="audio", content=[audio_part], dimensions=768),
        dict(name="video", content=[video_part], dimensions=768),
        dict(name="ordered-frames", content=[{"type": "video-frames", "frames": ["pattern.png", "reverse.png"]}], dimensions=768),
        dict(name="mixed", content=[{"type": "text", "text": "An abstract pattern. "}, image_part,
                                   {"type": "text", "text": " With a chirping sound. "}, audio_part], dimensions=768),
        dict(name="repeated", content=[image_part, audio_part, {"type": "image", "path": "reverse.png"}, audio_part], dimensions=768),
        dict(name="document-video-audio", content=[video_part, {"type": "text", "text": " A test clip. "}, audio_part],
             dimensions=256, task="document", title="Media probe"),
    ]
    if args.stress:
        sf.write(args.out / "long-tone.wav", np.resize(wave, 480_000), 16_000, subtype="FLOAT")
        stereo = np.repeat(wave, 3)
        sf.write(args.out / "stereo.wav", np.stack([stereo, stereo], axis=-1), 48_000, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", "2", "-i", str(args.out / "pattern.png"),
                        "-t", "34", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(args.out / "long-clip.mp4")], check=True, timeout=60)
        cases += [dict(name="audio-30-seconds", content=[{"type": "audio", "path": "long-tone.wav"}], dimensions=128, stress=True),
                  dict(name="audio-stereo-48khz", content=[{"type": "audio", "path": "stereo.wav"}], dimensions=128, stress=True),
                  dict(name="video-32-frames", content=[{"type": "video", "path": "long-clip.mp4"}], dimensions=128, stress=True)]
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
        config = json.loads((args.model / "config.json").read_text())
        marker = lambda key: tokenizer.id_to_token(config[key])
        image_block = marker("boi_token_id") + marker("image_token_id") * 253 + marker("eoi_token_id")
        audio_block = marker("boa_token_id") + marker("audio_token_id") * 750 + marker("eoa_token_index")
        video_block = marker("boi_token_id") + marker("video_token_id") * 128 + marker("eoi_token_id")
        prefix = image_block + audio_block + video_block * 32
        tail = " a" * (8190 - len(tokenizer.encode(prefix, add_special_tokens=False).ids))
        assert len(tokenizer.encode(prefix + tail, add_special_tokens=False).ids) + 2 == 8192
        cases.append(dict(name="mixed-8192", content=[image_part, {"type": "audio", "path": "long-tone.wav"},
                          {"type": "video-frames", "frames": ["pattern.png"] * 32}, {"type": "text", "text": tail}],
                          dimensions=128, stress=True))
    if args.reference:
        export_preprocessing(args)
    # SwiftPM may relink the CLI while building the optional export test. Capture
    # its identity after that build, and perform no further builds during probes.
    receipt = {"cli_model": args.cli_model or str(args.model.resolve()), "cli_sha256": digest(args.cli), "sha256": {name: digest(args.model / name) for name in
               ("config.json", "processor_config.json", "tokenizer.json", "model.safetensors")}, "cases": [], "checks": {}}
    receipt["assets_sha256"] = {path.name: digest(path) for path in args.out.iterdir() if path.suffix in (".png", ".wav", ".mp4")}
    receipt["hardware"] = {"cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                           "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
                           "macos": subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()}
    receipt["swap_before"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
    for case in cases:
        document = args.out / (case["name"] + ".inputs.json")
        document.write_text(json.dumps({"inputs": [{"content": case["content"]}]}))
        command = [str(args.cli.resolve()), "text", "embed", "--model", args.cli_model or str(args.model.resolve()), "--input-json", str(document),
                   "--dimensions", str(case["dimensions"]), "--task", case.get("task", "raw")]
        if "title" in case:
            command += ["--title", case["title"]]
        start = time.monotonic()
        result = subprocess.run(["/usr/bin/time", "-l", *command], capture_output=True, text=True, timeout=180)
        (args.out / (case["name"] + ".stderr.log")).write_text(result.stderr)
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        vector = payload["data"][0]["embedding"]
        assert len(payload["data"]) == 1 and payload["data"][0]["index"] == 0
        assert len(vector) == case["dimensions"] and all(math.isfinite(v) for v in vector)
        assert abs(np.linalg.norm(vector) - 1) < 1e-5
        footprint = re.search(r"(\d+)\s+peak memory footprint", result.stderr)
        case.update(payload=payload, elapsed_seconds=time.monotonic() - start,
                    peak_footprint_bytes=int(footprint[1]) if footprint else None)
        receipt["cases"].append(case)
        (args.out / "native.json").write_text(json.dumps(receipt, indent=2))
        print(case["name"], payload["usage"]["total_tokens"], f"{case['elapsed_seconds']:.2f}s", flush=True)
    selected = [case for case in cases if case["name"] in ("image-768", "audio", "video", "mixed")]
    batch_file = args.out / "batch.inputs.json"
    batch_file.write_text(json.dumps({"inputs": [{"content": case["content"]} for case in selected]}))
    batch = subprocess.run([str(args.cli.resolve()), "text", "embed", "--model", args.cli_model or str(args.model.resolve()),
                            "--input-json", str(batch_file)], capture_output=True, text=True, timeout=180)
    assert batch.returncode == 0, batch.stderr
    batch_payload = json.loads(batch.stdout)
    assert [entry["index"] for entry in batch_payload["data"]] == list(range(len(selected)))
    assert batch_payload["usage"]["total_tokens"] == sum(case["payload"]["usage"]["total_tokens"] for case in selected)
    for case, entry in zip(selected, batch_payload["data"]):
        assert np.max(np.abs(np.array(case["payload"]["data"][0]["embedding"]) - entry["embedding"])) < 1e-5
    (args.out / "batch.json").write_text(json.dumps(batch_payload, indent=2))
    receipt["checks"]["mixed_batch_order_and_solo_consistency"] = True
    rejected = subprocess.run([str(args.cli.resolve()), "text", "embed", "--model", args.cli_model or str(args.model.resolve()),
                               "--image", str(args.out / "pattern.png"), "--max-tokens", "2"], capture_output=True, text=True, timeout=60)
    assert rejected.returncode != 0 and "cannot be truncated" in rejected.stderr and not rejected.stdout
    receipt["checks"]["media_budget_rejection"] = True
    invalid_file = args.out / "reserved.inputs.json"
    invalid_file.write_text(json.dumps({"inputs": [{"content": [{"type": "text", "text": "<|image|>"}, image_part]}]}))
    invalid = subprocess.run([str(args.cli.resolve()), "text", "embed", "--model", args.cli_model or str(args.model.resolve()),
                              "--input-json", str(invalid_file)], capture_output=True, text=True, timeout=60)
    assert invalid.returncode != 0 and "reserved media markers" in invalid.stderr and not invalid.stdout
    receipt["checks"]["reserved_marker_rejection"] = True
    if args.stress:
        by_name = {case["name"]: case for case in cases}
        assert by_name["audio-30-seconds"]["payload"]["usage"]["total_tokens"] == 754
        assert by_name["video-32-frames"]["payload"]["usage"]["total_tokens"] == 4162
        assert by_name["audio-stereo-48khz"]["payload"]["usage"]["total_tokens"] == 30
        assert by_name["mixed-8192"]["payload"]["usage"]["total_tokens"] == 8192
        sf.write(args.out / "too-long.wav", np.zeros(480_001, dtype=np.float32), 16_000, subtype="FLOAT")
        invalid = subprocess.run([str(args.cli.resolve()), "text", "embed", "--model", args.cli_model or str(args.model.resolve()),
                                  "--audio", str(args.out / "too-long.wav")], capture_output=True, text=True, timeout=60)
        assert invalid.returncode != 0 and "at most 30 seconds" in invalid.stderr and not invalid.stdout
        receipt["checks"]["audio_duration_rejection"] = True
    if args.reference:
        reference(args, cases, receipt, wave)
    receipt["swap_after"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
    assert all(digest(args.model / name) == value for name, value in receipt["sha256"].items()), "Checkpoint changed during qualification"
    assert digest(args.cli) == receipt["cli_sha256"], "CLI changed during qualification; rerun against the final build"
    receipt["status"] = "passed"
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print("All media qualification assertions passed.", flush=True)


def export_preprocessing(args):
    environment = dict(os.environ, MERERUN_EMBEDDINGGEMMA2_MEDIA_RECEIPT=str(args.out))
    result = subprocess.run(["swift", "test", "--filter", "EmbeddingGemma2MediaReceiptTests"], env=environment,
                            capture_output=True, text=True, timeout=180)
    (args.out / "preprocessing-export.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr


def reference(args, cases, receipt, wave):
    import torch
    import transformers
    from safetensors.numpy import load_file
    from transformers import AutoProcessor, EmbeddingGemma2Model

    exported = load_file(str(args.out / "native-preprocessing.safetensors"))
    assert np.array_equal(exported["audio_samples"], wave), "Audio decoder dropped or changed samples"
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    audio = processor(audio=wave, return_tensors="np")
    error = float(np.max(np.abs(audio["input_features"] - exported["audio_features"])))
    assert np.array_equal(audio["input_features_mask"][0], exported["audio_mask"])
    assert error < 2e-3, f"Audio frontend mismatch: {error}"
    receipt["checks"]["audio_frontend_max_error"] = error
    torch.set_num_threads(6)
    model = EmbeddingGemma2Model.from_pretrained(str(args.model), local_files_only=True, dtype=torch.float32,
                                               attn_implementation="sdpa").eval()
    references = {"torch": torch.__version__, "transformers": transformers.__version__, "dtype": "float32",
                  "video_decoder": "MediaIO native frames; identical decoded pixels", "cases": []}
    for case in cases:
        if case.get("stress"):
            continue
        text, images, audios, videos = "", [], [], []
        for part in case["content"]:
            kind = part["type"]
            if kind == "text":
                text += part["text"]
            elif kind == "image":
                text += "<|image|>"
                images.append(Image.open(args.out / part["path"]).convert("RGB"))
            elif kind == "audio":
                text += "<|audio|>"
                audios.append(wave)
            else:
                text += "<|video|>"
                paths = part.get("frames") or [str(p.relative_to(args.out)) for p in sorted((args.out / "native-frames").glob("*.png"))]
                videos.append(np.stack([np.array(Image.open(args.out / path).convert("RGB")) for path in paths]))
        if case.get("task") == "document":
            text = f"title: {case['title']} | text: " + text
        inputs = processor(text=text, images=images or None, audio=audios or None, videos=videos or None, return_tensors="pt")
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1)
            vector = torch.nn.functional.normalize(((hidden * mask).sum(1) / mask.sum(1))[:, :case["dimensions"]], dim=-1)[0].numpy()
        native = case["payload"]["data"][0]["embedding"]
        cosine = float(np.dot(native, vector) / (np.linalg.norm(native) * np.linalg.norm(vector)))
        count = int(mask.sum())
        assert count == case["payload"]["usage"]["total_tokens"], f"Token mismatch: {case['name']}"
        entry = dict(name=case["name"], cosine=cosine, max_absolute_error=float(np.max(np.abs(np.array(native) - vector))),
                     tokens=count, vector=vector.tolist())
        references["cases"].append(entry)
        (args.out / "reference.json").write_text(json.dumps(references, indent=2))
        print("reference", case["name"], f"{cosine:.8f}", flush=True)
        assert cosine > .999, f"Encoder parity failed: {entry}"
    receipt["reference"] = references


if __name__ == "__main__":
    main()
