#!/usr/bin/env python3
"""Generate independent full-checkpoint Clef Flash parity requests and outputs.

Reference tooling only. Requires the versions listed in the dated receipt,
including mlx-vlm 0.7.4. Native inference does not use Python.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw
from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLVideoProcessor

REVISION = "6822f0f244ee9e19df76908ba3302f7fe40ceea6"
REFERENCE_SHA256 = "5b381f596a2507885a5a736bec7caef3200fb882f91ef2fea08a6bfa8f1877d2"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.reference.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Use clef_mlx.py from the pinned Flash revision")
    receipt = json.loads((Path(__file__).resolve().parents[2]
                         / "docs/benchmarks/receipts/clef-flash-native-2026-10-02.json").read_text())
    for name, expected in receipt["checkpoint_files"].items():
        digest = hashlib.sha256()
        with (args.checkpoint / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"Checkpoint file does not match pinned receipt: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), (235, 35, 30)).save(args.output / "red.png")
    Image.new("RGB", (64, 64), (235, 35, 30)).save(args.output / "frame.png")
    pattern = Image.new("RGB", (301, 197), (255, 255, 255))
    draw = ImageDraw.Draw(pattern)
    for x in range(0, 301, 7):
        draw.rectangle((x, 0, x + 3, 196), fill=(20, 60, 220))
    draw.ellipse((75, 30, 220, 175), fill=(235, 35, 30))
    pattern.save(args.output / "resize.png")
    spec = importlib.util.spec_from_file_location("clef_flash_reference", args.reference)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    model = reference.load(args.checkpoint, backend="vlm")
    # mlx-vlm 0.7.4 ignores nested video settings. Restore the checkpoint's
    # declared processor settings without changing its neural model or loader.
    config = json.loads((args.checkpoint / "processor_config.json").read_text())["video_processor"]
    kwargs = {key: config[key] for key in [
        "patch_size", "temporal_patch_size", "merge_size", "fps", "min_frames", "max_frames",
        "image_mean", "image_std", "rescale_factor", "do_rescale", "do_normalize", "do_convert_rgb",
    ]}
    kwargs.update(min_pixels=config["size"]["shortest_edge"], max_pixels=config["size"]["longest_edge"])
    model.processor.video_processor = Qwen3VLVideoProcessor(**kwargs)
    report = {"revision": REVISION, "reference_sha256": REFERENCE_SHA256, "cases": {}}
    for name in ["text", "image", "video", "resize"]:
        request = dict(receipt["cases"][name]["request_without_media"])
        if name in ["image", "resize"]:
            request["images"] = [str((args.output / ("red.png" if name == "image" else "resize.png")).resolve())]
        elif name == "video":
            request["videos"] = [[str((args.output / "frame.png").resolve())] * 4]
        (args.output / f"{name}.json").write_text(json.dumps(request, indent=2) + "\n")
        if request.get("videos"):
            request["videos"] = [np.stack([np.asarray(Image.open(path).convert("RGB")) for path in frames])
                                 for frames in request["videos"]]
        encoded, logits = model.logits(request, max_length=16384)
        probabilities = {question.question_id: dict(zip(question.option_ids, mx.softmax(logit.astype(mx.float32)).tolist()))
                         for question, logit in zip(encoded.questions, logits)}
        report["cases"][name] = {
            "input_tokens": len(encoded.input_ids), "ids": list(encoded.input_ids),
            "fields": [{"id": question.question_id, "question_span": list(question.question_span),
                        "option_spans": [list(span) for span in question.option_spans],
                        "option_ids": list(question.option_ids)} for question in encoded.questions],
            "answers": {question.question_id: reference.systemone_answer(
                request["questions"][question.question_id], probabilities[question.question_id])
                for question in encoded.questions},
        }
    (args.output / "reference.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
