#!/usr/bin/env python3
"""Run the unmodified upstream benchmark serially. Do not run other inference/builds beside it."""
import argparse
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("bench", type=Path)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
corpus = json.loads((Path(__file__).with_name("corpus.json")).read_text())
rows = []
for quant, filename in [("f32", "model.safetensors"), ("q8", "model.q8.gguf")]:
    for prompt in corpus:
        if prompt["id"] not in ["short-hello", "directions", "long-sentence"]:
            continue
        command = [str(args.bench), "--model", str(args.checkpoint / filename),
                   "--config", str(args.checkpoint / "config.json"), "--voice", str(args.checkpoint / "default-voice.safetensors"),
                   "--cpu", "--threads", "4", "--lang", "none", "--input", prompt["text"],
                   "--temperature", "0", "--seed", "42", "--warmup", "3", "--iters", "10", "--per-iter"]
        if quant == "q8":
            command.extend(["--quant", "q8"])
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
        (args.output / f'{quant}-{prompt["id"]}.log').write_text(process.stdout)
        for match in re.finditer(r"iter\s+(\d+): total\s+([\d.]+)ms\s+ttfa\s+([\d.]+)ms\s+frames\s+(\d+)", process.stdout):
            iteration, total, ttfa, frames = match.groups()
            rows.append(dict(format=quant, id=prompt["id"], iteration=int(iteration), totalSeconds=float(total) / 1000,
                             firstAudioSeconds=float(ttfa) / 1000, frames=int(frames)))
        print(quant, prompt["id"], "complete", flush=True)
(args.output / "timings.json").write_text(json.dumps(rows, indent=2) + "\n")
