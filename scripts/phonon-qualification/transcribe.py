#!/usr/bin/env python3
"""Transcribe qualification PCM locally through mere.run Parakeet (requires NumPy)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import wave
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("cli", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("directories", type=Path, nargs="+")
args = parser.parse_args()
rows = []
for directory in args.directories:
    for pcm in sorted(directory.glob("*.f32")):
        wav = pcm.with_suffix(".wav")
        samples = np.fromfile(pcm, np.float32)
        encoded = (np.clip(samples, -1, 1) * 32767).astype("<i2")
        with wave.open(str(wav), "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(24000)
            file.writeframes(encoded.tobytes())
        command = [str(args.cli.resolve()), "speech", "transcribe", str(wav.resolve()),
                   "--backend", "parakeet", "--language", "en"]
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        pcm.with_suffix(".asr.txt").write_text(process.stdout)
        pcm.with_suffix(".asr.log").write_text(process.stderr)
        # The first paragraph is the full transcript; later paragraphs carry timestamps.
        transcript = process.stdout.split("\n\n")[0].strip()
        rows.append(dict(runtime=directory.name, id=pcm.stem, transcript=transcript))
        print(directory.name, pcm.stem, transcript, flush=True)
args.output.write_text(json.dumps(rows, indent=2) + "\n")
args.output.with_suffix(".receipt.json").write_text(json.dumps(dict(
    cli_sha256=hashlib.sha256(args.cli.read_bytes()).hexdigest(), backend="parakeet", language="en",
    input_format="mono PCM16 24000Hz; clip to [-1,1], multiply by 32767, truncate",
    command="mere.run speech transcribe WAV --backend parakeet --language en"), indent=2) + "\n")
