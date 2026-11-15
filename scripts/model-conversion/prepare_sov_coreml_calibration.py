#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.7.0", "numpy==2.3.5", "safetensors==0.8.0", "pyarrow==26.0.1", "soundfile==0.14.0"]
# ///
"""Prepare independent audio and FP32 intermediate activations for ANE calibration.

Input is the Parquet file from hf-internal-testing/librispeech_asr_dummy at
5be91486e11a2d616f4ec5db8d3fd248585ac07a. No dataset Python is executed.
Training and held-out utterances are disjoint and exclude the qualification WAVs.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time


def audio_features(samples):
    import numpy as np
    import torch
    import torch.nn.functional as F
    windows = (len(samples) + 479999) // 480000
    signal = F.pad(torch.as_tensor(samples), (0, windows * 480000 - len(samples)))
    spectrum = torch.stft(signal, 400, hop_length=160, window=torch.hann_window(400), return_complex=True)
    power = spectrum.abs().square()[:, :-1]
    mel_max = 15 + np.log(8) / (np.log(6.4) / 27)
    points = np.linspace(0, mel_max, 130)
    hz = np.where(points < 15, points * 200 / 3, 1000 * np.exp((points - 15) * np.log(6.4) / 27))
    frequencies = np.arange(201) * 40
    bank = np.maximum(0, np.minimum((frequencies[None] - hz[:-2, None]) / (hz[1:-1, None] - hz[:-2, None]),
                                  (hz[2:, None] - frequencies[None]) / (hz[2:, None] - hz[1:-1, None])))
    bank *= 2 / (hz[2:] - hz[:-2])[:, None]
    features = (torch.as_tensor(bank, dtype=torch.float32) @ power).clamp_min(1e-10).log10()
    features = (torch.maximum(features, features.max() - 8) + 4) / 4
    return features.reshape(128, windows, 3000).permute(1, 0, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-count", type=int, default=12)
    parser.add_argument("--heldout-count", type=int, default=8)
    args = parser.parse_args()
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from safetensors.numpy import save_file
    import torch
    from convert_sov_coreml import build_graph
    torch.set_num_threads(8)
    rows = pq.read_table(args.parquet).to_pylist()
    # Interleave the split to cover the file's ordering without sharing examples.
    train = list(range(0, args.train_count * 3, 3))
    heldout = list(range(1, args.heldout_count * 3 + 1, 3))
    assert not set(train) & set(heldout)
    graph = build_graph(args.checkpoint)
    graph.linear_operator = True
    graph.attention_layout = "sdpa"
    args.output.mkdir(parents=True, exist_ok=True)
    samples = {f"layer{layer}/{split}": [] for layer in [0, 16, 31] for split in ["train", "heldout"]}
    features = {"train": [], "heldout": []}
    manifest = dict(dataset="hf-internal-testing/librispeech_asr_dummy", revision="5be91486e11a2d616f4ec5db8d3fd248585ac07a",
        parquetSHA256=hashlib.sha256(args.parquet.read_bytes()).hexdigest(), examples=[])
    for split, indices in [("train", train), ("heldout", heldout)]:
        graph.collect_activation_max = split == "train"
        for index in indices:
            row = rows[index]
            audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            assert rate == 16000 and audio.ndim == 1 and len(audio) <= 480000
            started = time.perf_counter()
            with torch.inference_mode():
                mel = audio_features(audio)
                features[split].append(mel.numpy().astype(np.float16))
                x = graph.stem(mel[:, :, None, :])
                for layer in range(32):
                    if layer in [0, 16, 31]:
                        samples[f"layer{layer}/{split}"].append(x.numpy().astype(np.float16))
                    x = graph.block(x, layer)
                graph.project(x)
            manifest["examples"].append(dict(id=row["id"], speaker=row["speaker_id"], split=split,
                audioSHA256=hashlib.sha256(row["audio"]["bytes"]).hexdigest(), seconds=len(audio) / 16000, preparationSeconds=time.perf_counter()-started))
            print(json.dumps(manifest["examples"][-1]), flush=True)
    save_file({k: np.ascontiguousarray(np.concatenate(v, axis=0)) for k, v in samples.items()}, str(args.output / "activations.safetensors"))
    save_file({k: np.ascontiguousarray(np.concatenate(v, axis=0)) for k, v in features.items()}, str(args.output / "mel.safetensors"))
    save_file({k: np.ascontiguousarray(v.numpy()) for k, v in graph.activation_max.items()}, str(args.output / "activation-max.safetensors"))
    save_file({**{"min/" + k: np.ascontiguousarray(v.numpy()) for k, v in graph.activation_min.items()},
               **{"max/" + k: np.ascontiguousarray(v.numpy()) for k, v in graph.activation_upper.items()}}, str(args.output / "activation-ranges.safetensors"))
    (args.output / "calibration.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
