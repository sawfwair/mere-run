#!/usr/bin/env python3
"""Compare BF16 and routed-expert INT4 using fresh native CLI processes on macOS.

Uses the two repository fixtures plus ten deterministic, distinct speakers from
the checksum-verified Mini LibriSpeech dev-clean-2 archive (SLR31, CC BY 4.0).
No model runtime or audio transformation is performed in Python.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tarfile
import time


def digest(path, algorithm="sha256"):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text.lower().replace("’", "'"))


def errors(reference, transcript):
    expected, actual = words(reference), words(transcript)
    row = list(range(len(actual) + 1))
    for i, left in enumerate(expected, 1):
        current = [i]
        for j, right in enumerate(actual, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = current
    return row[-1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare_cases(root, archive, output):
    original = json.loads((root / "docs/qualification/nemotron-transcribe-native-20261028/report.json").read_text())
    cases = []
    for case in original["cases"]:
        path = root / case["audio"]
        if digest(path) != case["audio_sha256"]:
            raise ValueError(f"Fixture checksum changed: {path}")
        cases.append({"id": case["case"], "audio": str(path), "reference": case["reference"],
                      "audio_sha256": digest(path), "duration_seconds": case["audio_seconds"],
                      "max_tokens": case["max_tokens"], "repeated": True})
    if digest(archive, "md5") != "6d7ab67ac6a1d2c993d050e16d61080d":
        raise ValueError("Expected the original SLR31 dev-clean-2.tar.gz archive")
    audio_dir = output / "audio"
    audio_dir.mkdir()
    # Lowest ten numeric speaker IDs, excluding the two speakers in the human fixture.
    # Per speaker, first lexicographic utterance with 30–60 reference words.
    with tarfile.open(archive) as handle:
        transcripts = {}
        members = {member.name: member for member in handle.getmembers()}
        for name in sorted(members):
            if name.endswith(".trans.txt"):
                for line in handle.extractfile(members[name]).read().decode().splitlines():
                    identifier, text = line.split(" ", 1)
                    speaker = identifier.split("-")[0]
                    if speaker not in {"1272", "5694"} and 30 <= len(words(text)) <= 60:
                        transcripts.setdefault(speaker, (identifier, text, str(Path(name).parent)))
        for speaker in sorted(transcripts, key=int)[:10]:
            identifier, reference, parent = transcripts[speaker]
            raw = handle.extractfile(members[f"{parent}/{identifier}.flac"]).read()
            path = audio_dir / f"{identifier}.flac"
            path.write_bytes(raw)
            if raw[:4] != b"fLaC" or raw[4] & 127 != 0:
                raise ValueError("Expected FLAC STREAMINFO as the first metadata block")
            packed = int.from_bytes(raw[18:26], "big")
            duration = (packed & ((1 << 36) - 1)) / (packed >> 44)
            cases.append({"id": identifier, "speaker": speaker, "audio": str(path), "reference": reference,
                          "audio_sha256": digest(path), "duration_seconds": duration,
                          "max_tokens": 256, "repeated": False})
    return cases


def run(binary, model, case, label, repeat, output):
    directory = output / f"{case['id']}-{label}-{repeat}"
    directory.mkdir()
    run_dir = directory / "run"
    command = ["/usr/bin/time", "-l", str(binary), "speech", "transcribe", case["audio"],
               "--backend", "nemotron", "--model", str(model), "--max-tokens", str(case["max_tokens"]),
               "--quiet", "--run-dir", str(run_dir)]
    environment = dict(os.environ, MLX_ENABLE_TF32="0")
    start = time.monotonic()
    with (directory / "stdout.txt").open("w") as stdout, (directory / "stderr.txt").open("w") as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr, env=environment, timeout=600)
    elapsed = time.monotonic() - start
    if process.returncode:
        raise RuntimeError(f"CLI failed ({process.returncode}); see {directory / 'stderr.txt'}")
    transcript = (directory / "stdout.txt").read_text().strip()
    stderr = (directory / "stderr.txt").read_text()
    def metric(name):
        match = re.search(r"(\d+)\s+" + name, stderr)
        if not match:
            raise ValueError(f"Missing macOS time metric: {name}")
        return int(match[1])
    state = json.loads((run_dir / "transcription-run.json").read_text())["state"]
    if state != "succeeded":
        raise RuntimeError(f"Durable run did not succeed: {directory}")
    record = {"case": case["id"], "format": label, "repeat": repeat, "transcript": transcript,
              "word_errors": errors(case["reference"], transcript), "reference_words": len(words(case["reference"])),
              "wall_seconds": elapsed, "max_rss_bytes": metric("maximum resident set size"),
              "peak_memory_footprint_bytes": metric("peak memory footprint"), "durable_run_state": state}
    record["wer"] = record["word_errors"] / record["reference_words"]
    write_json(directory / "result.json", record)
    print(json.dumps(record), flush=True)
    return record


def summarize(cases, records):
    result = {}
    for label in ["bf16", "int4"]:
        selected = [record for record in records if record["format"] == label]
        first = [record for record in selected if record["repeat"] == 0]
        total_words = sum(record["reference_words"] for record in first)
        result[label] = {
            "unique_cases": len(first), "reference_words": total_words,
            "word_errors": sum(record["word_errors"] for record in first),
            "wer": sum(record["word_errors"] for record in first) / total_words,
            "first_pass_wall_seconds": sum(record["wall_seconds"] for record in first),
            "max_peak_memory_footprint_bytes": max(record["peak_memory_footprint_bytes"] for record in selected),
            "max_rss_bytes": max(record["max_rss_bytes"] for record in selected),
            "repeated_cases": {},
        }
        for case in cases:
            matching = [record for record in selected if record["case"] == case["id"]]
            if case["repeated"]:
                result[label]["repeated_cases"][case["id"]] = {
                    "runs": len(matching), "wer_values": [record["wer"] for record in matching],
                    "wall_seconds": [record["wall_seconds"] for record in matching],
                    "median_wall_seconds": statistics.median(record["wall_seconds"] for record in matching),
                    "median_peak_memory_footprint_bytes": statistics.median(record["peak_memory_footprint_bytes"] for record in matching),
                    "max_peak_memory_footprint_bytes": max(record["peak_memory_footprint_bytes"] for record in matching),
                }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path(".build/release/mere.run"))
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--int4", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True)
    binary = args.binary.resolve()
    cases = prepare_cases(root, args.archive, output)
    write_json(output / "cases.json", cases)
    models = {"bf16": args.bf16.resolve(), "int4": args.int4.resolve()}
    # Unscored warm-up per format. Fresh CLI processes still load weights for every measured run.
    for label, model in models.items():
        run(binary, model, cases[0], label, "warmup", output)
    records = []
    for repeat in range(args.repeats):
        for index, case in enumerate(cases):
            if repeat and not case["repeated"]:
                continue
            order = ["bf16", "int4"] if (repeat + index) % 2 == 0 else ["int4", "bf16"]
            for label in order:
                records.append(run(binary, models[label], case, label, repeat, output))
                write_json(output / "records.json", records)
    sources = list((root / "Sources/AudioNemotronTranscribeModel").glob("*.swift"))
    sources.append(root / "Sources/AudioSTT/NemotronTranscribe/NemotronTranscribeGenerator.swift")
    write_json(output / "report.json", {
        "schema_version": 1, "binary_sha256": digest(binary), "mlx_enable_tf32": "0",
        "source_sha256": {str(path.relative_to(root)): digest(path) for path in sorted(sources)},
        "archive_md5": digest(args.archive, "md5"), "cases": cases, "records": records,
        "summary": summarize(cases, records),
        "timing_scope": "Fresh CLI processes, including loading, frontend, prefill, decode, durable publication; one warm-up per format; caches not cleared; alternating format order.",
        "metric": "Word Levenshtein distance; lowercase, curly apostrophes normalized, punctuation ignored, no spelling equivalences. Aggregate counts each recording once (repeat 0).",
    })


if __name__ == "__main__":
    main()
