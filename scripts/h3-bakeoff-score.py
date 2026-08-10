#!/usr/bin/env python3
"""Score a same-seed MiniMax-H3 candidate against its dense-quality MP4."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from array import array
from fractions import Fraction
from pathlib import Path
from typing import Any


class ScoreError(RuntimeError):
    """A deterministic media probe or score could not be produced."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one MiniMax-H3 bake-off MP4 with the matched dense-quality output. "
            "PSNR, SSIM, VMAF, and waveform metrics are diagnostics, not perceptual acceptance."
        )
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", required=True, type=Path, dest="json_path")
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--expected-width", type=int)
    parser.add_argument("--expected-height", type=int)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--expected-fps", type=float)
    parser.add_argument("--expected-sample-rate", type=int)
    parser.add_argument("--expected-channels", type=int)
    return parser.parse_args()


def resolve_tool(environment_key: str, fallback: str) -> str:
    configured = os.environ.get(environment_key)
    if configured:
        path = Path(configured)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise ScoreError(f"{environment_key} is not executable: {configured}")
    resolved = shutil.which(fallback)
    if not resolved:
        raise ScoreError(f"required tool is unavailable: {fallback}")
    return resolved


def run(command: list[str], *, capture_binary: bool = False) -> subprocess.CompletedProcess[Any]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=not capture_binary,
    )
    if result.returncode != 0:
        stderr = result.stderr if isinstance(result.stderr, str) else result.stderr.decode(errors="replace")
        raise ScoreError(f"command failed ({result.returncode}): {' '.join(command)}\n{stderr.strip()}")
    return result


def probe_media(path: Path, ffprobe: str) -> dict[str, Any]:
    if not path.is_file():
        raise ScoreError(f"media file does not exist: {path}")
    result = run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ScoreError(f"ffprobe returned invalid JSON for {path}: {error}") from error


def first_stream(probe: dict[str, Any], codec_type: str) -> dict[str, Any] | None:
    return next(
        (stream for stream in probe.get("streams", []) if stream.get("codec_type") == codec_type),
        None,
    )


def numeric(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def frame_rate(stream: dict[str, Any]) -> float | None:
    raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    try:
        value = float(Fraction(raw))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def frame_count(stream: dict[str, Any]) -> int | None:
    raw = stream.get("nb_read_frames") or stream.get("nb_frames")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def media_summary(probe: dict[str, Any]) -> dict[str, Any]:
    video = first_stream(probe, "video")
    audio = first_stream(probe, "audio")
    return {
        "format_duration_seconds": numeric(probe.get("format", {}).get("duration")),
        "video": None
        if video is None
        else {
            "codec": video.get("codec_name"),
            "width": video.get("width"),
            "height": video.get("height"),
            "frame_rate": frame_rate(video),
            "frame_count": frame_count(video),
            "duration_seconds": numeric(video.get("duration")),
        },
        "audio": None
        if audio is None
        else {
            "codec": audio.get("codec_name"),
            "sample_rate": int(audio["sample_rate"]) if audio.get("sample_rate") else None,
            "channels": audio.get("channels"),
            "duration_seconds": numeric(audio.get("duration")),
        },
    }


def structural_issues(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    expected: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    baseline_video = baseline.get("video")
    candidate_video = candidate.get("video")
    baseline_audio = baseline.get("audio")
    candidate_audio = candidate.get("audio")
    if baseline_video is None or candidate_video is None:
        issues.append("both artifacts must contain a video stream")
    else:
        for field in ("width", "height", "frame_count", "frame_rate"):
            if baseline_video.get(field) != candidate_video.get(field):
                issues.append(
                    f"video {field} differs: baseline={baseline_video.get(field)} "
                    f"candidate={candidate_video.get(field)}"
                )
        baseline_duration = baseline_video.get("duration_seconds")
        candidate_duration = candidate_video.get("duration_seconds")
        fps = baseline_video.get("frame_rate")
        if baseline_duration is not None and candidate_duration is not None and fps:
            if abs(baseline_duration - candidate_duration) > (1 / fps) + 1e-3:
                issues.append(
                    "video duration differs by more than one frame: "
                    f"baseline={baseline_duration} candidate={candidate_duration}"
                )
    if baseline_audio is None or candidate_audio is None:
        issues.append("both artifacts must contain an audio stream")
    else:
        for field in ("sample_rate", "channels"):
            if baseline_audio.get(field) != candidate_audio.get(field):
                issues.append(
                    f"audio {field} differs: baseline={baseline_audio.get(field)} "
                    f"candidate={candidate_audio.get(field)}"
                )
        baseline_duration = baseline_audio.get("duration_seconds")
        candidate_duration = candidate_audio.get("duration_seconds")
        if baseline_duration is not None and candidate_duration is not None:
            if abs(baseline_duration - candidate_duration) > 0.05:
                issues.append(
                    "audio duration differs by more than 50 ms: "
                    f"baseline={baseline_duration} candidate={candidate_duration}"
                )
    if candidate_video is not None:
        for field in ("width", "height", "frame_count", "frame_rate"):
            expected_value = expected.get(field)
            if expected_value is not None and candidate_video.get(field) != expected_value:
                issues.append(
                    f"video {field} violates request: expected={expected_value} "
                    f"actual={candidate_video.get(field)}"
                )
        expected_frames = expected.get("frame_count")
        expected_fps = expected.get("frame_rate")
        candidate_duration = candidate_video.get("duration_seconds")
        if expected_frames is not None and expected_fps and candidate_duration is not None:
            expected_duration = expected_frames / expected_fps
            if abs(candidate_duration - expected_duration) > (1 / expected_fps) + 1e-3:
                issues.append(
                    "video duration violates request: "
                    f"expected={expected_duration} actual={candidate_duration}"
                )
    if candidate_audio is not None:
        expected_sample_rate = expected.get("sample_rate")
        expected_channels = expected.get("channels")
        if expected_sample_rate is not None and candidate_audio.get("sample_rate") != expected_sample_rate:
            issues.append(
                "audio sample_rate violates request: "
                f"expected={expected_sample_rate} actual={candidate_audio.get('sample_rate')}"
            )
        if expected_channels is not None and candidate_audio.get("channels") != expected_channels:
            issues.append(
                "audio channels violates request: "
                f"expected={expected_channels} actual={candidate_audio.get('channels')}"
            )
        expected_frames = expected.get("frame_count")
        expected_fps = expected.get("frame_rate")
        candidate_duration = candidate_audio.get("duration_seconds")
        if expected_frames is not None and expected_fps and candidate_duration is not None:
            expected_duration = expected_frames / expected_fps
            if abs(candidate_duration - expected_duration) > 0.05:
                issues.append(
                    "audio duration violates request: "
                    f"expected={expected_duration} actual={candidate_duration}"
                )
    return issues


def video_filter_metric(
    ffmpeg: str,
    baseline: Path,
    candidate: Path,
    filter_name: str,
    pattern: str,
) -> float:
    result = run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-i",
            str(candidate),
            "-i",
            str(baseline),
            "-lavfi",
            f"[0:v]setpts=PTS-STARTPTS[dist];[1:v]setpts=PTS-STARTPTS[ref];"
            f"[dist][ref]{filter_name}",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    match = re.search(pattern, result.stderr)
    if not match:
        raise ScoreError(f"{filter_name} did not emit its summary")
    return float(match.group(1))


def write_contact_sheet(
    ffmpeg: str,
    baseline: Path,
    candidate: Path,
    destination: Path,
    frame_count_value: int,
) -> None:
    sample_count = min(8, frame_count_value)
    indices = sorted(
        {
            round(index * (frame_count_value - 1) / max(1, sample_count - 1))
            for index in range(sample_count)
        }
    )
    selection = "+".join(f"eq(n\\,{index})" for index in indices)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(candidate),
            "-i",
            str(baseline),
            "-filter_complex",
            f"[0:v]select={selection},scale=320:-2,tile=4x2:padding=2:margin=2[dist];"
            f"[1:v]select={selection},scale=320:-2,tile=4x2:padding=2:margin=2[ref];"
            "[ref][dist]vstack=inputs=2",
            "-frames:v",
            "1",
            "-y",
            str(destination),
        ]
    )


def decode_audio(
    ffmpeg: str,
    path: Path,
    sample_rate: int,
    channels: int,
) -> array[float]:
    result = run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-acodec",
            "pcm_f32le",
            "-f",
            "f32le",
            "-",
        ],
        capture_binary=True,
    )
    samples = array("f")
    samples.frombytes(result.stdout)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def audio_metrics(baseline: array[float], candidate: array[float]) -> dict[str, Any]:
    sample_count = min(len(baseline), len(candidate))
    if sample_count == 0:
        raise ScoreError("decoded audio is empty")
    baseline_energy = 0.0
    candidate_energy = 0.0
    difference_energy = 0.0
    dot = 0.0
    baseline_peak = 0.0
    candidate_peak = 0.0
    baseline_clipped = 0
    candidate_clipped = 0
    for index in range(sample_count):
        baseline_value = float(baseline[index])
        candidate_value = float(candidate[index])
        difference = candidate_value - baseline_value
        baseline_energy += baseline_value * baseline_value
        candidate_energy += candidate_value * candidate_value
        difference_energy += difference * difference
        dot += baseline_value * candidate_value
        baseline_peak = max(baseline_peak, abs(baseline_value))
        candidate_peak = max(candidate_peak, abs(candidate_value))
        baseline_clipped += abs(baseline_value) >= 0.999
        candidate_clipped += abs(candidate_value) >= 0.999
    denominator = math.sqrt(baseline_energy * candidate_energy)
    correlation = dot / denominator if denominator > 0 else None
    relative_l2 = math.sqrt(difference_energy / baseline_energy) if baseline_energy > 0 else None
    return {
        "compared_samples": sample_count,
        "baseline_samples": len(baseline),
        "candidate_samples": len(candidate),
        "zero_lag_correlation": correlation,
        "relative_l2": relative_l2,
        "baseline_rms": math.sqrt(baseline_energy / sample_count),
        "candidate_rms": math.sqrt(candidate_energy / sample_count),
        "baseline_peak": baseline_peak,
        "candidate_peak": candidate_peak,
        "baseline_clipped_fraction": baseline_clipped / sample_count,
        "candidate_clipped_fraction": candidate_clipped / sample_count,
    }


def tsv_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.9g}"
    return str(value)


def main() -> int:
    args = parse_args()
    try:
        ffmpeg = resolve_tool("MERERUN_FFMPEG", "ffmpeg")
        ffprobe = resolve_tool("MERERUN_FFPROBE", "ffprobe")
        baseline_probe = probe_media(args.baseline, ffprobe)
        candidate_probe = probe_media(args.candidate, ffprobe)
        baseline_summary = media_summary(baseline_probe)
        candidate_summary = media_summary(candidate_probe)
        expected = {
            "width": args.expected_width,
            "height": args.expected_height,
            "frame_count": args.expected_frames,
            "frame_rate": args.expected_fps,
            "sample_rate": args.expected_sample_rate,
            "channels": args.expected_channels,
        }
        issues = structural_issues(baseline_summary, candidate_summary, expected)
        if issues:
            raise ScoreError("; ".join(issues))

        ssim = video_filter_metric(
            ffmpeg,
            args.baseline,
            args.candidate,
            "ssim",
            r"All:([0-9.]+)",
        )
        psnr = video_filter_metric(
            ffmpeg,
            args.baseline,
            args.candidate,
            "psnr",
            r"average:([0-9.]+|inf)",
        )
        vmaf = video_filter_metric(
            ffmpeg,
            args.baseline,
            args.candidate,
            "libvmaf=n_threads=4",
            r"VMAF score: ([0-9.]+)",
        )
        audio = baseline_summary["audio"]
        if audio is None or audio["sample_rate"] is None or audio["channels"] is None:
            raise ScoreError("audio stream metadata is incomplete")
        baseline_audio = decode_audio(
            ffmpeg,
            args.baseline,
            audio["sample_rate"],
            audio["channels"],
        )
        candidate_audio = decode_audio(
            ffmpeg,
            args.candidate,
            audio["sample_rate"],
            audio["channels"],
        )
        waveform = audio_metrics(baseline_audio, candidate_audio)
        contact_sheet = None
        if args.contact_sheet is not None:
            video = candidate_summary["video"]
            if video is None or video["frame_count"] is None:
                raise ScoreError("video frame count is unavailable for contact-sheet generation")
            write_contact_sheet(
                ffmpeg,
                args.baseline,
                args.candidate,
                args.contact_sheet,
                video["frame_count"],
            )
            contact_sheet = str(args.contact_sheet.resolve())
        report = {
            "schema_version": 1,
            "baseline": {"path": str(args.baseline.resolve()), "media": baseline_summary},
            "candidate": {"path": str(args.candidate.resolve()), "media": candidate_summary},
            "expected": expected,
            "structure": {"passed": True, "issues": []},
            "video_diagnostics": {
                "ssim": ssim,
                "psnr_db": None if math.isinf(psnr) else psnr,
                "psnr_infinite": math.isinf(psnr),
                "vmaf": vmaf,
            },
            "audio_diagnostics": waveform,
            "contact_sheet": contact_sheet,
            "acceptance_note": (
                "These matched-seed metrics detect trajectory and integrity changes; they do not "
                "replace blinded visual review, semantic/reference retention review, or audio "
                "intelligibility and synchronization review."
            ),
        }
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            "\t".join(
                [
                    args.candidate.stem,
                    "passed",
                    tsv_value(ssim),
                    tsv_value(psnr),
                    tsv_value(vmaf),
                    tsv_value(waveform["zero_lag_correlation"]),
                    tsv_value(waveform["relative_l2"]),
                    str(args.json_path),
                ]
            )
        )
        return 0
    except ScoreError as error:
        failure = {
            "schema_version": 1,
            "baseline": str(args.baseline.resolve()),
            "candidate": str(args.candidate.resolve()),
            "structure": {"passed": False, "issues": [str(error)]},
        }
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        print(f"h3-bakeoff-score: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
