#!/usr/bin/env python3
"""Export the released DreamX AR trajectory dialect for Swift parity."""

import argparse
import importlib.util
import json
from pathlib import Path


def load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dreamx-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np

    trajectory = load_source_module(
        "dreamx_ar_trajectory_processor",
        args.dreamx_root / "utils" / "trajectory_processor.py",
    )
    segments = [("w", 3), ("j", 5)]
    pixel_frame_count = 21
    speed = 1.5
    _, camera_parameters, _ = trajectory.generate_trajectory_from_json(
        trajectory_spec=segments,
        num_frames=pixel_frame_count,
        speed=speed,
        return_cam_params=True,
    )
    cameras = [trajectory.Camera(row.tolist()) for row in camera_parameters]
    aligned_indices = [0] + list(range(1, pixel_frame_count, 4))
    cameras = [cameras[index] for index in aligned_indices]
    relative_c2w = []
    for chunk_start in range(0, len(cameras), 3):
        chunk_end = min(chunk_start + 3, len(cameras))
        if chunk_start == 0:
            chunk = cameras[chunk_start:chunk_end]
            relative_c2w.extend(trajectory.get_relative_pose(chunk))
        else:
            chunk = [cameras[chunk_start - 1]] + cameras[chunk_start:chunk_end]
            relative_c2w.extend(trajectory.get_relative_pose(chunk)[1:])
    views = np.linalg.inv(np.asarray(relative_c2w, dtype=np.float32)).astype(np.float32)
    intrinsic = np.zeros((len(views), 3, 3), dtype=np.float32)
    intrinsic[:, 0, 0] = 969.6969696969696 / (960 * 2)
    intrinsic[:, 1, 1] = 969.6969696969696 / (540 * 2)
    intrinsic[:, 0, 2] = 0.5
    intrinsic[:, 1, 2] = 0.5
    intrinsic[:, 2, 2] = 1
    output = {
        "source_revision": "AMAP-ML/DreamX-World@f2bf6bf",
        "trajectory": {
            "segments": segments,
            "pixel_frame_count": pixel_frame_count,
            "speed": speed,
            "chunk_relative": True,
        },
        "view_matrices": {
            "shape": [1, len(views), 4, 4],
            "values": views.reshape(-1).tolist(),
        },
        "intrinsics": {
            "shape": [1, len(views), 3, 3],
            "values": intrinsic.reshape(-1).tolist(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
