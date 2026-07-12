# Vision Geometry Multiview

## Purpose

`mere.run vision geometry-multiview` runs the Apache-2.0 Depth Anything 3
Small checkpoint entirely in native Swift/MLX. It jointly predicts consistent
relative depth, per-pixel confidence, OpenCV world-to-camera extrinsics, and
pixel-space intrinsics for one or more ordered images. Optional supplied
cameras activate the checkpoint's real pose-conditioning encoder.

Pull the exact audited model once, then solve a scene:

```bash
mere.run model pull vision-geometry-da3-small
mere.run vision geometry-multiview \
  ./view-00.png ./view-01.png ./view-02.png \
  --model vision-geometry-da3-small \
  --output ./scene-da3 \
  --json
```

Use `--process-resolution` to control image-token memory, `--reference-view`
to choose `first`, `middle`, `saddle-balanced`, or
`saddle-similarity-range`, and `--dry-run` to verify all input/model pins
without allocating the graph.

For pose-conditioned inference, pass a versioned camera document containing
one original-image camera per input, in the same order:

```json
{
  "schemaVersion": 1,
  "cameras": [
    {
      "intrinsics": {
        "imageWidth": 1920,
        "imageHeight": 1080,
        "normalizedFX": 0.9,
        "normalizedFY": 1.6,
        "normalizedCX": 0.5,
        "normalizedCY": 0.5
      },
      "extrinsics": {
        "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "translation": [0, 0, 0]
      }
    }
  ]
}
```

```bash
mere.run vision geometry-multiview ./view-00.png \
  --cameras ./cameras.json \
  --output ./conditioned-scene
```

The output contains per-view depth/confidence EXRs and review PNGs, processed
RGB inputs, `cameras.json`, deterministic colored `scene.ply`, and
`scene.glb`. The GLB is a point-cloud primitive, not a triangle mesh. Depth and
scene coordinates remain explicitly relative for DA3-Small.

`transforms.json` plus `scene.ply` form a Nerfstudio/3DGS initialization
handoff. They contain calibrated cameras and colored seed points only. DA3-Small
does not predict Gaussian scales, rotations, spherical harmonics, or opacity,
so the manifest explicitly reports `containsGaussianParameters: false`.

The model confidence is the authoritative `exp(logit) + 1` output. By default,
points below its 40th percentile are excluded; change this with
`--confidence-percentile`. Point selection is deterministic and bounded with
`--max-points`.

## Sources

- [DA3-Small checkpoint](https://huggingface.co/depth-anything/DA3-SMALL)
- [Depth Anything 3 source](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/41736238f5bced4debf3f2a12375d2466874866d)
