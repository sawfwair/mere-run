# Video Depth Anything Small

This module is the native Swift/MLX implementation of the pinned permissive
Video Depth Anything Small relative and metric variants.

## Runtime contract

- `VideoDepthAnythingResources` accepts only exact source or reproducibly
  converted packages with pinned file hashes, conversion environment, source
  code revision, and license evidence.
- `VideoDepthAnythingLimits` centralizes bounded input size, encoded bytes,
  decoded dimensions/pixels, network dimensions, frame count, and aggregate
  decode budget for CLI and API requests.
- The generator snapshots the input video before model resolution, extracts
  bounded frames, runs the reference overlapping temporal window policy, and
  streams deterministic depth artifacts.
- Relative and metric checkpoints retain distinct depth semantics and exact
  runtime checkpoint digests in every output manifest.

Inference is native MLX. Python is used only by the offline, exact-version
conversion and parity tooling under `scripts/`.
