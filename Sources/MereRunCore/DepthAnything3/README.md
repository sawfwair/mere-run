# Depth Anything 3 Small

This module is the native Swift/MLX implementation of the pinned permissive
Depth Anything 3 Small checkpoint used for multi-view geometry and camera
solving.

## Runtime flow

1. `DepthAnything3Resources` resolves and verifies the exact managed weights,
   configuration, source revision, and installed license evidence.
2. `DepthAnything3Preprocessor` admits ordered immutable image snapshots and
   applies the reference resize and normalization policy.
3. `DepthAnything3Model` and its backbone/DPT components run native MLX
   inference; `DepthAnything3Camera` handles predicted or caller-conditioned
   cameras through throwing validation APIs.
4. `DepthAnything3Postprocessor` produces depth, confidence, intrinsics, and
   extrinsics for `MultiViewGeometryExporter`.

The public output is camera-aware point geometry, not a claimed mesh or trained
3D Gaussian field. The 3DGS handoff contains transforms plus a colored point
cloud and explicitly records that Gaussian parameters are absent.

`DepthAnything3GenerationOperation` shares validated settings, current-image and
camera checks, native execution, transactional scene export, and awaited
unloading across CLI and API. `DepthAnything3CameraValidation` checks supplied
cameras against the current headers during planning and the admitted snapshot
dimensions during execution. Callers retain model and transport policy.
