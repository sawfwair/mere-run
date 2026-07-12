# MoGe-2 ViT-S Normal

This module is the native Swift/MLX implementation of the pinned permissive
MoGe-2 ViT-S Normal checkpoint used for single-image metric geometry.

`MoGe2Generator` admits a bounded immutable image snapshot, resolves exact
managed ONNX weights plus installed license evidence, applies reference
preprocessing, and runs the DINOv2 backbone and MoGe heads in MLX. The
postprocessor returns metric points and depth, normals, validity, and camera
intrinsics. Shared geometry exporters add previews, EXR, camera JSON, PLY,
input/checkpoint provenance, and artifact hashes.

Token count is centrally capped for CLI and API requests so image resolution
cannot create an unbounded attention workload.
