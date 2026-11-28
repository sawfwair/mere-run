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

`MoGe2GenerationOperation` owns complete CLI/API request execution and awaited
unloading. Its validated settings and observational plan share dimension and
token-grid rules with native execution. Callers retain admission, transport,
and presentation. The generator retains its public compatibility interface,
immutable input snapshots, checkpoint validation, model computation, and export.
