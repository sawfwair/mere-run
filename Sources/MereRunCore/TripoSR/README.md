# TripoSR reconstruction runtime

This module is the native Swift/MLX implementation of the pinned permissive
TripoSR single-image reconstruction checkpoint.

## Runtime flow

1. `TripoSRResources` verifies the exact converted package, source checkpoint,
   conversion provenance, and license evidence.
2. `TripoSRPreprocessor` consumes an immutable admitted image snapshot and
   applies deterministic foreground and image-conditioning transforms.
3. `TripoSRModel` and `TripoSRRenderer` predict the scene representation and
   query density/color fields under explicit memory controls.
4. `TripoSRIsosurface` extracts a mesh; `TripoSRAssetExporter` writes OBJ, PLY,
   GLB, input identity, checkpoint identity, timings, and artifact hashes.

Configuration and memory limits are throwing public APIs so malformed external
requests cannot terminate the process through a precondition trap.
