# InstantMesh reconstruction runtime

This module is a native Swift/MLX port of the pinned permissive InstantMesh
reconstruction stage.

## Scope and trust contract

- The command accepts exactly four or six user-supplied ordered views. It does
  not bundle or invoke Zero123++ view generation.
- `InstantMeshResources` requires the exact converted package, conversion
  provenance, source checkpoint identity, and license evidence.
- Inputs are admitted as immutable, size-bounded snapshots before model
  resolution or decoding.
- `InstantMeshPreprocessor` performs the reference bicubic preprocessing and
  deterministic camera rig construction with throwing validation.
- `InstantMeshModel`, renderer, and isosurface extractor produce native geometry
  and color fields; the exporter writes OBJ, PLY, GLB, and a hashed run manifest.

The runtime uses its own clean-room isosurface implementation. NVIDIA's
proprietary FlexiCubes source is not included or executed.
