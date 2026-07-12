# Native 3D asset interchange

This module owns the model-independent mesh value type and deterministic OBJ,
PLY, and binary glTF writers used by native reconstruction runtimes.

## Contract

- `MeshAsset` validates topology, optional normals, vertex colors, texture
  coordinates, units, and coordinate-system metadata before export.
- `MeshArtifactExporter` writes into a staging directory and publishes an
  artifact manifest only after every file has been hashed.
- Input identity records are supplied by the inference boundary after immutable
  admission; exporters reject a mismatched input count or path.
- `MeshGLBWriter` builds glTF's heterogeneous JSON object at the serialization
  boundary, which is why it is listed in the repository's dynamic-JSON
  inventory. No untyped JSON escapes this writer.

Model-specific inference and provenance belong in `TripoSR` and `InstantMesh`.
