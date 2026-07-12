# Native geometry interchange

Shared geometry types, input admission, provenance, and deterministic artifact
writers live here.

## Boundaries

- `VFXImageInputValidator` copies caller-controlled inputs into bounded private
  snapshots while streaming exact byte counts and SHA-256 identities.
- `GeometryArtifactExporter` publishes MoGe metric depth, points, normals,
  validity, camera data, previews, and PLY with a schema-versioned manifest.
- `MultiViewGeometryExporter` publishes DA3 per-view depth/confidence, cameras,
  point-cloud PLY/GLB, and a camera-plus-point 3DGS initialization handoff.
- Point-cloud and geometry value types validate dimensions, finite values,
  coordinate systems, and units before serialization.

`PointCloudGLBWriter` and the Nerfstudio transforms writer inside
`MultiViewGeometryExporter` use heterogeneous dictionaries only at their
standards-defined serialization boundaries; both are explicitly inventoried by
the readiness gate.
