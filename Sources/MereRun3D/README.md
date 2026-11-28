# MereRun3D

3D reconstruction, geometry estimation, single-image depth, video depth, and mesh/point-cloud export runtimes carved out of `MereRunCore`.

- `Asset3D/` and `Geometry/`: shared 3D geometry types, mesh/point-cloud GLB exporters, and multi-view geometry export.
- `DepthAnything3/`, `VisionGeometry/MoGe2/`, `MarigoldV2/`, and `VideoDepth/`: monocular/multi-view geometry and depth generation operations.
- `TripoSR/`, `InstantMesh/`, `Trellis2/`, and `Idol/`: image-to-3D and multi-view 3D reconstruction pipelines.

Model catalog metadata and checkpoint pin definitions (`GeometryModelPins`, `*Resources` metadata) remain in `MereRunCore` so model discovery and validation stay centralized while on-device consumers (such as iOS Studio) avoid compiling or linking desktop 3D runtimes.
