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
- Vertex colors are sRGB-encoded bytes (`colorsRGBA8`) because reconstruction
  models regress display-referred pixel values. OBJ and PLY write those bytes
  verbatim per the display-referred de-facto convention of both formats.
  glTF 2.0 defines `COLOR_0` as linear, so `MeshGLBWriter` decodes RGB through
  the exact sRGB EOTF (`VertexColorTransfer`) and stores 16-bit normalized
  values to keep decoded darks from banding. Alpha is linear coverage in every
  format. Standards-compliant viewers (three.js, Blender) therefore display
  identical colors for all three artifacts.

- Exporters may pass uniform `MeshPBRMaterialFactors` (metallic, roughness);
  `MeshGLBWriter` writes them as core glTF material factors and defaults to a
  fully rough dielectric when absent.

Model-specific inference and provenance belong in `TripoSR`, `InstantMesh`, and
`Trellis2`. TRELLIS.2 additionally writes its six-channel sparse PBR field as a
model-specific `.pbrvox` sidecar because the canonical mesh contract currently
stores vertex color, not a generated texture atlas; its field-median metallic
and roughness ride along as the GLB's uniform material factors.
