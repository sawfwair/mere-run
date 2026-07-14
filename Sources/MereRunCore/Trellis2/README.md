# Native TRELLIS.2

This directory is the native Swift/MLX 512-resolution image-to-3D runtime for
Microsoft's TRELLIS.2-4B checkpoint. It consumes the official safetensors
directly; installation does not create converted weight copies and inference
does not launch Python, PyTorch, or CUDA.

## Pipeline

1. `Trellis2ImageConditioning.swift` crops transparent foregrounds, composites
   on black, resizes to 512, and runs the pinned DINOv3 ViT-L/16 conditioner.
2. `Trellis2FlowModel.swift` runs the 12-step sparse-structure, shape, and
   texture flow models. Texture sampling concatenates the normalized shape
   latent at every model evaluation.
3. `Trellis2StructureDecoder.swift` produces occupied 32-grid coordinates.
4. `Trellis2SparseDecoder.swift` decodes the adaptive shape subdivision tree
   and the matching six-channel PBR O-Voxel field using native sparse
   submanifold convolutions.
5. `Trellis2FlexibleDualGrid.swift` extracts the indexed 512-grid mesh, rotates
   O-Voxel's Z-up basis into the canonical Y-up mesh contract, fills small
   boundary loops at the reference threshold, and samples RGBA, metallic, and
   roughness at its vertices.
6. `Trellis2ArtifactExporter.swift` writes canonical OBJ, PLY, and GLB meshes,
   a deterministic `.pbrvox` sidecar that preserves all six PBR channels, and
   a self-contained `-textured.glb` whose per-quad atlas blocks
   (`Trellis2TextureAtlasBaker`) bake the field into an sRGB baseColorTexture
   and a linear metallicRoughnessTexture.

Every large stage is loaded and released separately so the three 1.3B flow
checkpoints and two sparse decoders are not resident together.

## Public boundary

```bash
mere.run model pull image-3d-trellis2-4b
mere.run vision image-to-3d-trellis2 ./object.png --output ./object-3d
# Equivalent catalog-oriented spelling:
mere.run image reconstruct-3d-trellis2 ./object.png --output ./object-3d
```

Transparent alpha is the default foreground contract. Fully opaque input is
rejected unless the caller explicitly passes `--already-framed`; the runtime
does not hide a background-removal service or model behind this command.

The initial native surface is the official 512 pipeline. The 1024/1536 cascade
checkpoints are not represented as supported. GLB/PLY/OBJ carry sampled vertex
RGBA; the additional `-textured.glb` artifact bakes the field into standard
glTF PBR textures over a per-quad block atlas (dual-grid quads recovered from
consecutive triangle pairs; blocks Morton-ordered by quad centroid so mip
levels blend surface-local colors). Metallic, roughness, alpha, and base
color remain losslessly available in the hashed `.pbrvox` sidecar and run
manifest. Mesh artifacts use the canonical X-right, Y-up, Z-forward basis;
`.pbrvox` coordinates remain integer O-Voxel indices in the model's native
Z-up grid so consumers can reproduce material sampling exactly.

The TRELLIS.2 weights are MIT-licensed. DINOv3 is a separately pinned,
license-gated dependency; users must accept its terms and authenticate with
Hugging Face before `model pull` can complete.
