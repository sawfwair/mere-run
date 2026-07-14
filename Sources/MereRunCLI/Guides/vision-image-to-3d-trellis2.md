# Native TRELLIS.2 Image to 3D

## Purpose

`mere.run vision image-to-3d-trellis2` reconstructs one isolated object as a
512-resolution O-Voxel mesh with base color, metallic, roughness, and alpha.
The equivalent image-family command is `mere.run image reconstruct-3d-trellis2`.
All neural stages run in native Swift/MLX from Microsoft's official
safetensors; no Python, PyTorch, CUDA, or converted weight copy is used.

First accept the DINOv3 license on Hugging Face and authenticate the local Hub
client. Then install the pinned model graph and run reconstruction:

```bash
mere.run model pull image-3d-trellis2-4b
mere.run vision image-to-3d-trellis2 ./chair.png \
  --output ./chair-trellis2 \
  --seed 42 \
  --json
```

The default foreground policy requires meaningful transparent alpha and crops
the object before 512px DINOv3 conditioning. Fully opaque images fail with an
actionable error. Pass `--already-framed` only when the image is already an
isolated, centered object on an appropriate black background.

`--max-tokens` is a decoded O-Voxel safety limit and defaults to 2,097,152.
Raising it can materially increase unified-memory use. This is deliberately
separate from Microsoft's 49,152-token cascade-resolution budget, which does
not cap the direct 512 decoder. The three flow stages use the official 12-step
Euler schedules. `--seed` controls deterministic MLX noise and defaults to 42.

Use `--dry-run` to validate the input, checksum every pinned component, and
print the plan without loading a neural graph:

```bash
mere.run image reconstruct-3d-trellis2 ./chair.png --dry-run --already-framed
```

The exported mesh is, by default, the watertight narrow-band dual-contour
envelope of the raw crust (band 1 voxel, no projection), mirroring upstream's
shipped `to_glb` remesh: the flexible dual grid's open seams and pinholes are
sealed and the material renders single-sided. Crust tears wider than roughly
twice the band survive as tunnels; for fuzzy or furry subjects raise
`--remesh-band` to 2 or 3 (each unit inflates the surface by one voxel,
about 0.2% of the object). Colors are always sampled at the closest point on
the original crust, so wider bands do not wash out or darken the appearance.
Pass `--no-remesh` to export the raw porous dual-grid crust instead
(double-sided, upstream's non-remesh topology).

The output directory contains OBJ, binary PLY, and GLB files with sampled RGBA
vertex color; a `-textured.glb` whose per-quad atlas bakes the field into
standard glTF PBR textures; a shared mesh manifest; a `.pbrvox` sparse material
field; and an authoritative TRELLIS.2 run manifest. `.pbrvox` preserves base
color RGB, metallic, roughness, and alpha for every decoded O-Voxel. All
artifacts and all seven checkpoint components are named, pinned, and hashed in
provenance.

Coordinates are normalized object space with X right, Y up, and Z forward.
They are not meters, and unseen geometry is inferred from the single image.
The `.pbrvox` sidecar instead preserves integer indices in TRELLIS.2's native
Z-up O-Voxel grid so its six-channel material field stays lossless.

This command currently supports the official 512 pipeline. It does not claim
support for the 1024/1536 cascade or byte-identical mesh ordering with the CUDA
reference implementation.

## Sources

- [Microsoft TRELLIS.2 source](https://github.com/microsoft/TRELLIS.2)
- [Pinned TRELLIS.2-4B checkpoint](https://huggingface.co/microsoft/TRELLIS.2-4B/tree/af44b45f2e35a493886929c6d786e563ec68364d)
- [TRELLIS.2 paper](https://arxiv.org/abs/2512.14692)
- [DINOv3 source](https://github.com/facebookresearch/dinov3)
- [Pinned DINOv3 ViT-L/16 checkpoint](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/tree/ea8dc2863c51be0a264bab82070e3e8836b02d51)
