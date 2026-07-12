# Native Multi-view Image to 3D

## Purpose

`mere.run image reconstruct-3d-multiview` reconstructs a normalized colored
object mesh from exactly four or six ordered views with the Apache-2.0
InstantMesh Base reconstruction network. The VFX-oriented alias is
`mere.run vision image-to-3d-multiview`.

The command does not create missing views. Every view must be supplied by the
user and must be licensed for the intended production. The Zero123++ view
generator, its noncommercial derivative weights, and all diffusion inference
are excluded from mere.run.

## Prepare the checkpoint once

The upstream Hugging Face repository publishes a Lightning `.ckpt`. A managed
pull downloads and checksum-verifies that exact source artifact, but mere.run
will never deserialize its Pickle payload at runtime:

```bash
mere.run model pull image-3d-instantmesh-base
mere.run model info image-3d-instantmesh-base
```

The first `model info` line prints the managed model root. Use that root with
the audited offline converter (quote paths because the default macOS root
contains spaces):

```bash
MODEL_ROOT="/path/from/model-info"
python3 scripts/model-conversion/convert_instantmesh_base.py \
  --source "$MODEL_ROOT/instant_mesh_base.ckpt" \
  --output "$MODEL_ROOT/native" \
  --license-file /path/to/InstantMesh/LICENSE
```

The converter verifies the pinned 1,253,574,354-byte source and SHA-256, uses
PyTorch's weights-only loader outside the runtime, checks the frozen 455-tensor
inventory, requires CPython 3.11.15 with the exact packages in
`scripts/model-conversion/requirements-vfx.txt`, and writes deterministic
`model.safetensors`, `config.json`, `SOURCE.json`, and the exact upstream
`LICENSE`. Runtime preflight verifies all four files byte for byte. The `native`
child keeps the pulled source and mere.run manifest
intact while making the managed model id runnable by the CLI and API. You may
instead convert anywhere else and pass that directory explicitly to the CLI.
Python is conversion tooling only; inference is native Swift/MLX.

## Reconstruct

Pass views in the released model's expected turntable order. The six-view rig
uses azimuth/elevation pairs `30/+20`, `90/-10`, `150/+20`, `210/-10`,
`270/+20`, and `330/-10` degrees. Four-view input uses six-view entries 0, 2,
4, and 5, in that order:

```bash
mere.run image reconstruct-3d-multiview \
  --view ./view-0.png \
  --view ./view-1.png \
  --view ./view-2.png \
  --view ./view-3.png \
  --model "$HOME/Models/instantmesh-base-native" \
  --output ./object-3d \
  --json
```

Six-view input uses six repeated `--view` options. Input alpha is composited on
white and every image is resized to the checkpoint's 320 by 320 conditioning
size. Without `--cameras`, mere.run applies the released deterministic camera
conditioning convention; this does not synthesize images. For calibrated
inputs, pass a JSON document with one row-major C2W 3x4 matrix followed by
`fx, fy, cx, cy` per view. The snippet below shows one row only; the real file
must repeat the row shape exactly four or six times to match the uploads:

```json
{
  "schemaVersion": 1,
  "cameras": [
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 4, 1.866, 1.866, 0.5, 0.5]
  ]
}
```

The CLI and API inspect decoded dimensions before allocating full image
buffers. Each view is limited to 16,384 pixels per side and 64 million pixels;
the ordered set is limited to 256 million pixels. These limits apply even when
the compressed uploads themselves are small.

The camera count must equal the view count. `--dry-run` verifies every input,
camera, and converted checkpoint artifact without loading the graph.

`--resolution` defaults to the trained 128-cell grid and can range from 2 to
256. Grid evaluation grows cubically. `--no-vertex-colors` skips neural color
queries when geometry alone is sufficient.

## Outputs and topology boundary

The output directory contains deterministic same-geometry OBJ, binary PLY, and
GLB triangle meshes, a shared mesh manifest, and an authoritative
schema-versioned InstantMesh run manifest. The run manifest durably records
ordered views and hashes, source/prepared sizes, all 16 camera-conditioning
values per view, converted/source pins, extraction controls, whether the pinned
upstream empty-field sign repair was applied, exclusions, topology caveat,
mesh summary, and the hash of each mesh plus the shared mesh manifest.
Structured output returns hashes and paths for both manifests.

Coordinates are normalized object space with X right, Y up, and Z forward, not
meters. The result is marked `inferredUnseenGeometry: true` because the model
reconstructs a field between supplied observations.

Field inference, SDF, deformation, and learned color are the pinned InstantMesh
network. Meshing is deterministic native marching tetrahedra. mere.run does not
port NVIDIA's separately licensed FlexiCubes/renderer implementation. mere.run
does not claim triangle-topology parity with upstream FlexiCubes. When the
learned field has only one sign in its interior, mere.run reproduces the pinned
upstream reconstruction graph's documented center/boundary sign repair before
meshing and records that fact per run.

## Local API

The API accepts only uploaded image bytes and the managed model id; it rejects
client filesystem paths and uploaded checkpoints:

```bash
curl http://127.0.0.1:8080/v1/vision/image-to-3d-multiview \
  -F model=image-3d-instantmesh-base \
  -F resolution=128 \
  -F 'image[]=@view-0.png' \
  -F 'image[]=@view-1.png' \
  -F 'image[]=@view-2.png' \
  -F 'image[]=@view-3.png'
```

The server's managed model directory must contain the verified converted
package. Installing only the upstream `.ckpt` returns the same explicit
offline-conversion requirement as the CLI.

## Sources

- [Pinned InstantMesh checkpoint](https://huggingface.co/TencentARC/InstantMesh/tree/b785b4ecfb6636ef34a08c748f96f6a5686244d0)
- [Pinned InstantMesh source](https://github.com/TencentARC/InstantMesh/tree/08822c52fdc399b93ea00e4fa9e596344ed52ccc)
