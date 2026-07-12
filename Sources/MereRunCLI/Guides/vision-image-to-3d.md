# Native Image to 3D

## Purpose

`mere.run vision image-to-3d` reconstructs one object image with the
MIT-licensed TripoSR checkpoint entirely in native Swift/MLX. The equivalent
catalog-oriented spelling is `mere.run image reconstruct-3d`.

Pull the exact audited checkpoint once, then reconstruct an asset:

```bash
mere.run model pull image-3d-triposr
mere.run vision image-to-3d ./chair.png \
  --output ./chair-3d \
  --json
```

Use `--dry-run` to checksum-verify the checkpoint and inspect the complete
execution plan without loading the 419-million-parameter graph:

```bash
mere.run image reconstruct-3d ./chair.png --dry-run
```

Transparent PNG input is cropped, square-padded to `--foreground-ratio 0.85`,
and composited on the checkpoint's gray background. An opaque image is treated
as already prepared; mere.run does not silently run a background-removal
sidecar. Use `--already-framed` to skip crop/pad explicitly.

Admission is based on decoded dimensions, not compressed file size. The CLI
and API inspect the image header before decoding and reject inputs above 16,384
pixels on either side or 64 million pixels total, including small compressed
files that advertise an unsafe decoded canvas.

`--resolution` controls the native density grid and defaults to `256`. Runtime
and density memory grow cubically, so use `128` for faster iteration before a
final high-resolution extraction. `--density-threshold` defaults to the
checkpoint's authoritative activated-density value of `25`. Use
`--no-vertex-colors` only when geometry without learned vertex color is enough.

The output directory contains same-geometry OBJ, binary PLY, and GLB triangle
meshes, a shared mesh manifest, and an authoritative schema-versioned TripoSR
run manifest. The run manifest durably records checkpoint format and source
pin, foreground policy and crop result, extraction resolution and threshold,
vertex-color policy, native topology algorithm, mesh counts/bounds, and hashes
for the shared manifest plus every mesh. The command's structured result also
reports both manifest paths and SHA-256 values plus stage timings.

Coordinates are normalized object space with X right, Y up, and Z forward;
they are not meters. The single input image cannot observe the back of the
object, so manifests explicitly mark `inferredUnseenGeometry: true`.

Mesh extraction uses deterministic native marching tetrahedra over the same
TripoSR density field and threshold. It represents the same sampled isosurface,
but triangle topology is not expected to match upstream `torchmcubes` marching
cubes byte-for-byte.

The local API exposes the same native generator through server-owned uploads
and output locations:

```bash
curl http://127.0.0.1:8080/v1/vision/image-to-3d \
  -F model=image-3d-triposr \
  -F resolution=256 \
  -F foreground_ratio=0.85 \
  -F image=@chair.png
```

Client filesystem paths and uploaded checkpoint files are rejected. The API
accepts only the managed model ID and returns the run manifest as `manifest`,
the shared geometry manifest as `mesh_manifest`, and hashed `file:` artifact
URLs.

## Sources

- [Pinned TripoSR checkpoint](https://huggingface.co/stabilityai/TripoSR/tree/5b521936b01fbe1890f6f9baed0254ab6351c04a)
- [TripoSR source](https://github.com/VAST-AI-Research/TripoSR/tree/107cefdc244c39106fa830359024f6a2f1c78871)
