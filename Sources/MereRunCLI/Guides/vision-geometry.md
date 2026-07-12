# Native image geometry

`mere.run vision geometry` runs MoGe-2 ViT-S Normal inside the native Swift/MLX
runtime. It produces metric camera-space depth and points, normals, a validity
mask, centered camera intrinsics, compositor previews, and a colored PLY point
cloud. The authoritative deployment weights are pinned and checksum-verified.

```bash
mere.run model pull vision-geometry-moge2-small
mere.run vision geometry ./frame.png --output ./frame-geometry --json
```

Inspect the plan without loading or downloading weights:

```bash
mere.run vision geometry ./frame.png --dry-run
```

Quality is controlled with `--resolution-level 0...9` (default `9`) or an
explicit `--token-count`. Use `--max-points` to cap only the PLY density; EXR
and camera outputs remain full resolution.

Coordinate convention is X right, Y down, Z forward. Depth and point values are
meters. Pixel centers are used for reprojection, and both normalized and pixel
camera matrices are written to the camera JSON and run manifest.

The local API exposes the same native generator through a multipart endpoint:

```bash
curl http://127.0.0.1:8080/v1/vision/geometry \
  -F model=vision-geometry-moge2-small \
  -F resolution_level=9 \
  -F image=@frame.png
```

The server owns the upload and output locations; the request cannot supply
arbitrary filesystem paths. The response includes camera/depth metadata plus
`file:` URLs, byte counts, and SHA-256 values for every artifact and manifest.
