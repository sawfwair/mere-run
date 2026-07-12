# Vision Depth Video

## Purpose

`mere.run vision depth-video` runs Video Depth Anything Small entirely in the
native Swift/MLX runtime. It extracts the source video, processes the upstream
32-frame windows with 10-frame overlap, aligns relative-depth windows, and
writes one floating-point depth EXR and review PNG per source frame.

Pull either audited Apache-2.0 checkpoint, then run the corresponding model id:

```bash
mere.run model pull vision-depth-vda-small
mere.run vision depth-video ./shot.mp4 \
  --model vision-depth-vda-small \
  --output ./shot-depth \
  --json

mere.run model pull vision-depth-vda-small-metric
mere.run vision depth-video ./shot.mp4 \
  --model vision-depth-vda-small-metric \
  --output ./shot-depth-metric \
  --json
```

`--model` also accepts either exact pinned upstream `.pth` file. A release-tool
conversion directory containing `model.safetensors`, `config.json`, and
`SOURCE.json` is accepted after its source identity and generated-file checksum
are verified. Inference never runs Python or deserializes arbitrary pickle.

Inspect a verified execution plan without loading the MLX graph:

```bash
mere.run vision depth-video ./shot.mp4 \
  --model /models/vda-small-native \
  --dry-run
```

The output directory contains `depth-sequence-manifest.json`, a `frames/`
directory of depth EXRs and PNG previews, and `depth-review.mp4` assembled at
the source FPS. Relative-model values are affine-relative; metric-model values
are camera-space Z depth in meters.

Requests are bounded before checkpoint or output work begins. `--input-size`
accepts 14 through 1008. `--max-frames` defaults to 240 and accepts at most
2400. Decoded frames are also checked for side length, per-frame pixels, total
extracted pixels, and the patch-aligned network tensor budget; reduce
`--max-frames` for high-resolution plates that exceed the aggregate budget.

The local API exposes the same native path with multipart upload-only input:

```bash
curl http://127.0.0.1:8080/v1/vision/depth-video \
  -F model=vision-depth-vda-small \
  -F input_size=518 \
  -F max_frames=240 \
  -F video=@shot.mp4
```

Only `vision-depth-vda-small` and `vision-depth-vda-small-metric` are accepted.
The route rejects client-supplied filesystem or output paths, writes the upload
and result under server-owned temporary directories, and returns hashed `file:`
URLs for the manifest, source-FPS review MP4, and every per-frame EXR/PNG.

VDA does not estimate confidence or camera intrinsics. The command therefore
does not fabricate confidence maps, intrinsics, point clouds, or reprojected
geometry. Those fields remain explicitly absent in the structured result.

Alignment is duration-bounded: each completed window is affine-aligned and
spooled immediately, while only the eight frames that a later window can still
crossfade remain in memory. EXRs and the sequence-wide normalized previews are
then emitted one frame at a time. Encoder and DPT-tail work is micro-batched by
default on Apple Silicon; temporal attention still spans each 32-frame window
to preserve the authoritative graph. Increase the bounded `--max-frames`
control explicitly when a shot needs more than the 240-frame production
default.

## Sources

- [Video Depth Anything at the pinned source revision](https://github.com/DepthAnything/Video-Depth-Anything/tree/4f5ae23172ba60fd7bc11ef671cca678842c7072)
- [Relative VDA-S model repository](https://huggingface.co/depth-anything/Video-Depth-Anything-Small)
- [Metric VDA-S model repository](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small)
