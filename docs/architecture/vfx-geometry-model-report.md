# OSS depth and 3D models for native VFX workflows

Research snapshot: 2026-07-11.

This report records the five permissively licensed model lanes selected for
native Swift/MLX execution in `mere.run`. It separates what each neural model
actually predicts from downstream files that `mere.run` derives, and it makes
the InstantMesh licensing boundary explicit. The exact artifact identities in
this document come from `GeometryModelPins`; upstream capability and license
claims link to the corresponding Hugging Face model card or source repository.

## Recommendation at a glance

| Need | Model | Authoritative input | Native result | Best plugin uses |
| --- | --- | --- | --- | --- |
| Geometry from one plate | MoGe-2 ViT-S Normal | One RGB image | Metric point/depth maps, normals, validity, camera intrinsics | Depth/normal passes, projection setup, relighting inputs, shadow/holdout geometry |
| Stable depth through a shot | Video Depth Anything Small | A decoded video sequence | Temporally aligned relative or metric depth frames | Depth mattes, rack-focus/DOF, fog, occlusion, depth-aware grading |
| Geometry and cameras from views | Depth Anything 3 Small | One or more RGB views; cameras optional | Relative depth, confidence, camera intrinsics/extrinsics, colored points | Camera/point-cloud bootstrap, set reconstruction, 3DGS/Nerfstudio initialization |
| Fast object proxy from one image | TripoSR | One isolated object image | Normalized colored object mesh | Prop proxy, previz asset, collision/holdout mesh, rendered turntable |
| Better object proxy from supplied views | InstantMesh Base, reconstruction only | Exactly 4 or 6 ordered object views; cameras optional | Normalized colored object mesh | Turntable-to-mesh, multi-view prop reconstruction, set dressing and occluders |

Use MoGe for a single scene image, VDA for a temporal plate, DA3 for a
multi-view scene, TripoSR for a single isolated object, and InstantMesh when
four or six licensed object views already exist. These models complement one
another; they are not interchangeable depth checkpoints.

## Reproducibility ledger

All revisions are immutable commit IDs. Byte counts and SHA-256 values are
runtime admission controls, not approximate download sizes.

### MoGe-2 ViT-S Normal

- Managed ID: `vision-geometry-moge2-small`
- Pinned model: [`Ruicheng/moge-2-vits-normal-onnx@e50ffda`](https://huggingface.co/Ruicheng/moge-2-vits-normal-onnx/tree/e50ffda41565591092adea54c6ac83d6212e1e23)
- Pinned source: [`microsoft/MoGe@0744441`](https://github.com/microsoft/MoGe/tree/07444410f1e33f402353b99d6ccd26bd31e469e8)
- `model.onnx`: 140,852,051 bytes
- SHA-256: `24eacb5dc7a2c54c7bc98f7de085ffbed79ad006ea5b664c2c2cdc02ff3a52f0`
- License boundary: MIT for MoGe; the bundled DINOv2 portion is Apache-2.0.
  The conversion repository has no Hugging Face license tag, so distribution
  must retain the notices from the [upstream MoGe project](https://github.com/microsoft/MoGe/blob/07444410f1e33f402353b99d6ccd26bd31e469e8/LICENSE)
  and DINOv2 rather than relying on model-card metadata alone.

### Video Depth Anything Small

- Managed ID: `vision-depth-vda-small`
- Pinned model: [`depth-anything/Video-Depth-Anything-Small@2568753`](https://huggingface.co/depth-anything/Video-Depth-Anything-Small/tree/256875362cff76724b920335dfb4b29dd611f66e)
- Pinned source: [`DepthAnything/Video-Depth-Anything@4f5ae23`](https://github.com/DepthAnything/Video-Depth-Anything/tree/4f5ae23172ba60fd7bc11ef671cca678842c7072)
- `video_depth_anything_vits.pth`: 116,440,756 bytes
- SHA-256: `13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609`
- License: Apache-2.0.

Metric variant:

- Managed ID: `vision-depth-vda-small-metric`
- Pinned model: [`depth-anything/Metric-Video-Depth-Anything-Small@273d090`](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/tree/273d090f2ce17df50c2872d82c8322c45da5b4dd)
- `metric_video_depth_anything_vits.pth`: 116,444,063 bytes
- SHA-256: `3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33`
- License: Apache-2.0.

### Depth Anything 3 Small

- Managed ID: `vision-geometry-da3-small`
- Pinned model: [`depth-anything/DA3-SMALL@e08cab6`](https://huggingface.co/depth-anything/DA3-SMALL/tree/e08cab65ca0ec38e7826075418411ab90cab4da3)
- Pinned source: [`ByteDance-Seed/Depth-Anything-3@4173623`](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/41736238f5bced4debf3f2a12375d2466874866d)
- `config.json`: 1,202 bytes; SHA-256
  `a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2`
- `model.safetensors`: 137,248,940 bytes; SHA-256
  `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`
- License: Apache-2.0.

### TripoSR

- Managed ID: `image-3d-triposr`
- Pinned model: [`stabilityai/TripoSR@5b52193`](https://huggingface.co/stabilityai/TripoSR/tree/5b521936b01fbe1890f6f9baed0254ab6351c04a)
- Pinned source: [`VAST-AI-Research/TripoSR@107cefd`](https://github.com/VAST-AI-Research/TripoSR/tree/107cefdc244c39106fa830359024f6a2f1c78871)
- `config.yaml`: 987 bytes; SHA-256
  `74ca708ce086bf68e97709ea6b3d91f14717921c04691e84043f0eb8fcc68e62`
- `model.ckpt`: 1,677,246,742 bytes; SHA-256
  `429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee`
- License: MIT for code and pretrained model, as stated by both the
  [model card](https://huggingface.co/stabilityai/TripoSR) and
  [source repository](https://github.com/VAST-AI-Research/TripoSR/tree/107cefdc244c39106fa830359024f6a2f1c78871#license).

### InstantMesh Base, reconstruction only

- Managed ID: `image-3d-instantmesh-base`
- Pinned model: [`TencentARC/InstantMesh@b785b4e`](https://huggingface.co/TencentARC/InstantMesh/tree/b785b4ecfb6636ef34a08c748f96f6a5686244d0)
- Pinned source: [`TencentARC/InstantMesh@08822c5`](https://github.com/TencentARC/InstantMesh/tree/08822c52fdc399b93ea00e4fa9e596344ed52ccc)
- `instant_mesh_base.ckpt`: 1,253,574,354 bytes
- SHA-256: `22701cd25201d624ebb1568b93cf91b43a2c32006835c08fe73e1f3c9f6c44b5`
- License boundary: Apache-2.0 reconstruction checkpoint only. The runtime
  excludes view synthesis and NVIDIA's proprietary FlexiCubes implementation.

## Native conversion and runtime status

| Model | Published format | Native admission path | Offline conversion status |
| --- | --- | --- | --- |
| MoGe-2 ViT-S Normal | ONNX | Verified ONNX initializers are mapped directly into the MLX graph; ONNX Runtime is not used | No conversion required |
| VDA-S relative/metric | PyTorch state dict | Exact pinned PTH is parsed by a restricted non-executing tensor reader; exact converted packages are also accepted | Deterministic conversion proven byte-identical twice per variant, with frozen environment, 351-tensor inventory, and mandatory license |
| DA3-S | Safetensors | Exact pinned safetensors and config are checksum-verified and loaded directly | No conversion required |
| TripoSR | Lightning/PyTorch CKPT | Exact CKPT is accepted by the restricted state-dict grammar; verified converted safetensors is also accepted | Deterministic conversion proven byte-identical twice |
| InstantMesh Base | Lightning CKPT | Runtime accepts only the verified reconstruction-only safetensors package | Deterministic conversion proven byte-identical twice; view-generation tensors are never accepted |

Every inference path is native Swift/MLX. Python appears only in audited
release conversion and upstream-reference fixture generation, never as a
runtime sidecar.

## Model profiles

### 1. MoGe-2 ViT-S Normal

The official MoGe project describes the ViT-S Normal model as a roughly 35M
parameter, single-forward-pass estimator for metric points, metric depth,
normals, validity, and camera field of view. Its output point and normal maps
use OpenCV camera coordinates: x right, y down, z forward. See the
[official output contract](https://github.com/microsoft/MoGe/tree/07444410f1e33f402353b99d6ccd26bd31e469e8#minimal-code-example).

`mere.run` reads the exact pinned ONNX initializer archive, maps it into the
native MLX graph, and does not launch ONNX Runtime or Python. Postprocessing
solves focal length and shift, preserves the checkpoint's metric scale, and
exports depth and normal EXRs/PNGs, validity, camera JSON, PLY points, and a
hash-bearing manifest.

This is the strongest still-image VFX primitive in the set because one pass
provides mutually consistent depth, normals, and camera data. It can drive
depth/normal AOVs, approximate shadow catchers, depth-aware relighting,
projection-camera setup, fog/DOF, holdouts, and point-based set proxies. It is
not a multi-view camera solver and does not create a watertight object mesh.

Apple Silicon fit is strong: the small backbone is the lightest still-geometry
choice here, token count is capped and controllable, and inference is entirely
MLX. The published 128x128, 3,600-token native row completed in 0.88 seconds,
with 748.9 MiB max RSS and an 8.50 GiB MLX/Metal peak footprint.

### 2. Video Depth Anything Small and Metric Small

The [official model card](https://huggingface.co/depth-anything/Video-Depth-Anything-Small)
positions VDA as consistent depth for arbitrarily long videos, built on Depth
Anything V2. The exact small checkpoints contain 351 tensors and 29,080,193
serialized scalars; 384 scalars are a training-only DINOv2 mask token, leaving
29,079,809 scalars used by inference.

The model consumes RGB frames in 32-frame windows. `mere.run` reproduces the
upstream 32/10/22 window-overlap schedule, aligns overlaps, and streams final
frames to disk so memory does not grow with shot duration. Outputs are one
depth EXR and preview PNG per source frame, a source-FPS review movie, and a
frame-accurate manifest. The relative checkpoint emits affine-relative depth;
the metric checkpoint emits meters. Neither checkpoint authoritatively emits
camera poses or confidence, so plugins must not synthesize either field.

Native inference can load the exact PTH through a restricted, non-executing
state-dict reader. An audited offline converter can also emit provenance-bound
safetensors after validating every key, dtype, shape, byte count, and source
digest. Python is conversion/reference tooling only.

Recommended plugin features are temporal depth mattes, depth-based holdouts,
atmospheric perspective, fog, rack focus, synthetic DOF, depth-aware grading,
and coarse occlusion ordering. Use the metric variant when meter semantics are
required, but do not treat it as a camera track.

### 3. Depth Anything 3 Small

The [DA3-SMALL model card](https://huggingface.co/depth-anything/DA3-SMALL)
defines relative depth, pose estimation, and pose conditioning. Its concrete
outputs are per-view depth and confidence plus world-to-camera extrinsics and
intrinsics. The selected checkpoint contains 437 float32 tensors and
34,299,463 scalars. The card labels the model "0.08B"; the serialized inventory
and 137,248,940-byte safetensors file are the exact runtime authority.

The native graph implements the DINO backbone, local/global attention, 2D
RoPE, depth/confidence/ray heads, camera decoder, and known-camera encoder.
Input can be one view or a batch of views. When cameras are supplied, the model
uses them for pose-conditioned depth; otherwise it predicts cameras. Depth is
relative in both cases.

`mere.run` exports per-view depth and confidence EXRs, processed images,
camera JSON, a confidence-filtered colored PLY, a glTF 2.0 point-cloud GLB,
and a Nerfstudio/3DGS initialization handoff. The GLB contains points, not
triangles. The 3DGS handoff contains cameras and colored points, not learned
Gaussian parameters. Those distinctions should remain visible in every plugin
contract and UI.

DA3 is the right primitive for multi-view camera bootstrap, plate alignment,
point-based set reconstruction, scene-scale occlusion proxies, and seeding a
separate Nerfstudio or 3DGS optimization. It is not the right primitive for
metric measurements or a finished mesh.

### 4. TripoSR

The [official model card](https://huggingface.co/stabilityai/TripoSR) defines
TripoSR as a feed-forward reconstruction model from one image. The checkpoint
contains 549 tensors and 419,275,628 scalars. The native model is a ViT-B/16
image encoder feeding 3,072 triplane tokens through 16 transformer blocks,
then a 3 x 40 x 64 x 64 scene field with density and color heads.

The runtime accepts either the exact pinned CKPT through the restricted
non-executing state-dict grammar or a deterministic converted safetensors
package. The frozen CPython 3.11.15 / PyTorch 2.13.0 / safetensors 0.8.0
converter rejects any other serialization environment and requires the exact
upstream MIT license. Two independent conversions produced the same
1,677,170,936-byte file with SHA-256
`f72bb520b8b1a5639600ac818496f22d6ccb3b42d3942412bd1e2375ef780a2b`.
The verified package also pins its 378-byte config, 855-byte `SOURCE.json`, and
1,080-byte license.
Attention, feed-forward, field-query, and isosurface chunking keep the exact
graph usable on unified memory.

The result is a normalized object-space mesh with optional learned vertex
colors, exported as OBJ, PLY, GLB, and manifests. It is useful for a fast prop
proxy, collision/holdout geometry, previz set dressing, or a plugin-rendered
turntable. It does not infer real-world scale or a production camera, and a
single input cannot reveal genuinely unseen object detail.

### 5. InstantMesh Base, reconstruction only

The upstream project combines a sparse-view reconstruction network with a
single-image-to-multiview diffusion stage. Only the reconstruction network is
in scope. `mere.run` accepts exactly four or six ordered, user-supplied views,
with either the official camera rig or explicit 16-value camera conditioning.
It never downloads or runs the view generator.

The pinned reconstruction checkpoint contains 455 tensors and 313,352,516
scalars. Its native graph uses a camera-conditioned ViT-B/16 encoder, 3,072
triplane tokens, 12 transformer blocks, a 3 x 40 x 64 x 64 scene field, and
SDF, deformation, color, and cell-weight heads. Because the published
Lightning CKPT has a nested pickle root, runtime inference accepts only the
deterministic non-executable safetensors package made by the audited converter.
Two conversions produced the same 1,253,463,832-byte file with SHA-256
`2380601d17f6a817de0bf5328188ccea397af9d75c07b4b3cc476322dcca76af`.
The frozen converter environment is the same as TripoSR's, requires the exact
upstream Apache-2.0 license, and also pins the 486-byte config, 1,074-byte
`SOURCE.json`, and 11,357-byte license.

The runtime exports normalized colored OBJ, PLY, GLB, and provenance manifests.
Those manifests preserve every ordered input digest and all 16 camera values
per view. They also record whether the pinned upstream one-signed-field repair
(center positive sentinel plus negative two-cell boundary shell) was applied
before native meshing.
This lane is best for turntable-to-mesh, four/six-view prop reconstruction,
set-dressing proxies, and occlusion/collision assets. It is deliberately not a
single-image plugin: callers must supply views they are licensed to use.

The graph uses view batching plus attention, feed-forward, field-query, and
isosurface chunking for Apple Silicon. The benchmark ledger now includes native
six-view grid-16 and grid-24 runs through the verified converted package; the
measured figures are summarized below.

## License and commercial-use boundary

The selected MoGe, VDA, DA3, TripoSR, and InstantMesh reconstruction artifacts
have permissive MIT or Apache-2.0 lanes. That does not make every file mentioned
by an upstream project safe to ship. The engineering boundary is:

- **Zero123++ is excluded.** The official Zero123++ repository states that its
  code is Apache-2.0 but its weights are CC-BY-NC-4.0 and cannot be used in a
  commercial product pipeline. InstantMesh's customized
  `diffusion_pytorch_model.bin` is a derivative view generator and is therefore
  excluded conservatively. See the
  [Zero123++ license statement](https://github.com/SUDO-AI-3D/zero123plus#license).
- **Users supply the InstantMesh views.** `mere.run` does not create the four or
  six views, does not invoke a hidden hosted generator, and does not silently
  fall back to a noncommercial model.
- **NVIDIA FlexiCubes source is excluded.** The exact InstantMesh source revision
  carries an all-rights-reserved header that prohibits use or distribution
  without an express NVIDIA agreement in its
  [FlexiCubes implementation](https://github.com/TencentARC/InstantMesh/blob/08822c52fdc399b93ea00e4fa9e596344ed52ccc/src/models/geometry/rep_3d/flexicubes.py).
  `mere.run` neither vendors nor executes that implementation.
- **Notices still matter.** MIT and Apache-2.0 attribution, the MoGe DINOv2
  notice, and input-asset rights remain distribution obligations. This is an
  engineering policy, not a substitute for product-specific legal review.

## What "parity" means

Exact neural parity means the native MLX graph is compared with the pinned
upstream implementation at model-internal boundaries using the same weights
and deterministic fixtures. It means numerical agreement within explicit
floating-point tolerances, not bit-for-bit identity on every device.

The parity surfaces are:

- MoGe: raw point map, normal map, validity probability, and metric scale.
- VDA: relative and metric raw depth, including memory-preserving microbatch
  execution before temporal alignment/export.
- DA3: depth, confidence, intrinsics, and extrinsics, both predicted-camera and
  pose-conditioned.
- TripoSR: image tokens, triplane tokens, scene code, raw/activated density,
  and color.
- InstantMesh: image tokens, triplane tokens, scene code, SDF, deformation,
  color, and learned cell weights.

InstantMesh SDF parity is measured before extraction. If an extraction grid's
interior is entirely one-signed, the run then applies the pinned upstream
center/boundary sign repair, records that intervention, and meshes the repaired
field. It never presents a repaired sample as a raw neural-parity value.

Neural-field parity does **not** imply triangle-topology parity:

- TripoSR upstream uses `torchmcubes` marching cubes. The native exporter uses
  deterministic marching tetrahedra, so field values and colors can match
  while vertex order, triangle count, and connectivity differ.
- InstantMesh upstream uses the excluded proprietary FlexiCubes code. The
  native exporter uses a permissive marching-tetrahedra implementation over
  the parity-tested SDF/deformation/color field. It intentionally reports
  `topologyMatchesUpstreamFlexiCubes=false`.

Plugins should compare semantic assets, bounds, field samples, and durable
hashes produced by the same native settings. They must not promise identical
upstream triangle IDs or topology.

## Measured Apple Silicon evidence

The full methodology and tables live in
[VFX geometry Apple Silicon benchmarks](../benchmarks/vfx-geometry-apple-silicon.md).
The current measurements are from a debug build on an M4 Max with 128 GB unified
memory and macOS 26.4:

- MoGe-2 ViT-S Normal: one 128 x 128 image at 3,600 tokens completed in
  0.88 seconds, exported 13,186 valid metric points, and reported 748.9 MiB
  max RSS plus an 8.50 GiB MLX/Metal peak footprint.
- VDA-S: 24 frames at network size 56 completed in 13.53 seconds with a
  683.0 MiB peak footprint; 100 frames completed in 16.91 seconds with a
  690.2 MiB peak footprint. The 24-frame production-size 518 run completed in
  14.68 seconds and reached an 8.66 GiB peak footprint.
- DA3-S: two predicted-camera views at process resolution 56 completed in
  2.65 seconds; the pose-conditioned run completed in 0.54 seconds. Resolution
  504 completed in 4.52 seconds with a 2.83 GiB peak footprint and exported
  76,208 points.
- TripoSR: a deterministic 512 x 512 source image, prepared at the same
  512 x 512 model input, completed in 3.83 seconds at density grid 16 and
  7.34 seconds at grid 32. Both fresh processes reported 6.42 GiB max RSS;
  peak footprints were 5.22 and 5.37 GiB respectively.
- InstantMesh Base: six licensed, user-controlled 320 x 320 Suzanne views
  completed in 2.42 seconds at SDF grid 16, producing 1,586 vertices and 3,164
  triangles with a 3.97 GiB peak footprint. Grid 24 completed in 2.16 seconds,
  producing 3,746 vertices and 7,488 triangles with a 4.03 GiB peak footprint.
  Both fresh processes reported 3.62 GiB max RSS.

These are implementation proofs, not cross-device promises. Plugins should
publish the exact token/grid/view settings beside any timing or minimum-memory
guidance.

## Plugin contract guidance

Keep model truth explicit in plugin outputs:

- Carry model ID, pinned revision, source digest, depth units, coordinate
  system, camera semantics, and every artifact hash into the durable manifest.
- Label VDA relative depth as affine-relative and VDA metric depth as meters.
  Do not add confidence or camera fields to either model.
- Label DA3 output as relative, its GLB as a point cloud, and its 3DGS handoff
  as initialization only. Never imply learned Gaussians or a triangle mesh.
- Label TripoSR and InstantMesh meshes as normalized object space unless a
  separate scale/alignment operation establishes world units.
- For InstantMesh, reject inputs other than four or six views and expose that
  view generation is absent. Do not offer Zero123++ as an automatic fallback.
- Record the native polygonizer and state plainly that TripoSR and InstantMesh
  do not claim upstream triangle-topology parity.

That contract gives VFX tools useful depth, camera, point, and mesh primitives
without blurring relative versus metric geometry, points versus surfaces, or
permissive reconstruction versus excluded generation components.
