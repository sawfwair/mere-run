# Vision Runtime

The widest surface in `mere.run`. Ask a question about a photo, cut out any
object you can name, follow it through a video, read a document, recover metric
depth and camera intrinsics from a single frame, or reconstruct a PBR mesh —
nineteen commands, every one of them running on your own machine.

## Commands

**Understand and read**

| Command | What it does |
| --- | --- |
| `mere.run vision inspect` | Describe or answer questions about an image using Qwen3-VL. |
| `mere.run vision caption` | Generate training-friendly captions for images. |
| `mere.run vision ocr` | Extract text from images using LightOnOCR, GLM-OCR, or Infinity-Parser2. |

**Find, isolate, and follow**

| Command | What it does |
| --- | --- |
| `mere.run vision ground` | Ground text queries in an image with the native Falcon Perception runtime. |
| `mere.run vision segment` | Segment prompted objects in an image with the native SAM 3.1 runtime. |
| `mere.run vision track` | Track prompted objects through a video with the native SAM 3.1 runtime. |
| `mere.run vision track-live` | Capture from a camera and track text-prompted objects with native SAM 3.1. |

**Faces**

| Command | What it does |
| --- | --- |
| `mere.run vision face detect` | Detect faces and five-point landmarks in an image. |
| `mere.run vision face embed` | Create a normalized ArcFace embedding for one face in an image. |
| `mere.run vision face compare` | Compare one face from each of two images with cosine similarity. |
| `mere.run vision face batch` | Analyze many images with one warm detector/recognizer session and emit JSONL. |

**Measure, reconstruct, and move**

| Command | What it does |
| --- | --- |
| `mere.run vision pose` | Detect body, hand, and face landmarks with the native platform runtime. |
| `mere.run vision flow` | Generate dense optical flow between two equal-size images. |
| `mere.run vision depth-video` | Generate temporally consistent relative or metric video depth with native VDA-S. |
| `mere.run vision geometry` | Generate metric depth, normals, camera intrinsics, and a point cloud with native MoGe-2. |
| `mere.run vision geometry-multiview` | Solve native DA3-Small multi-view relative geometry, confidence, and cameras. |
| `mere.run vision image-to-3d` | Reconstruct a colored object mesh from one image with native TripoSR. |
| `mere.run vision image-to-3d-trellis2` | Reconstruct a 512-resolution PBR O-Voxel mesh with native MLX TRELLIS.2. |
| `mere.run vision image-to-3d-multiview` | VFX alias for native 4/6-view InstantMesh reconstruction. |

## Model family

- `vision-ocr-lighton`
- `vision-ground-falcon-perception`
- `vision-face-buffalo-l`
- `vision-segment-sam31`
- `vision-geometry-moge2-small`
- `vision-depth-vda-small`
- `vision-depth-vda-small-metric`
- `vision-geometry-da3-small`
- `image-3d-triposr`
- `image-3d-trellis2-4b`
- `image-3d-instantmesh-base`

Captioning and inspect flows also depend on vision-language support code in
`MereRunCore`.

Grounding runs natively through the Swift/MLX Falcon Perception stack in
`MereRunCore`.

Segmentation and tracking run natively through the Swift/MLX SAM 3.1 stack in
`MereRunCore`.

Face analysis runs the Buffalo-L detector and ArcFace R50 recognizer locally
through ONNX Runtime. The default `auto` provider uses CPU for this model based
on the repository's Apple Silicon parity run; `--execution-provider coreml`
and `--execution-provider cpu` remain explicit controls.

The detector is useful beyond identity search: it provides face boxes and
five-point landmarks for alignment, crops, counts, redaction, face-aware
reframing, thumbnail selection, and dataset quality gates. The embedding model
also supports verification, unnamed-person clustering, similarity search,
representative prototype selection, and identity-outlier review. Neither model
claims emotion, demographic attributes, liveness, deepfake detection, or
general image understanding.

## Current SAM 3.1 scope

- `vision segment` supports text prompts plus geometry prompting with boxes and points
- `vision track` supports text, box, and point prompts on the init frame, then propagates tracked objects through later frames
- `vision track-live` records a camera clip, searches a short warm-up window for seed objects, and then runs the same native tracking path over the saved recording
- the managed model package `vision-segment-sam31` is the single SAM 3.1 package for segmentation and tracking

## Current implementation notes

- still-image text prompting uses the native detector path
- still-image box and point prompting use the native interactive SAM prompt path
- offline video tracking currently uses native prompt propagation built on top of the image segmenter rather than a full SAM memory-bank tracker
- live capture is text-prompt seeded only in the current CLI surface

## Fused attention policy

Supported SAM 3.1, LightOn OCR, and selected vision-encoder attention shapes use
MLX fused scaled-dot-product attention by default. Unsupported shapes retain
their portable implementation. Set `MERERUN_FUSED_SDPA=0` for an emergency
compatibility fallback or a controlled A/B.

The installed-model gate on 2026-07-11 used release binaries on an M4 Max with
128 GB unified memory. A SAM 3.1 text-prompt segmentation run was about 3%
faster (1.36s to 1.32s) and reduced peak footprint by 13.2%; all 11 exported
masks were bit-exact, with only negligible score/box deltas. A warm LightOn OCR
run improved from 2.87s to 1.97s (1.46x throughput) and reduced peak footprint
by 55%, with byte-identical text. Process RSS was effectively flat in both
comparisons, so these measurements are not presented as general RSS savings or
as guarantees for other models, prompts, or shapes.

## Typical workflows

### Caption an image

```bash
swift run mere.run vision caption ./image.png
```

For dataset captioning, use a domain prompt file and focus terms when the
generic captioner would miss the training objective:

```bash
swift run mere.run vision caption ./cards/*.jpg \
  --output-dir ./captions \
  --prompt-file ./card-caption-prompt.txt \
  --focus "full card border" "printed title text" "visible gag" \
  --trigger-token cardstyle \
  --temperature 0.1
```

### Inspect an image with a question

```bash
swift run mere.run vision inspect ./image.png "What objects are visible?"
```

### Detect, embed, and compare faces

```bash
swift run mere.run model pull vision-face-buffalo-l --accept-model-license
swift run mere.run vision face detect ./group.jpg --json
swift run mere.run vision face embed ./reference.jpg --json
swift run mere.run vision face compare ./reference.jpg ./candidate.jpg --json
```

`detect --include-embeddings` emits one normalized 512-dimensional embedding
per detected face. `embed` selects the largest face by default or accepts
`--face-index`; `compare` returns cosine similarity. Use the companion
`mere-face-tools` plugin for resumable folder indexing, SQLite search, and
review/export workflows.

For folder-scale processing, `face batch` keeps the detector and embedding
sessions warm across many images:

```bash
swift run mere.run vision face batch --input-list ./images.txt --jsonl-output ./faces.jsonl
```

`--input-list` reads one image path per line (positional image paths also
work), `--jsonl-output` writes one durable JSONL record per image instead of
stdout, and `--fail-fast` stops at the first unreadable or invalid image.

Buffalo-L pretrained weights are provided by InsightFace for non-commercial
research use. The pull command requires `--accept-model-license` to acknowledge
the upstream restriction and prints the authoritative license URL before the
download starts. The weights are not bundled with mere.run.

### Segment an image with SAM 3.1

```bash
swift run mere.run model pull vision-segment-sam31 --accept-model-license
swift run mere.run vision segment ./image.png --prompt "a person"
```

### Ground objects with Falcon Perception

```bash
swift run mere.run model pull vision-ground-falcon-perception
swift run mere.run vision ground ./image.png --query "cat" "person in red" \
  --mask-output-dir ./masks
```

`--query` (alias `--prompt`) accepts one or more grounding expressions in a
single run; at least one is required. If `--model`/`-m` is omitted, the command
resolves the managed `vision-ground-falcon-perception` package from the local
model store; it also accepts a local Falcon Perception model root directory.
The annotated image defaults to `<stem>_grounded.<ext>` (`--output`/`-o`
overrides it), JSON metadata defaults to `<stem>_grounded.json`
(`--json-output` overrides it), and `--mask-output-dir` exports one PNG mask
per detection.

### Track objects through a video

```bash
swift run mere.run model pull vision-segment-sam31 --accept-model-license
swift run mere.run vision track ./clip.mp4 --prompt "a dog" --init-frame 12
```

### Track a recorded live camera session

```bash
swift run mere.run vision track-live --output ./live.mp4 --prompt "a person"
```

### Extract pose landmarks

```bash
swift run mere.run vision pose ./person.png \
  --json-output ./person-pose.json \
  --minimum-confidence 0.2
```

The native pose result contains body, hand, and face subjects. Landmark
coordinates use a normalized bottom-left coordinate system and retain per-point
confidence for downstream temporal filtering and motion export.

### Generate a dense motion pass

```bash
swift run mere.run vision flow ./frame-001.png ./frame-002.png \
  --output ./frame-001-to-002.flo \
  --accuracy high
```

The two images must have equal dimensions. Output vectors use the Middlebury
`.flo` format and preserve full-resolution 32-bit horizontal and vertical
motion components.

### Generate temporally consistent video depth

```bash
swift run mere.run model pull vision-depth-vda-small
swift run mere.run vision depth-video ./clip.mp4 --output ./clip-depth
```

Native Video Depth Anything Small writes per-frame depth EXRs, preview PNGs, a
review MP4, and a depth-sequence manifest JSON into the output directory
(default `<stem>-depth` next to the input). `--input-size` bounds the longest
network edge before aspect-ratio adjustment (default 518) and `--max-frames`
bounds decoded source frames (default 240). The default model is relative
depth; `--model vision-depth-vda-small-metric` switches to the metric variant.
`--dry-run` hashes and decodes the bounded input, verifies media/network limits
and the checkpoint, then prints the plan without inference; `--json` prints the
structured result on stdout.

### Recover metric geometry from a single image

```bash
swift run mere.run model pull vision-geometry-moge2-small
swift run mere.run vision geometry ./photo.jpg --output ./photo-geometry
```

Native MoGe-2 emits metric depth and normal EXRs with preview PNGs, a validity
mask, camera intrinsics JSON, a point-cloud PLY, and a manifest (default
directory `<stem>-geometry`). `--resolution-level` selects quality 0 through 9
(default 9), `--token-count` overrides the DINO base-token count (1 to 3600),
and `--max-points` caps the PLY point count. `--dry-run` and `--json` behave as
in `depth-video`.

### Solve multi-view geometry and cameras

```bash
swift run mere.run model pull vision-geometry-da3-small
swift run mere.run vision geometry-multiview ./view-01.jpg ./view-02.jpg ./view-03.jpg \
  --output ./scene
```

Native DA3-Small solves relative depth, per-view confidence, and cameras
across the ordered views, exporting per-view depth/confidence EXRs and preview
PNGs, camera JSON, colored point clouds (PLY and GLB), a Nerfstudio/3DGS
initialization handoff, and a scene manifest (default directory
`<first-stem>-da3-scene`). `--cameras` supplies one calibrated W2C camera per
image as JSON; `--process-resolution` bounds the longest processed side
(default 504); `--reference-view` picks `first`, `middle`, `saddle-balanced`
(default), or `saddle-similarity-range`; `--confidence-percentile` (default 40)
discards low-confidence points and `--max-points` caps scene exports.
`--dry-run` and `--json` behave as in `depth-video`.

### Reconstruct a PBR object with TRELLIS.2

Accept the DINOv3 checkpoint license on Hugging Face before the first pull,
then run the native 512-resolution pipeline:

```bash
swift run mere.run model pull image-3d-trellis2-4b --accept-model-license
swift run mere.run vision image-to-3d-trellis2 ./object.png \
  --output ./object-trellis2 \
  --seed 42
```

Transparent alpha is required by default. `--already-framed` explicitly opts
an opaque, isolated object into black-background conditioning. The result
contains canonical colored OBJ/PLY/GLB meshes and a hashed `.pbrvox` sidecar
that preserves base color, metallic, roughness, and alpha.

## Output artifacts

### `vision ground`

- annotated image written to `<stem>_grounded.<ext>` unless `--output` is provided
- JSON metadata written to `<stem>_grounded.json` unless `--json-output` is provided
- optional mask PNGs written to `--mask-output-dir`

The JSON includes:

- `schemaVersion`
- model and input/output paths
- query list
- detections with `query`, normalized `xy`, normalized `hw`, derived `box`, optional `score`, and optional `maskPath`

### `vision face`

- `detect` emits image dimensions, elapsed inference time, face scores, pixel
  boxes, and five landmarks; `--include-embeddings` adds identity vectors
- `embed` emits one selected face and its normalized 512-value embedding
- `compare` emits the selected face indexes and cosine similarity
- `batch` keeps the sessions warm and emits one durable JSONL result per image;
  `--include-embeddings` enables recognition/search vectors
- `--json` keeps stdout machine-readable; `--json-output` writes the same
  sorted payload atomically

### `vision segment`

- annotated image written to `<stem>_segmented.<ext>` unless `--output` is provided
- JSON metadata written to `<stem>_segmented.json` unless `--json-output` is provided
- optional mask PNGs written to `--mask-output-dir`

The JSON includes:

- `schemaVersion`
- model and input/output paths
- prompts, threshold, and resolution
- detections with `label`, `score`, `box`, `maskAreaPixels`, and optional `objectID`, `promptKind`, `maskPath`, and `candidateIndex`

### `vision track` and `vision track-live`

- annotated video written to `<stem>_tracked.mp4` for `vision track`
- JSON metadata written to `<stem>_tracked.json` for `vision track`
- `vision track-live` requires an explicit output video path
- `vision track-live` defaults to frame 0 but searches a short warm-up window when that frame yields no seed objects
- optional per-frame mask PNGs written under frame-named subdirectories when `--mask-output-dir` is set on `vision track`

The tracking JSON includes:

- `schemaVersion`
- model and input/output paths
- fps, frame size, init frame, and dropped frame count
- stable tracked object metadata
- per-frame detections with `objectID`, `label`, `score`, `visible`, `box`, `maskAreaPixels`, and optional `maskPath`

### `vision pose`

- JSON metadata written to `<stem>_pose.json` unless `--json-output` is provided
- typed body, hand, and face subjects with stable point names
- normalized coordinates, image dimensions, and confidence values
- no separately installed model package on Apple platforms; inference is owned
  by the native platform runtime in `MereRunCore`

### `vision flow`

- dense full-resolution two-component optical-flow vectors
- standard Middlebury `.flo` output plus typed JSON metadata
- selectable native accuracy and magnitude statistics
- explicit equal-dimension validation

### OCR

```bash
swift run mere.run vision ocr ./page.png --backend lighton
swift run mere.run vision ocr ./page.png --backend infinity --infinity-task doc2md
```

For an external Infinity-Parser2 parity eval against an already-running vLLM server:

```bash
swift run mere.run vision ocr ./page.png \
  --backend infinity \
  --infinity-runtime external \
  --infinity-api-url http://127.0.0.1:8000/v1/chat/completions
```

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/VisionCaptionCommand.swift`
- `Sources/MereRunCLI/Commands/VisionInspectCommand.swift`
- `Sources/MereRunCLI/Commands/VisionFaceCommand.swift`
- `Sources/MereRunCLI/Commands/VisionGroundCommand.swift`
- `Sources/MereRunCLI/Commands/VisionSegmentCommand.swift`
- `Sources/MereRunCLI/Commands/VisionTrackCommand.swift`
- `Sources/MereRunCLI/Commands/VisionTrackLiveCommand.swift`
- `Sources/MereRunCLI/Commands/VisionPoseCommand.swift`
- `Sources/MereRunCLI/Commands/VisionFlowCommand.swift`
- `Sources/MereRunCLI/Commands/VisionDepthVideoCommand.swift`
- `Sources/MereRunCLI/Commands/VisionGeometryCommand.swift`
- `Sources/MereRunCLI/Commands/VisionGeometryMultiViewCommand.swift`
- `Sources/MereRunCLI/Commands/VisionOCRCommand.swift`

### Pose runtime

- `Sources/MereRunCore/Pose/NativePoseDetector.swift`

### Optical-flow runtime

- `Sources/MereRunCore/OpticalFlow/NativeOpticalFlowGenerator.swift`

### OCR runtime

- `Sources/MereRunCore/LightOnOCR/LightOnOCRGenerator.swift`
- `Sources/MereRunCore/LightOnOCR/LightOnOCRGenerator+Loading.swift`
- `Sources/MereRunCore/LightOnOCR/LightOnOCRGenerator+Inference.swift`
- `Sources/MereRunCore/LightOnOCR/LightOnOCRSupport.swift`
- `Sources/MereRunCore/Q35/Q35Generator.swift`
- `Sources/MereRunCore/Q35/Q35Model.swift`
- `Sources/MereRunCore/Q35/Q35VisionTower.swift`

### Vision-language support

- `Sources/MereRunCore/VLM/`
- `Sources/MereRunCore/QwenVLCaptioner.swift`
- `Sources/MereRunCore/Qwen25VLEncoder.swift`
- `Sources/MereRunCore/QwenVisionAttention.swift`

### Falcon grounding runtime

- `Sources/MereRunCore/FalconPerception/FalconPerceptionConfig.swift`
- `Sources/MereRunCore/FalconPerception/FalconPerceptionResources.swift`
- `Sources/MereRunCore/FalconPerception/FalconPerceptionTokenizer.swift`
- `Sources/MereRunCore/FalconPerception/FalconPerceptionProcessor.swift`
- `Sources/MereRunCore/FalconPerception/FalconPerceptionModel.swift`
- `Sources/MereRunCore/FalconPerception/FalconPerceptionAnyUp.swift`
- `Sources/MereRunCore/FalconPerception/FalconPerceptionGrounder.swift`

### SAM 3.1 runtime

- `Sources/MereRunCore/SAM3/SAM31Config.swift`
- `Sources/MereRunCore/SAM3/SAM31Resources.swift`
- `Sources/MereRunCore/SAM3/SAM31Tokenizer.swift`
- `Sources/MereRunCore/SAM3/SAM31Model.swift`
- `Sources/MereRunCore/SAM3/SAM31InteractiveSAM.swift`
- `Sources/MereRunCore/SAM3/SAM31Prompts.swift`
- `Sources/MereRunCore/SAM3/SAM31ImageSegmenter.swift`
- `Sources/MereRunCore/SAM3/SAM31VideoIO.swift`
- `Sources/MereRunCore/SAM3/SAM31VideoTracker.swift`
- `Sources/MereRunCore/SAM3/SAM31CameraCapture.swift`

## How the OCR path works

1. the CLI resolves the OCR model
2. the OCR runtime loads the required components
3. the input image is normalized into the expected tensor form
4. OCR inference runs
5. text is emitted without internal bring-up logs on stdout

LightOnOCR uses the dedicated LightOn runtime and remains the default `vision ocr`
backend. Native Infinity-Parser2 uses the Q35 text runtime plus the Qwen-family
vision tower, with `vision-ocr-infinity-flash` as the default Infinity model,
`vision-ocr-infinity-pro-int8` as the quality-focused Pro eval option, and
`vision-ocr-infinity-pro` as the full BF16 heavyweight compatibility target.

GLM-OCR remains an external CLI adapter that shells out to `glmocr`. Infinity can
also run as an external parity adapter through `--infinity-runtime external`,
which shells out to the `parser` executable from `infinity_parser2` and can
target an upstream Transformers, vLLM engine, or vLLM server run.

## How segmentation and tracking work

1. the CLI resolves `vision-segment-sam31` from the model store or uses the local root passed with `--model`
2. the native SAM 3.1 runtime validates the root, loads tokenizer/config/weights, and preprocesses the input image or video frames
3. still-image text prompts run the detector once, then text + DETR + mask decode per prompt
4. geometry prompts use the interactive SAM path, and video tracking reuses those prompts frame to frame after the seed frame
5. native postprocessing applies thresholding, mask resize, score ordering, NMS, and optional mask export
6. the runtime writes annotated media plus structured JSON metadata

## How grounding works

1. the CLI resolves `vision-ground-falcon-perception` from the model store or uses the local root passed with `--model`
2. the native Falcon runtime validates the root, loads config/tokenizer/weights, and preprocesses the image plus text query
3. the model autoregressively emits grounded detections, including coordinate and size tokens, and decodes optional segmentation masks
4. native postprocessing derives normalized centers, sizes, bounding boxes, and optional exported mask artifacts
5. the runtime writes an annotated image plus structured JSON metadata designed for downstream agent use

## How caption and inspect differ

- `caption` is a direct descriptive task
- `inspect` is a question-driven vision-language path

They share some of the same underlying vision support code, but they are
presented as separate public tasks because the user intent differs.
