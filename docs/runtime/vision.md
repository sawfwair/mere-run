# Vision Runtime

This page covers captioning, inspection, grounding, segmentation, tracking, pose extraction, optical flow, and OCR.

## Public surface

- `mere.run vision caption`
- `mere.run vision inspect`
- `mere.run vision ground`
- `mere.run vision segment`
- `mere.run vision track`
- `mere.run vision track-live`
- `mere.run vision pose`
- `mere.run vision flow`
- `mere.run vision ocr`

## Model family

- `vision-ocr-lighton`
- `vision-ground-falcon-perception`
- `vision-segment-sam31`

Captioning and inspect flows also depend on vision-language support code in
`MereRunCore`.

Grounding runs natively through the Swift/MLX Falcon Perception stack in
`MereRunCore`.

Segmentation and tracking run natively through the Swift/MLX SAM 3.1 stack in
`MereRunCore`.

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

### Segment an image with SAM 3.1

```bash
swift run mere.run model pull vision-segment-sam31
swift run mere.run vision segment ./image.png --prompt "a person"
```

### Ground objects with Falcon Perception

```bash
swift run mere.run model pull vision-ground-falcon-perception
swift run mere.run vision ground ./image.png --query "a person"
```

### Track objects through a video

```bash
swift run mere.run model pull vision-segment-sam31
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
- `Sources/MereRunCLI/Commands/VisionGroundCommand.swift`
- `Sources/MereRunCLI/Commands/VisionSegmentCommand.swift`
- `Sources/MereRunCLI/Commands/VisionTrackCommand.swift`
- `Sources/MereRunCLI/Commands/VisionTrackLiveCommand.swift`
- `Sources/MereRunCLI/Commands/VisionPoseCommand.swift`
- `Sources/MereRunCLI/Commands/VisionFlowCommand.swift`
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
