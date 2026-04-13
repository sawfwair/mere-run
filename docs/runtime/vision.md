# Vision Runtime

This page covers captioning, inspection, grounding, segmentation, tracking, and OCR.

## Public surface

- `mere.run vision caption`
- `mere.run vision inspect`
- `mere.run vision ground`
- `mere.run vision segment`
- `mere.run vision track`
- `mere.run vision track-live`
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
- `vision track-live` records a camera clip and then runs the same native tracking path over the saved recording
- the managed model package `vision-segment-sam31` is the single SAM 3.1 package for segmentation and tracking

## Current implementation notes

- still-image text prompting uses the native detector path
- still-image box and point prompting use the native interactive SAM prompt path
- offline video tracking currently uses native prompt propagation built on top of the image segmenter rather than a full SAM memory-bank tracker
- live capture is text-prompt seeded only in the current CLI surface

## Typical workflows

### Caption an image

```bash
swift run mere.run vision caption ./image.png
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
- optional per-frame mask PNGs written under frame-named subdirectories when `--mask-output-dir` is set on `vision track`

The tracking JSON includes:

- `schemaVersion`
- model and input/output paths
- fps, frame size, init frame, and dropped frame count
- stable tracked object metadata
- per-frame detections with `objectID`, `label`, `score`, `visible`, `box`, `maskAreaPixels`, and optional `maskPath`

### OCR

```bash
swift run mere.run vision ocr ./page.png --backend lighton
```

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/VisionCaptionCommand.swift`
- `Sources/MereRunCLI/Commands/VisionInspectCommand.swift`
- `Sources/MereRunCLI/Commands/VisionGroundCommand.swift`
- `Sources/MereRunCLI/Commands/VisionSegmentCommand.swift`
- `Sources/MereRunCLI/Commands/VisionTrackCommand.swift`
- `Sources/MereRunCLI/Commands/VisionTrackLiveCommand.swift`
- `Sources/MereRunCLI/Commands/VisionOCRCommand.swift`

### OCR runtime

- `Sources/MereRunCore/LightOnOCR/LightOnOCRGenerator.swift`
- `Sources/MereRunCore/LightOnOCR/LightOnOCRGenerator+Loading.swift`
- `Sources/MereRunCore/LightOnOCR/LightOnOCRGenerator+Inference.swift`
- `Sources/MereRunCore/LightOnOCR/LightOnOCRSupport.swift`

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
5. text is emitted without the internal bring-up noise that older builds used
   to print

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
