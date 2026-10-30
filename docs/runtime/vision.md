# Vision runtime

The vision runtime analyzes images and video on your machine. Use it to caption
or question a photo, embed images for retrieval, read a document, find and mask
named objects, follow them through a video, analyze faces, measure pose and
motion, recover depth and camera geometry, or reconstruct a textured 3D mesh.
Every model runs locally, and most run natively in Swift and MLX.

To chat with a vision-language model about an image, use `text chat --image`
instead. For more information, see [Vision input](./text.md#image-input) in
the text runtime and [Nemotron 3.5 Bolt](./nemotron-bolt.md).

## Commands

The `mere.run vision` command group has 20 subcommands. The default model for
each command is the one it resolves when you omit `--model`. A dash means the
command uses a platform framework or a downloaded default rather than a managed
model ID.

**Understand and read**

| Command | What it does | Default model |
| --- | --- | --- |
| `vision caption` | Generate training-friendly captions for images. | Qwen3-VL 2B Instruct 4-bit, downloaded on first use |
| `vision inspect` | Describe or answer questions about an image using Qwen3-VL. | Default local vision-language model, downloaded on first use |
| `vision embed` | Generate shared text and image embeddings with native Qwen3-VL. | `vision-embed-qwen3-vl-2b` |
| `vision ocr` | Extract text from images with LightOnOCR, GLM-OCR, Infinity-Parser2, or Unlimited-OCR-2. | `vision-ocr-lighton` |

**Find, isolate, and follow**

| Command | What it does | Default model |
| --- | --- | --- |
| `vision ground` | Ground text queries in an image with native Falcon Perception. | `vision-ground-falcon-perception` |
| `vision serve` | Serve resident Falcon Perception grounding over loopback HTTP. | `vision-ground-falcon-perception` |
| `vision segment` | Segment prompted objects in an image with native SAM 3.1. | `vision-segment-sam31` |
| `vision track` | Track prompted objects through a video with native SAM 3.1. | `vision-segment-sam31` |
| `vision track-live` | Capture from a camera and track text-prompted objects with native SAM 3.1. | `vision-segment-sam31` |

**Faces, pose, and motion**

| Command | What it does | Default model |
| --- | --- | --- |
| `vision face detect` | Detect faces and five-point landmarks in an image. | `vision-face-buffalo-l` |
| `vision face embed` | Create a normalized ArcFace embedding for one face in an image. | `vision-face-buffalo-l` |
| `vision face compare` | Compare one face from each of two images with cosine similarity. | `vision-face-buffalo-l` |
| `vision face batch` | Analyze many images with one warm detector and recognizer session, and emit JSONL. | `vision-face-buffalo-l` |
| `vision pose` | Detect body, hand, and face landmarks with the native platform runtime. | - |
| `vision flow` | Generate dense optical flow between two equal-size images. | - |

**Depth, geometry, and 3D**

| Command | What it does | Default model |
| --- | --- | --- |
| `vision depth` | Estimate affine-invariant depth for a still image with native Marigold V2. | `vision-depth-marigold-v2` |
| `vision depth-video` | Generate temporally consistent relative or metric video depth with native VDA-S. | `vision-depth-vda-small` |
| `vision geometry` | Generate metric depth, normals, camera intrinsics, and a point cloud with native MoGe-2. | `vision-geometry-moge2-small` |
| `vision geometry-multiview` | Solve native DA3-Small multi-view relative geometry, confidence, and cameras. | `vision-geometry-da3-small` |
| `vision image-to-3d` | Reconstruct a colored object mesh from one image with native TripoSR. | `image-3d-triposr` |
| `vision image-to-3d-trellis2` | Reconstruct a 512-resolution PBR O-Voxel mesh with native MLX TRELLIS.2. | `image-3d-trellis2-4b` |
| `vision image-to-3d-multiview` | Reconstruct a colored mesh from four or six supplied views with native InstantMesh. | `image-3d-instantmesh-base` |

The three `image-to-3d` commands run the same implementations as `image
reconstruct-3d`, `image reconstruct-3d-trellis2`, and `image
reconstruct-3d-multiview`. For the image-family spellings, see the
[Image runtime](./image.md).

## Choose a model

When a task has more than one managed model, use the following table to pick
one. Memory values are the minimum and recommended unified memory from the
managed capability catalog.

| Task | Model ID | Memory (min / recommended) | Choose it when |
| --- | --- | --- | --- |
| OCR | `vision-ocr-lighton` | 8 / 16 GB | You want the default: LightOnOCR 3 1B, which passed every case in its qualification. |
| OCR | `vision-ocr-lighton-3-0.8b` | 8 / 16 GB | You want the smallest LightOnOCR 3 download and can review handwriting by hand. |
| OCR | `vision-ocr-lighton-3-4b` | 16 / 24 GB | You want LightOn's quality recommendation and have the memory for it. |
| OCR | `vision-ocr-unlimited-2` | 16 / 24 GB (estimated) | You want native ordered-document parsing; full-checkpoint qualification is outstanding. |
| OCR | `vision-ocr-infinity-pro-int8` | 64 / 96 GB | You want Infinity-Parser2 document-to-JSON or Markdown parsing. |
| OCR | `vision-ocr-infinity-pro` | 128 / 192 GB | You're running full-precision Infinity-Parser2 quality evaluations. |
| Video depth | `vision-depth-vda-small` | 16 / 32 GB | You want relative depth for a clip. |
| Video depth | `vision-depth-vda-small-metric` | 16 / 32 GB | You want depth in metric units for a clip. |
| Still depth | `vision-depth-marigold-v2` | 64 / 96 GB | You want affine-invariant depth from Marigold V2. |
| Still geometry | `vision-geometry-moge2-small` | 8 / 16 GB | You need metric depth, normals, intrinsics, or a point cloud from one image. |
| Multi-view geometry | `vision-geometry-da3-small` | 12 / 24 GB | You have several views of one scene and need cameras and a fused point cloud. |
| Object mesh | `image-3d-triposr` | 16 / 32 GB | You want a fast colored mesh from one image. |
| Object mesh | `image-3d-trellis2-4b` | 64 / 128 GB | You want a PBR mesh with metallic, roughness, and alpha from one image. |
| Object mesh | `image-3d-instantmesh-base` | 24 / 48 GB | You already have four or six views of the object. |

Other managed vision models have one role each: `vision-embed-qwen3-vl-2b`
(16 / 24 GB), `vision-ground-falcon-perception` (16 / 24 GB),
`vision-segment-sam31` (16 / 24 GB), and `vision-face-buffalo-l` (8 / 16 GB).

Some downloads require you to acknowledge upstream terms with
`--accept-model-license`:

- `vision-segment-sam31`: Meta's SAM License.
- `vision-face-buffalo-l`: InsightFace non-commercial research terms.
- `image-3d-trellis2-4b`: the gated DINOv3 License. Accept it on Hugging Face
  before the first pull.

## macOS Studio

The macOS Studio app exposes the same commands through its shared capability
contract, as the ten tasks of the **Vision** domain. **Read**, **Find**,
**Segment**, and **Track** show the answer beside the input: boxes drawn over
the picture, masks composited on it, track spans under the video's scrubber,
and the raw result document one segment away. **Depth**, **Pose**, **Faces**,
**Flow**, **Geometry**, and **Live** cover still and video depth, native pose
and optical flow, Buffalo-L detection, embedding, comparison and batch
analysis, single- and multi-view geometry, and live camera tracking. They draw
face and pose overlays and direction-colored flow vectors, play annotated and
depth video, embed orbitable point clouds, report request-scoped progress, and
keep structured and visual sidecars in the Library.

Image-to-3D commands live in the **3D** domain so the app has one
reconstruction flow. The grounding server has its own **Server ▸ Vision
server** task. The **Command view** (⌥⌘C) and the **Command Console** show and
run the raw command for any of these tasks.

## Understand images

### Caption images

`vision caption` writes one `.txt` caption beside each image, or into
`--output-dir`. If you omit `--model`, it downloads
`mlx-community/Qwen3-VL-2B-Instruct-4bit` on first use.

```bash
mere.run vision caption ./image.png
```

For dataset captioning, add a domain prompt file and focus terms when the
generic captioner misses the training objective. `--trigger-token`
prefixes every saved caption with a LoRA trigger token.

```bash
mere.run vision caption ./cards/*.jpg \
  --output-dir ./captions \
  --prompt-file ./card-caption-prompt.txt \
  --focus "full card border" "printed title text" "visible gag" \
  --trigger-token cardstyle \
  --temperature 0.1
```

Defaults are 96 new tokens, temperature 0.2, and top-p 0.9.

### Ask a question about an image

`vision inspect` prints the model's answer on stdout and keeps progress on
stderr. The prompt defaults to `Describe this image.`

```bash
mere.run vision inspect ./image.png --prompt "What objects are visible?"
```

Defaults are 2,048 new tokens, temperature 0.7, and top-p 0.9. Positional
prompt words still work, but `--prompt` is preferred. `caption` is a direct
descriptive task, and `inspect` is question-driven; they share vision-language
support code but serve different intents.

### Embed text and images

`vision embed` produces L2-normalized vectors in one shared space for text,
images, and mixed text-and-image records. Use them for semantic retrieval,
clustering, and candidate ranking. The managed `vision-embed-qwen3-vl-2b`
model pins the official BF16 Qwen3-VL Embedding checkpoint and returns 2,048
dimensions.

```bash
mere.run model pull vision-embed-qwen3-vl-2b
mere.run vision embed \
  --image ./vehicle.jpg \
  --text "a white SUV" \
  --instruction "Retrieve visually similar vehicles" \
  --pretty
```

Each `--text` and `--image` value becomes an independent item. To pair text
with an image or keep stable caller IDs, pass a JSON batch with `--input-json`
(or `-` for stdin). Image paths in the JSON resolve relative to the JSON file.

```json
{
  "inputs": [
    {"id": "query", "text": "a maroon pickup truck"},
    {"id": "candidate-1", "image": "./crop-001.jpg"},
    {"id": "candidate-2", "text": "rear view", "image": "./crop-002.jpg"}
  ]
}
```

```bash
mere.run vision embed --input-json ./retrieval-inputs.json --output ./vectors.json
```

JSON always goes to stdout; `--output` also writes it to a file.
`--dimensions` truncates vectors to 1 through 2,048 values and renormalizes
them. Each item allows 8,192 tokens, and images resize to between 4,096 and
1,843,200 pixels (`--min-pixels`, `--max-pixels`). Because the vectors are
normalized, a dot product gives cosine similarity. A high score isn't proof of
identity; keep match thresholds, reranking, and review in your application.

For text, code, image, audio, and video vectors from one Google model, see
[EmbeddingGemma 2](./embeddinggemma2.md).

### Extract text with OCR

`vision ocr` reads one or more images and prints the text on stdout, or writes
`.txt` files with `--output-dir`. Render PDF pages to upright images before you
call it. The `--backend` option selects the engine:

| Backend | Runtime | Default model |
| --- | --- | --- |
| `lighton` (default) | Native LightOnOCR | `vision-ocr-lighton` |
| `infinity` | Native Infinity-Parser2, or an external parity adapter | `vision-ocr-infinity-pro-int8` |
| `glm` | External `glmocr` CLI that you install and configure | Not applicable |
| `unlimited` | Native Unlimited-OCR-2, with all images forming one ordered document | `vision-ocr-unlimited-2` |

```bash
mere.run vision ocr ./page.png
mere.run vision ocr ./page.png --backend infinity --infinity-task doc2md
```

Each backend reads only its own options and warns about the rest. `--compare`
runs LightOn beside a secondary backend; with `--backend lighton`, the
secondary backend is GLM. Defaults are 4,096 new tokens and temperature 0.2.

#### Unlimited-OCR-2 documents

```bash
mere.run model pull vision-ocr-unlimited-2
mere.run vision ocr ./page-01.png ./page-02.png --backend unlimited --temperature 0 --max-tokens 16384
```

The output preserves `<PAGE>` markers. `--unlimited-model` accepts the managed
ID or an original BF16 checkpoint directory; `--unlimited-crop auto|base|tiles`
controls image tiling. This backend doesn't support `--compare`. It retains
the document KV cache and caps prompt plus output at 131,072 tokens. See
[Unlimited-OCR-2](./unlimited-ocr-2.md) for implementation and validation limits.

#### LightOnOCR 3

`vision-ocr-lighton` downloads the pinned official LightOnOCR-3-1B checkpoint,
which runs on the native Pixtral path. `vision-ocr-lighton-3-0.8b` and
`vision-ocr-lighton-3-4b` run on the native Qwen3.5 vision runtime. The 4B
model is LightOn's quality recommendation, and 0.8B is the smaller option. The
default stays 1B.

```bash
mere.run model pull vision-ocr-lighton-3-4b
mere.run vision ocr ./page.png --model vision-ocr-lighton-3-4b --lighton-mode grounding
```

`--lighton-mode` selects the output:

- `plain` (default) sends an empty image-only prompt and returns a Markdown
  transcription.
- `grounding` sends the exact trained instruction and preserves raw labels and
  boxes such as `![text](80,175,920,225)`, followed by content. Coordinates are
  normalized to 0 through 1,000. Image regions can contain short descriptions,
  and chart regions can contain HTML tables with estimated values. The runtime
  doesn't validate those estimates or convert the output to JSON.

The mode also applies to the LightOn side of `--compare`. Qwen3.5 variants
disable thinking and process aspect-preserving images between 65,536 and
5,000,000 pixels, resizing further when the 16K context budget requires it.
The 1B model keeps Pixtral's 1,540-pixel long-edge convention.

Existing `vision-ocr-lighton` installs are preserved and might still contain
LightOnOCR 2. To replace one with 3-1B, back it up and run `mere.run model pull
vision-ocr-lighton --force`, or download 3-1B to a separate directory and pass
that directory with `--model`.

All three released BF16 checkpoints completed the
[LightOnOCR 3 native qualification](../benchmarks/lightonocr3-native-qualification-2026-10-08.md)
on 2026-10-08. 1B and 4B passed all ten checked output cases each; 0.8B passed
eight and failed the handwritten-word crop in both modes. The generated invoice
rows, chart values, and table and chart grounding regions passed for every
variant. This is a small image-input qualification; full transcription
accuracy, upstream numerical parity, and comparative throughput remain
unqualified. For upstream details, see the
[LightOnOCR 3 release](https://huggingface.co/blog/lightonai/lightonocr-3) and
the [1B model card](https://huggingface.co/lightonai/LightOnOCR-3-1B).

#### Infinity-Parser2 and GLM-OCR

Native Infinity-Parser2 runs on the Q35 text runtime and the Qwen-family vision
tower. `vision-ocr-infinity-pro-int8` (about 38 GiB) is the default native
model; `vision-ocr-infinity-pro` (about 131 GiB) is the full BF16 model for
heavyweight evaluations. Neither downloads automatically. `--infinity-task`
selects `doc2json` (default), `doc2md`, or `custom` with `--infinity-prompt`.

For a parity check against upstream, `--infinity-runtime external` runs the
`parser` executable from `infinity_parser2` with the Transformers, vLLM engine,
or vLLM server backend. To compare against a vLLM server that is already
running, run:

```bash
mere.run vision ocr ./page.png \
  --backend infinity \
  --infinity-runtime external \
  --infinity-api-url http://127.0.0.1:8000/v1/chat/completions
```

GLM-OCR runs the upstream `glmocr` CLI from your `PATH`, or from
`--glmocr-cli`. Install and configure it from the
[GLM-OCR repository](https://github.com/zai-org/GLM-OCR); `--glm-config`
forwards a configuration file.

## Find, isolate, and follow objects

### Ground objects with Falcon Perception

`vision ground` finds the regions that match one or more text queries and
optionally exports a mask for each detection.

```bash
mere.run model pull vision-ground-falcon-perception
mere.run vision ground ./image.png --query "cat" "person in red" \
  --mask-output-dir ./masks
```

`--query` (alias `--prompt`) takes one or more expressions; at least one is
required. `--model` accepts the managed ID or a local Falcon Perception model
root. The command writes these files:

- An annotated image at `<stem>_grounded.<ext>`, or at `--output`.
- JSON metadata at `<stem>_grounded.json`, or at `--json-output`. It contains
  `schemaVersion`, the model and paths, the query list, and detections with
  `query`, normalized `xy` and `hw`, a derived `box`, and optional `score` and
  `maskPath`.
- Optional per-detection PNG masks in `--mask-output-dir`.

`--preflight --json` validates the image, model, queries, and output plan
without loading the model. For local model roots, preflight applies the same
manifest, checkpoint-file, and tokenizer-layout checks as execution; these
checks don't load tensors or establish accuracy. `--quiet` prints only the
annotated image path, and `--receipt` prints a final JSON line that lists every
output.

Coordinate decoding follows the Falcon reference's repeated-location policy.
Each query keeps its own coordinate history. If both predicted axes are within
a strict 1% of an earlier coordinate, the decoder suppresses both bins and
tries again, for at most 100 attempts. The selected coordinate feeds later
token generation, so this isn't postprocessing of final boxes. When the attempt
limit is reached, the last candidate is kept, so duplicates are still possible.
Each query stops at the model's end-of-sequence token or the tokenizer's
`<|end_of_query|>` token.

In portable graphs, the built-in `vision.ground` node runs the same runtime. It
takes an image and a JSON array of queries and produces a verified annotated
`image`, structured `detections` JSON, and a `masks` directory. Treat these
outputs as candidate geometry, not authoritative evidence.

### Serve grounding to other processes

For repeated frames, `vision serve` keeps Falcon Perception resident behind a
loopback HTTP service:

```bash
mere.run vision serve --port 8091

curl http://127.0.0.1:8091/v1/vision/ground \
  -F 'stream_id=camera-1' \
  -F 'frame_id=frame-000042' \
  -F 'query=vehicle' \
  -F 'query=person' \
  -F 'image=@frame.webp;type=image/webp'
```

The service accepts PNG, JPEG, or WebP bytes and returns normalized
detections, the input image SHA-256, request identifiers, and inference timing.
It assigns no application meaning and keeps no application state; clients own
cadence, temporal association, and policy. It binds to `127.0.0.1` by default.
A non-loopback `--host` requires `--api-key` or `MERERUN_API_KEY`.

To ground several images in one native prefill and decode pass, use the batch
endpoint:

```bash
curl http://127.0.0.1:8091/v1/vision/ground-batch \
  -F 'stream_id=batch-0001' \
  -F 'frame_id[]=frame-000042' \
  -F 'frame_id[]=frame-000043' \
  -F 'query=target' \
  -F 'image[]=@frame-000042.webp;type=image/webp' \
  -F 'image[]=@frame-000043.webp;type=image/webp'
```

`frame_id[]` is optional; when present, it needs one value for each image.
Every query applies to every image. The server enforces these limits:

| Option | Limit | Default |
| --- | --- | --- |
| `--max-batch-size` | Image-query pairs in one batch, 1 through 32 | 8 |
| `--max-frame-bytes` | Encoded bytes for one image | 16,777,216 |
| `--max-batch-bytes` | Combined encoded bytes in one batch | 67,108,864 |

The batch response contains ordered per-image hashes and normalized
detections. It omits masks and annotated images; use the single-image endpoint
when you need them.

### Segment an image with SAM 3.1

`vision segment` masks objects named by text prompts, or picked by box and
point prompts in image pixels.

```bash
mere.run model pull vision-segment-sam31 --accept-model-license
mere.run vision segment ./image.png --prompt "a person"
mere.run vision segment ./image.png --box 120,80,460,610 --point 300,300,positive
```

Boxes and points combine by label:

- `--box x1,y1,x2,y2[,label]` is one object.
- `--point x,y,positive[,label]` or `x,y,negative[,label]` refines the box with
  the same label that contains it, or else the first box with that label.
- Unlabeled points refine the one unlabeled box when there is exactly one, and
  otherwise form one object together.
- Points that match no box must include a positive point.

Text prompts run the native detector path. Box and point prompts run the
interactive SAM path; `--multimask` returns several mask candidates for each
prompted object. The default score threshold is 0.05 and the default square
input resolution is 1,008. `--show-boxes` draws boxes over the mask overlays.

The command writes an annotated image at `<stem>_segmented.<ext>` and JSON at
`<stem>_segmented.json`, unless you pass `--output` or `--json-output`.
`--mask-output-dir` exports one PNG for each object. The JSON contains
`schemaVersion`, the model and paths, prompts, threshold, resolution, and
detections with `label`, `score`, `box`, `maskAreaPixels`, and optional
`objectID`, `promptKind`, `maskPath`, and `candidateIndex`. `--preflight
--json` inspects the request without loading SAM.

### Track objects through a video

`vision track` seeds objects on one frame with text, box, or point prompts and
propagates them through later frames.

```bash
mere.run model pull vision-segment-sam31 --accept-model-license
mere.run vision track ./clip.mp4 --prompt "a dog" --init-frame 12
```

`--init-frame` defaults to 0, and `--end-frame` sets an inclusive last frame.
Box and point prompts follow the same grouping as `vision segment`; points
seed the object on the initial frame and return when a lost object is
re-seeded. Tracking uses native prompt propagation built on the image
segmenter, not a full SAM memory-bank tracker.

The command writes an annotated video at `<stem>_tracked.mp4` and JSON at
`<stem>_tracked.json`. `--mask-output-dir` writes per-frame mask PNGs into
frame-named subdirectories. The JSON contains `schemaVersion`, the model and
paths, fps, frame size, initial frame, dropped-frame count, stable object
metadata, and per-frame detections with `objectID`, `label`, `score`,
`visible`, `box`, `maskAreaPixels`, and optional `maskPath`. `--preflight
--json` checks the video, prompts, model, and outputs without loading SAM.

### Track a live camera session

`vision track-live` records a camera clip, searches for seed objects, and then
runs the same tracking path over the recording. It accepts text prompts only
and requires an output path.

```bash
mere.run vision track-live --output ./live.mp4 --prompt "a person"
```

Capture defaults to camera 0 for 10 seconds (`--camera`,
`--duration-seconds`). If the initial frame yields no objects, the command
searches up to `--seed-search-frames` more frames (default 30). It removes the
temporary recording when it's interrupted.

## Analyze faces

The `vision face` commands run the Buffalo-L detector and ArcFace R50
recognizer locally through ONNX Runtime. InsightFace provides the pretrained
weights for non-commercial research use, so the pull requires
`--accept-model-license` and prints the license URL before the download
starts. The weights aren't bundled with mere.run. The catalog reports
Buffalo-L as unsupported on Linux.

```bash
mere.run model pull vision-face-buffalo-l --accept-model-license
mere.run vision face detect ./group.jpg --json
mere.run vision face embed ./reference.jpg --json
mere.run vision face compare ./reference.jpg ./candidate.jpg --json
```

Each command does one job:

- `detect` returns image dimensions, inference time, face scores, pixel boxes,
  and five landmarks. `--include-embeddings` adds a normalized
  512-dimensional embedding for each face, and `--max-faces` caps detections.
- `embed` returns one face and its embedding. It picks the largest face unless
  you pass `--face-index`.
- `compare` returns the selected face indexes and their cosine similarity.
  `--reference-face-index` and `--candidate-face-index` override the
  largest-face default.

The detector score threshold defaults to 0.65. `--execution-provider` accepts
`auto`, `coreml`, or `cpu`; `auto` uses the CPU for this model, based on the
repository's Apple Silicon parity run. `--json` prints to stdout, and
`--json-output` writes the same sorted payload atomically.

For folders, `face batch` keeps both sessions warm across many images:

```bash
mere.run vision face batch --input-list ./images.txt --jsonl-output ./faces.jsonl
```

`--input-list` reads one image path per line, and positional paths also work.
`--jsonl-output` writes one durable JSONL record per image instead of stdout.
`--include-embeddings` adds vectors for search, and `--fail-fast` stops at the
first unreadable or invalid image.

The detector supports more than identity search: face boxes and landmarks
drive alignment, crops, counts, redaction, face-aware reframing, thumbnail
selection, and dataset quality checks. The embeddings support verification,
unnamed-person clustering, similarity search, prototype selection, and
identity-outlier review. Neither model claims emotion, demographic attributes,
liveness, deepfake detection, or general image understanding. For resumable
folder indexing, SQLite search, and review workflows, use the companion
`mere-face-tools` plugin.

## Measure pose and motion

### Extract pose landmarks

`vision pose` uses the native platform runtime, so there's no model to
install on Apple platforms.

```bash
mere.run vision pose ./person.png \
  --json-output ./person-pose.json \
  --minimum-confidence 0.2
```

The JSON, written to `<stem>_pose.json` by default, contains typed body, hand,
and face subjects with stable point names, image dimensions, and per-point
confidence for temporal filtering and motion export. Coordinates are
normalized with a bottom-left origin. `--no-body`, `--no-hands`, and
`--no-face` skip subjects, `--max-hands` defaults to 2, and
`--minimum-confidence` defaults to 0.1. `--json` also prints the payload on
stdout.

### Generate dense optical flow

`vision flow` computes a motion vector for every pixel between two images of
equal size.

```bash
mere.run vision flow ./frame-001.png ./frame-002.png \
  --output ./frame-001-to-002.flo \
  --accuracy high
```

The output uses the Middlebury `.flo` format with full-resolution 32-bit
horizontal and vertical components. Typed JSON metadata, including magnitude
statistics, goes to `--json-output` or, with `--json`, to stdout.
`--accuracy` accepts `low`, `medium`, `high` (default), or `very-high`. The
command rejects images with different dimensions.

## Estimate depth and geometry

These commands share a pattern: `--output` names the output directory,
`--dry-run` verifies inputs and the checkpoint and prints the plan without
running inference, and `--json` prints the structured result on stdout. For
measured load, inference, and memory on an M4 Max, see the
[VFX geometry benchmarks](https://github.com/sawfwair/mere-run/blob/main/docs/benchmarks/vfx-geometry-apple-silicon.md).

### Estimate depth for a still image

`vision depth` runs native Marigold V2. Treat it as experimental: accuracy
parity with the reference hasn't been established.

```bash
mere.run model pull vision-depth-marigold-v2
mere.run vision depth ./photo.jpg --output ./photo-depth
```

The output directory (default `<stem>-depth` next to the input) contains a
depth EXR, a preview PNG, and a manifest JSON. Inference is one
rectified-flow step at a fixed timestep on a frozen, 4-bit quantized
Qwen-Image-Edit-2509 transformer with rank-128 adapters, so there's no step
count or guidance scale to set.

Depth is affine-invariant: it's recovered up to an unknown scale and shift for
each image. Values are normalized per image so that larger means farther, and
the manifest records the applied mapping and the raw range. The model doesn't
estimate a camera, so no intrinsics or point cloud are written. Use `vision
geometry` when you need metric units or a camera.

Options:

- `--max-edge` caps the longest inference edge, rounded to a multiple of 16
  (default 1,024). `--native` runs at the source resolution instead. Memory
  grows with resolution.
- `--checkpoint` selects a published variant: `log-stage2` (default, the paper
  model), `log-stage1`, `log-layered`, `uniform-base`, `uniform-layered`,
  `disparity-base`, or `disparity-layered`. The `*-layered` variants predict
  see-through depth behind glass. Only the default is part of the managed
  install; for the others, pass a local Marigold repository root to `--model`.
- `--receipt` prints a final JSON line that names the EXR, preview, and
  manifest. You can't combine it with `--dry-run`.

`vision depth` runs from the CLI and Studio's **Vision ▸ Depth** task only. It
has no API route: the depth API serves video depth, and the catalog declares
`vision-depth-marigold-v2` CLI-only so API discovery doesn't list it. Machine
admission treats every run as a large job from the catalog's 64 GB minimum,
because the BF16 base loads before quantization even though the download is
below the large-download threshold. Two depth runs therefore don't share the
machine.

The native runtime uses MLX affine 4-bit weights and deterministic VAE
encoding. It keeps the first image-modulation projection in full precision,
because quantizing it destroys depth structure; prequantized checkpoints must
preserve it too. The reference uses bitsandbytes NF4 and samples the VAE
posterior, and its quantized memory figures don't describe native loading,
which needs the BF16 base first.

### Generate temporally consistent video depth

`vision depth-video` runs native Video Depth Anything Small.

```bash
mere.run model pull vision-depth-vda-small
mere.run vision depth-video ./clip.mp4 --output ./clip-depth
```

The output directory (default `<stem>-depth`) contains per-frame depth EXRs,
preview PNGs, a review MP4, and a depth-sequence manifest. `--input-size`
caps the longest network edge, 14 through 1,008, before aspect-ratio
adjustment (default 518). `--max-frames` caps decoded source frames, 1 through
2,400 (default 240). The default model gives relative depth;
`--model vision-depth-vda-small-metric` gives metric depth. `--dry-run` also
hashes and decodes the bounded input before it prints the plan.

### Recover metric geometry from one image

`vision geometry` runs native MoGe-2.

```bash
mere.run model pull vision-geometry-moge2-small
mere.run vision geometry ./photo.jpg --output ./photo-geometry
```

The output directory (default `<stem>-geometry`) contains metric depth and
normal EXRs with preview PNGs, a validity mask, camera intrinsics JSON, a
point-cloud PLY, and a manifest. `--resolution-level` selects quality 0
through 9 (default 9), `--token-count` overrides the DINO base-token count (1
through 3,600), and `--max-points` caps the PLY point count.

### Solve multi-view geometry and cameras

`vision geometry-multiview` runs native DA3-Small across views of one scene,
in the order you pass them.

```bash
mere.run model pull vision-geometry-da3-small
mere.run vision geometry-multiview ./view-01.jpg ./view-02.jpg ./view-03.jpg \
  --output ./scene
```

It solves relative depth, per-view confidence, and cameras. The output
directory (default `<first-stem>-da3-scene`) contains per-view depth and
confidence EXRs with preview PNGs, camera JSON, colored PLY and GLB point
clouds, a Nerfstudio and 3D Gaussian splatting initialization handoff, and a
scene manifest.

| Option | What it sets | Default |
| --- | --- | --- |
| `--cameras` | JSON with one calibrated world-to-camera (W2C) camera per image | None |
| `--process-resolution` | Upper bound for the longest processed side | 504 |
| `--reference-view` | `first`, `middle`, `saddle-balanced`, or `saddle-similarity-range` | `saddle-balanced` |
| `--confidence-percentile` | Points below this percentile are discarded | 40 |
| `--max-points` | Colored points in scene exports | 1,000,000 |

## Reconstruct 3D objects

### Reconstruct a PBR object with TRELLIS.2

`vision image-to-3d-trellis2` reconstructs one isolated object as a
512-resolution O-Voxel mesh with base color, metallic, roughness, and alpha.
Accept the DINOv3 license on Hugging Face and authenticate your Hugging Face
client before the first pull.

```bash
mere.run model pull image-3d-trellis2-4b --accept-model-license
mere.run vision image-to-3d-trellis2 ./object.png \
  --output ./object-trellis2 \
  --seed 42
```

The input needs transparent alpha; the command crops the object before
conditioning. `--already-framed` opts an opaque, isolated, centered object into
black-background conditioning instead.

- `--seed` (default 42) controls deterministic MLX noise. `--texture-seed`
  re-rolls only the texture flow and replays geometry from `--seed`, so you
  can search palettes without changing a good shape.
- `--max-tokens` caps decoded O-Voxels (default 2,097,152). Raising it can
  increase memory use substantially.
- By default, the mesh is a watertight narrow-band remesh of the raw crust.
  `--remesh-band` (default 1.0 voxel) widens tear tolerance, `--seal-radius`
  (default 12 voxels; 0 turns it off) seals occluded cavities, and
  `--no-remesh` exports the raw porous crust.

The output contains OBJ, PLY, and GLB meshes with vertex color, a
`-textured.glb` with baked glTF PBR textures, a mesh manifest, a hashed
`.pbrvox` sidecar that preserves base color, metallic, roughness, and alpha for
every O-Voxel, and a run manifest. The command supports the official 512
pipeline only. For details, run `mere.run guide vision image-to-3d-trellis2`.

### Reconstruct with TripoSR or InstantMesh

`vision image-to-3d` runs native TripoSR on one image, and
`vision image-to-3d-multiview` runs native InstantMesh on views you supply:

```bash
mere.run model pull image-3d-triposr
mere.run vision image-to-3d ./object.png --output ./object-3d

mere.run model pull image-3d-instantmesh-base
mere.run vision image-to-3d-multiview \
  --view ./front.png --view ./right.png --view ./back.png --view ./left.png \
  --output ./object-instantmesh
```

TripoSR crops transparent PNG foregrounds automatically
(`--foreground-ratio`, default 0.85) unless you pass `--already-framed`.
`--resolution` sets its density grid, 2 through 512 (default 256), and
`--density-threshold` sets the isosurface threshold (default 25.0).

InstantMesh needs `--view` exactly four or six times, in order; it doesn't
generate views. `--resolution` sets its grid, 2 through 256 (default 128), and
`--cameras` takes an optional camera JSON file.

Both commands accept `--no-vertex-colors` for geometry-only export, plus
`--dry-run` and `--json`.

## Fused attention policy

Supported SAM 3.1, LightOn OCR, and selected vision-encoder attention shapes
use MLX fused scaled-dot-product attention by default. Unsupported shapes keep
their portable implementation. Set `MERERUN_FUSED_SDPA=0` for an emergency
compatibility fallback or a controlled A/B comparison.

The installed-model gate on 2026-07-11 used release binaries on an M4 Max with
128 GB unified memory. A SAM 3.1 text-prompt segmentation run was about 3%
faster (1.36 seconds to 1.32 seconds) and reduced peak footprint by 13.2%; all
11 exported masks were bit-exact, with only negligible score and box deltas. A
warm LightOn OCR run improved from 2.87 seconds to 1.97 seconds (1.46x
throughput) and reduced peak footprint by 55%, with byte-identical text.
Process RSS was effectively flat in both comparisons, so these measurements
aren't general RSS savings or guarantees for other models, prompts, or shapes.

## How the runtimes work

### Segmentation and tracking

1. The CLI resolves `vision-segment-sam31` from the model store, or uses the
   local root passed with `--model`.
2. The native SAM 3.1 runtime validates the root, loads the tokenizer,
   configuration, and weights, and preprocesses the image or video frames.
3. Text prompts run the detector once, and then run text, Detection
   Transformer (DETR), and mask decoding for each prompt.
4. Box and point prompts use the interactive SAM path. Video tracking reuses
   those prompts frame to frame after the seed frame.
5. Postprocessing applies thresholding, mask resizing, score ordering,
   non-maximum suppression (NMS), and optional mask export.
6. The runtime writes annotated media and structured JSON.

### Grounding

1. The CLI resolves `vision-ground-falcon-perception` from the model store, or
   uses the local root passed with `--model`.
2. The native Falcon runtime validates the root, loads the configuration,
   tokenizer, and weights, and preprocesses the image and queries.
3. The model autoregressively emits detections as coordinate and size tokens,
   and decodes optional segmentation masks.
4. Postprocessing derives normalized centers, sizes, boxes, and optional mask
   files.
5. The runtime writes an annotated image and structured JSON.

### OCR

1. The CLI resolves the OCR model for the selected backend.
2. The runtime loads the required components and normalizes the image into the
   expected tensor form.
3. Inference runs, and the command prints text on stdout without internal logs.

LightOnOCR 3-1B uses the dedicated LightOn runtime; the 0.8B and 4B variants
and Infinity-Parser2 use the Q35 runtime with the Qwen-family vision tower.
GLM-OCR and external Infinity runs shell out to upstream executables.

## Runtime entrypoints

The following table lists where each part of the runtime lives:

| Area | Location |
| --- | --- |
| CLI commands | `Sources/MereRunCLI/Commands/Vision*Command.swift` |
| Captioning and inspection | `Sources/MereRunCore/VLM/` |
| Multimodal embeddings | `Sources/MereRunCore/MultimodalEmbeddings/` |
| LightOn OCR | `Sources/MereRunCore/LightOnOCR/` |
| Q35 OCR runtime | `Sources/MereRunCore/Q35/`, `Sources/MereRunQwenModel/Q35Model.swift`, `Sources/MereRunQwenModel/Q35VisionTower.swift` |
| Falcon grounding | `Sources/MereRunCore/FalconPerception/` |
| SAM 3.1 | `Sources/MereRunCore/SAM3/` |
| Face analysis | `Sources/MereRunCore/FaceAnalysis/` |
| Pose | `Sources/MereRunCore/Pose/NativePoseDetector.swift` |
| Optical flow | `Sources/MereRunCore/OpticalFlow/NativeOpticalFlowGenerator.swift` |
| Marigold V2 | `Sources/MereRunCore/MarigoldV2/` |
| Video depth | `Sources/MereRunCore/VideoDepth/` |
| MoGe-2 | `Sources/MereRunCore/VisionGeometry/MoGe2/` |
| Depth Anything 3 | `Sources/MereRunCore/DepthAnything3/` |
| 3D reconstruction | `Sources/MereRunCore/TripoSR/`, `Sources/MereRunCore/Trellis2/`, `Sources/MereRunCore/InstantMesh/` |
