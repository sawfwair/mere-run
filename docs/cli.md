# mere.run CLI

`mere.run` is the public command-line interface for the OSS `mere.run` package. It exposes a modality-first command tree for image, text, speech, vision, music, video, model management, and local API serving.

If you are looking for the broader docs set, start at [mere.run Documentation](/).

## Overview

```bash
swift run mere.run --help
```

Public tree:

- `mere.run image generate`
- `mere.run image validate`
- `mere.run text chat`
- `mere.run text code`
- `mere.run text embed`
- `mere.run speech synthesize`
- `mere.run speech transcribe`
- `mere.run speech profile { list, create, delete }`
- `mere.run vision caption`
- `mere.run vision inspect`
- `mere.run vision segment`
- `mere.run vision track`
- `mere.run vision track-live`
- `mere.run vision ocr`
- `mere.run music generate`
- `mere.run video generate`
- `mere.run video export-latents`
- `mere.run model { list, info, pull, remove, repair-manifests }`
- `mere.run api serve`

## Global model-store override

The CLI honors the shared models root override:

```bash
swift run mere.run --models-root /Volumes/FastSSD/mererun-models model list
```

That is equivalent to setting:

```bash
export MERERUN_MODELS_DIR=/Volumes/FastSSD/mererun-models
```

## Canonical managed model IDs

See [`model-sources.md`](./model-sources.md) for the full source story. The most common managed IDs are:

- Images: `image-klein-nano`, `image-klein-base`, `image-klein-max`, `image-zimage-nano`, `image-zimage-base`, `image-zimage-max`
- Text chat: `text-chat-mebot`, `text-chat-psi-agent`, `text-chat-q35`, `text-chat-q35-nano`
- Text code: `text-code-qwen3`
- Text embed: `text-embed-qwen3-0.6b`
- Speech TTS: `speech-tts-qwen3-nano`, `speech-tts-qwen3-customvoice`
- Speech ASR: `speech-asr-qwen3`, `speech-asr-parakeet`
- Vision OCR: `vision-ocr-lighton`
- Vision segmentation / tracking: `vision-segment-sam31`
- Music: `music-acestep`
- Video: `video-ltx-av`

For subsystem-specific implementation guides, see:

- [Image Runtime](./runtime/image.md)
- [Text Runtime](./runtime/text.md)
- [Speech Runtime](./runtime/speech.md)
- [Vision Runtime](./runtime/vision.md)
- [Music Runtime](./runtime/music.md)
- [Video Runtime](./runtime/video.md)

## Common workflows

### Pull and inspect models

```bash
export MERERUN_MODEL_SOURCE_BASE_URL=https://your-host.example/models/
swift run mere.run model list
swift run mere.run model pull image-zimage-max
swift run mere.run model info image-zimage-max
```

### Generate an image

```bash
swift run mere.run image generate \
  --prompt "a ceramic mug in soft morning light" \
  --output ./mug.png
```

### Chat locally

```bash
swift run mere.run text chat \
  --prompt "Explain classifier-free guidance."
```

### Generate speech and transcribe it back

```bash
swift run mere.run speech synthesize \
  "Hello from mere.run" \
  --output ./hello.wav

swift run mere.run speech transcribe ./hello.wav --backend auto
```

### Inspect, segment, track, and OCR

```bash
swift run mere.run vision inspect ./diagram.png "What does this diagram show?"
swift run mere.run vision segment ./photo.jpg --prompt "a cat"
swift run mere.run vision track ./clip.mp4 --prompt "a cat"
swift run mere.run vision ocr ./page.png --backend lighton
```

### Generate music

```bash
swift run mere.run model pull music-acestep
swift run mere.run music generate \
  "upbeat electronic groove" \
  --output ./track.wav
```

### Generate video

```bash
swift run mere.run video generate \
  "a cinematic drone flythrough over snowy mountains" \
  --variant unified-av \
  --model-root ~/Library/Application\ Support/MereRun/models/video-ltx-av \
  --output ./clip.mp4
```

### Serve a local API

```bash
swift run mere.run api serve --engine text-chat-q35
```

## Command reference

Model installation in the OSS repo is explicit. Before running `mere.run model pull`, configure either `MERERUN_MODEL_SOURCE_BASE_URL`, a signed download endpoint, or direct R2 credentials. See [`configuration.md`](./configuration.md) and [`model-sources.md`](./model-sources.md).

### `mere.run image generate`

Generate a PNG with a local image model.

```bash
swift run mere.run image generate --prompt "<text>" [options]
```

Key options:

- `--prompt`: required text prompt
- `--model`: canonical model id or local model path
- `--output`: output PNG path
- `--width`, `--height`
- `--steps`
- `--cfg`
- `--input`: image-to-image source
- `--strength`: image-to-image strength
- `--lora`, `--lora-scale`
- `--quiet`

Examples:

```bash
swift run mere.run image generate --prompt "a black cat on a red sofa"
swift run mere.run image generate --model image-zimage-nano --prompt "retro robot illustration" --output ./robot.png
swift run mere.run image generate --prompt "turn this into a pencil sketch" --input ./photo.png --strength 0.6
```

### `mere.run image validate`

Run advanced deterministic validation for the local image families.

```bash
swift run mere.run image validate --family zimage --test all
swift run mere.run image validate --family klein --test vae --output ./validation_output
swift run mere.run image validate --save-reference
swift run mere.run image validate --compare --reference-dir ./validation_output
```

Key options:

- `--family`: `zimage` or `klein`
- `--test`: `vae`, `encoder`, `transformer`, `pipeline`, or `all`
- `--output`
- `--save-reference`
- `--compare`
- `--reference-dir`

### `mere.run text chat`

Run local text chat with the Q35 or Psi family.

```bash
swift run mere.run text chat --prompt "<text>" [options]
```

Key options:

- `--prompt`
- `--system`
- `--model`: canonical model id
- `--model-root`: explicit local model root
- `--max-tokens`
- `--temperature`
- `--top-p`
- `--thinking`
- `--stats`
- `--quiet`

Examples:

```bash
swift run mere.run text chat --prompt "What is classifier-free guidance?"
swift run mere.run text chat --model text-chat-q35-nano --prompt "Explain speculative decoding."
swift run mere.run text chat --thinking --stats --prompt "How would you design a tokenizer?"
```

### `mere.run text code`

Run local code generation with GGUF models through the vendored `llama.cpp` runtime.

```bash
swift run mere.run text code --prompt "<text>" [options]
```

Key options:

- `--prompt`
- `--model`: GGUF file or canonical code model id if your local setup resolves it
- `--stream`
- `--stats`
- `--temperature`
- `--top-p`
- `--max-tokens`

Examples:

```bash
swift run mere.run text code --prompt "Write a Swift function to reverse a string"
swift run mere.run text code --model ./Qwen3-Coder-Next-Q4_K_M.gguf --stream --prompt "Implement a trie in Rust"
```

### `mere.run text embed`

Generate embeddings with the native Qwen3 embedding model.

```bash
swift run mere.run text embed "semantic search query"
swift run mere.run text embed "foo" "bar" --output embeddings.json --pretty
```

Key options:

- positional text arguments
- `--model`
- `--max-tokens`
- `--output`
- `--pretty`

### `mere.run speech synthesize`

Generate speech from text with Qwen3-TTS.

```bash
swift run mere.run speech synthesize "<text>" --output ./speech.wav [options]
```

Key options:

- `--output`: required
- `--model`: canonical speech TTS id or local model path
- `--voice`
- `--mode`: `style` or `clone`
- `--profile`
- `--ref-audio`
- `--ref-text`
- `--language`
- `--save-profile`
- `--temperature`
- `--stream`
- `--stream-chunk-tokens`
- `--quiet`

Examples:

```bash
swift run mere.run speech synthesize "Hello from mere.run" --output ./hello.wav
swift run mere.run speech synthesize "Welcome aboard" --voice "A calm British male voice" --output ./welcome.wav
swift run mere.run speech synthesize "Read this in my cloned voice" --mode clone --profile my-voice --output ./clone.wav
```

### `mere.run speech transcribe`

Transcribe or translate local audio with the speech backends.

```bash
swift run mere.run speech transcribe <audio.wav> [options]
```

Key options:

- positional audio path
- `--backend`: `auto`, `qwen`, or `parakeet`
- `--task`: `transcribe` or `translate`
- `--model`
- `--language`
- `--max-tokens`
- `--stream`
- `--stream-chunk-ms`
- `--stream-decode-ms`
- `--no-timestamps`
- `--output`
- `--quiet`

Examples:

```bash
swift run mere.run speech transcribe ./audio.wav
swift run mere.run speech transcribe ./audio.wav --task translate --backend qwen
swift run mere.run speech transcribe ./audio.wav --stream --output ./transcript.txt
```

### `mere.run speech profile`

Manage reusable voice clone profiles.

Subcommands:

- `mere.run speech profile list`
- `mere.run speech profile create`
- `mere.run speech profile delete`

Examples:

```bash
swift run mere.run speech profile list
swift run mere.run speech profile create \
  --name narrator \
  --audio ./ref.wav \
  --text "reference transcript"
swift run mere.run speech profile delete --id <uuid>
```

### `mere.run vision caption`

Generate captions for one or more images.

```bash
swift run mere.run vision caption ./images/*.png
swift run mere.run vision caption ./images/*.png --output-dir ./captions
```

### `mere.run vision inspect`

Ask a direct question about an image.

```bash
swift run mere.run vision inspect ./diagram.png "What does this diagram show?"
```

### `mere.run vision segment`

Segment prompted objects in an image using the native SAM 3.1 runtime.

```bash
swift run mere.run model pull vision-segment-sam31
swift run mere.run vision segment ./photo.jpg --prompt "a cat"
```

Key options:

- `--prompt`: one or more text object prompts
- `--box`: one or more `x1,y1,x2,y2[,label]` geometry prompts
- `--point`: one or more `x,y,positive[,label]` or `x,y,negative[,label]` geometry prompts
- `--model`: managed model id or local SAM 3.1 model root
- `--output`: annotated image path
- `--json-output`: metadata path
- `--mask-output-dir`: optional per-object mask export directory
- `--threshold`
- `--resolution`
- `--show-boxes`
- `--multimask`: emit up to three candidates per geometry-prompted object

Defaults:

- annotated image: `<image-stem>_segmented.<ext>`
- JSON metadata: `<image-stem>_segmented.json`

Notes:

- still-image runs accept text, box, and point prompts in the same invocation
- `--mask-output-dir` writes one PNG mask per exported detection candidate
- zero detections still produce annotated output plus JSON metadata

Examples:

```bash
swift run mere.run vision segment ./photo.jpg --prompt "a cat"
swift run mere.run vision segment ./photo.jpg --prompt "a person" "a phone" --show-boxes
swift run mere.run vision segment ./photo.jpg --box "120,80,420,760,person" --mask-output-dir ./masks
swift run mere.run vision segment ./photo.jpg --point "512,384,positive,person" --point "700,200,negative,person"
swift run mere.run vision segment ./photo.jpg --prompt "a dog" --output ./photo-segmented.png --json-output ./photo-segmented.json
```

### `mere.run vision track`

Track prompted objects through a video with the native SAM 3.1 runtime.

```bash
swift run mere.run model pull vision-segment-sam31
swift run mere.run vision track ./clip.mp4 --prompt "a dog"
```

Key options:

- `--prompt`: one or more text prompts used to seed objects on the init frame
- `--box`: one or more `x1,y1,x2,y2[,label]` geometry prompts
- `--point`: one or more `x,y,positive[,label]` or `x,y,negative[,label]` geometry prompts
- `--init-frame`: starting frame index for seeding
- `--end-frame`: optional inclusive final frame index
- `--output`: annotated video path
- `--json-output`: tracking metadata path
- `--mask-output-dir`: optional per-frame mask export directory
- `--show-boxes`
- `--show-labels`

Defaults:

- annotated video: `<video-stem>_tracked.mp4`
- JSON metadata: `<video-stem>_tracked.json`

Notes:

- text prompts seed objects on `--init-frame`, then the native tracker reuses geometry prompts for later frames
- box and point prompts seed explicit tracked objects directly on the init frame
- `--mask-output-dir` writes per-frame mask PNGs under frame-named subdirectories
- zero seeded objects still produce an annotated video and JSON summary

Examples:

```bash
swift run mere.run vision track ./clip.mp4 --prompt "a dog" --init-frame 12
swift run mere.run vision track ./clip.mp4 --box "40,50,120,180,dog" --box "200,80,320,260,person" --show-boxes
```

### `mere.run vision track-live`

Capture a camera clip and run native SAM 3.1 tracking over the recorded session.

```bash
swift run mere.run vision track-live --output ./live.mp4 --prompt "a person"
```

Key options:

- `--prompt`: one or more text prompts used to seed objects from the first captured frame
- `--camera`: camera device index
- `--duration-seconds`
- `--output`: annotated video path
- `--json-output`: tracking metadata path
- `--show-boxes`
- `--show-labels`

Notes:

- `track-live` currently records a camera clip first, then runs tracking over the recorded media
- live mode accepts text prompts only in the current implementation
- `--output` is required; `--json-output` is optional

### `mere.run vision ocr`

Extract text from one or more images.

```bash
swift run mere.run vision ocr <images...> [options]
```

Key options:

- `--backend`: `lighton`, `glm`, or `compare`
- `--model`: path to the LightOn OCR root when using the LightOn backend
- `--max-tokens`
- `--quiet`

Examples:

```bash
swift run mere.run model pull vision-ocr-lighton
swift run mere.run vision ocr ./page.png --backend lighton --model ~/Library/Application\ Support/MereRun/models/vision-ocr-lighton
swift run mere.run vision ocr ./page.png --backend glm
```

### `mere.run music generate`

Generate music from a caption and optional lyrics using the native ACE-Step pipeline.

```bash
swift run mere.run music generate "<caption>" [options]
```

Key options:

- `--output`
- `--checkpoints-root`
- `--lyrics`
- `--lyrics-file`
- `--duration`
- `--steps`
- `--use-lm`
- `--lm-subdirectory`
- `--text-subdirectory`
- `--seed`
- `--quiet`

Environment:

- `MERERUN_MUSIC_ACESTEP_ROOT`

Examples:

```bash
swift run mere.run music generate "upbeat electronic groove" --output ./track.wav
swift run mere.run music generate \
  "ambient piano and soft rain" \
  --lyrics-file ./lyrics.txt \
  --duration 8 \
  --steps 4 \
  --output ./ambient.wav
```

### `mere.run video generate`

Generate MP4 video with the native LTX pipelines.

```bash
swift run mere.run video generate "<prompt>" [options]
```

Key options:

- `--variant`: `distilled` or `unified-av`
- `--model-root`
- `--output`
- `--width`, `--height`
- `--num-frames`
- `--fps`
- `--seed`
- `--image`
- `--image-strength`
- `--quiet`

Environment:

- `MERERUN_VIDEO_LTX_MODEL_ROOT`

Examples:

```bash
swift run mere.run video generate \
  "a cinematic drone flythrough over snowy mountains" \
  --variant unified-av \
  --model-root ~/Library/Application\ Support/MereRun/models/video-ltx-av

swift run mere.run video generate \
  "woman walking in neon rain" \
  --image frame.png \
  --output ./rain.mp4
```

### `mere.run video export-latents`

Run native distilled LTX denoising and export the final latent tensor.

```bash
swift run mere.run video export-latents \
  --model-root /path/to/distilled-ltx \
  --output out.safetensors \
  "a cinematic drone flyover at sunrise"
```

### `mere.run model list`

List all managed model IDs and whether they are installed.

```bash
swift run mere.run model list
```

### `mere.run model pull`

Download a managed model archive into the local model store.

```bash
swift run mere.run model pull image-zimage-max
swift run mere.run model pull --all
```

### `mere.run model info`

Inspect a canonical model ID or a local model root.

```bash
swift run mere.run model info image-zimage-max
swift run mere.run model info /path/to/model/root --components
swift run mere.run model info text-chat-q35 --json
```

### `mere.run model remove`

Delete an installed managed model by canonical ID.

```bash
swift run mere.run model remove image-zimage-max
swift run mere.run model remove image-zimage-max --force
```

### `mere.run model repair-manifests`

Write missing `mererun_model.json` manifests for known local model roots.

```bash
swift run mere.run model repair-manifests
swift run mere.run model repair-manifests --dry-run
```

### `mere.run api serve`

Start an OpenAI-compatible local API server.

```bash
swift run mere.run api serve [options]
```

Current endpoint surface:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Engine values:

- `text-code`
- `text-chat-klein`
- `text-chat-q35`

Examples:

```bash
swift run mere.run api serve
swift run mere.run api serve --engine text-chat-q35
swift run mere.run api serve --engine text-code --model ./Qwen3-Coder-Next-Q4_K_M.gguf
swift run mere.run api serve --host 0.0.0.0 --port 11434
```

## Validation and smoke runs

Standard repo validation:

```bash
./scripts/check.sh
```

Fast smoke suite:

```bash
./scripts/e2e_smoke.sh --core
```

Installed-model sweep:

```bash
./scripts/migrate_model_store.sh
./scripts/e2e_smoke.sh --installed
```
