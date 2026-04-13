---
name: mere-run
description: Use mere.run for local AI inference — text chat, code generation, image generation, speech synthesis/transcription, vision (captioning, inspection, grounding, segmentation, tracking, OCR), music, video, embeddings, and API serving. Activate when the user wants to run local AI models, generate content, or serve models via API.
user-invocable: false
---

# mere.run — Local AI Inference Toolkit

mere.run is a native Swift/MLX CLI for on-device AI inference on Apple Silicon. All models run locally with no cloud dependency. Models auto-download on first use.

Binary: `mere.run` (or `swift run mere.run` during development)

## Quick Reference

### Text Chat

```bash
mere.run text chat -p "Explain quantum computing simply"
mere.run text chat -p "Write a haiku" -s "You are a poet" --temperature 0.9
mere.run text chat -p "Analyze this" --model text-chat-q35-nano --stats
mere.run text chat -p "Think step by step" --thinking
```

**Models:** `text-chat-gemma4` (default, Gemma 4 E4B native), `text-chat-q35-nano` (Qwen3.5-35B-A3B 4-bit), `text-chat-q35`, `text-chat-psi-agent`

**Key options:** `-p` prompt, `-s` system prompt, `--max-tokens` (default 2048), `--temperature` (default 0.7), `--top-p` (default 0.9), `--model`, `--thinking`, `--stats`, `--kv-bits` (Gemma4 KV cache quantization)

### Code Generation

```bash
mere.run text code -p "Write a Swift function to reverse a string"
mere.run text code -p "Explain this code" -m ./model.gguf --stream
```

Uses llama.cpp with GGUF models (Qwen3-Coder). Key options: `-p`, `-s`, `-m` model path, `--stream`, `--stats`

### Text Embeddings

```bash
mere.run text embed "hello world"
mere.run text embed "foo" "bar" --output embeddings.json --pretty
```

Native Qwen3-Embedding-0.6B. Options: `-m` model, `--max-tokens`, `-o` output JSON, `--pretty`

### Image Generation

```bash
mere.run image generate -p "a sunset over mountains" -o sunset.png
mere.run image generate -p "portrait" -W 512 -H 768 -s 8 --seed 42
mere.run image generate -p "stylized" -i input.png --strength 0.75
mere.run image generate -p "detailed scene" -l style.safetensors --lora-scale 0.8
```

**Key options:** `-p` prompt, `-n` negative prompt, `-o` output path, `-W`/`-H` dimensions (default 1024x1024), `-s` steps (default 4), `--seed`, `-i` input image (img2img), `--strength` (img2img, default 0.75), `-l` LoRA path, `--lora-scale`, `--cfg` CFG scale

### Speech Synthesis (TTS)

```bash
mere.run speech synthesize "Hello, world!" -o hello.wav
mere.run speech synthesize "Welcome." --voice "A calm British male voice" -o welcome.wav
mere.run speech synthesize "Clone test" --mode clone --ref-audio ref.wav -o cloned.wav
mere.run speech synthesize "Clone test" --mode clone --profile myvoice -o out.wav
```

Qwen3-TTS. **Key options:** `-o` output WAV (required), `-v` voice description, `--mode` style|clone, `--profile` saved voice, `--ref-audio`/`--ref-text` for cloning, `--save-profile` to save clone, `--stream`, `--temperature` (default 0.6)

### Speech Transcription (ASR)

```bash
mere.run speech transcribe audio.wav
mere.run speech transcribe audio.wav --backend parakeet -o transcript.txt
mere.run speech transcribe audio.wav --task translate
mere.run speech transcribe audio.wav --stream
```

Auto-routes: Parakeet for transcription, Qwen for translation. **Key options:** `-o` output file, `--backend` auto|parakeet|qwen, `--task` transcribe|translate, `--language`, `--stream`, `--timestamps`/`--no-timestamps`

### Voice Profiles

```bash
mere.run speech profile list
mere.run speech profile show myvoice
mere.run speech profile delete myvoice
```

### Vision — Captioning

```bash
mere.run vision caption photo1.jpg photo2.jpg
mere.run vision caption *.png --output-dir ./captions --prompt "Describe the scene"
```

Qwen3-VL. Writes `.txt` caption per image. Options: `-m` model, `-o` output dir, `--prompt`, `--max-tokens` (default 96), `--temperature` (default 0.2)

### Vision — Inspection (VLM Q&A)

```bash
mere.run vision inspect photo.jpg "What is happening in this image?"
mere.run vision inspect diagram.png "List all the components"
```

Qwen3-VL. Options: `-m` model, `--max-tokens` (default 2048), `--temperature` (default 0.7)

### Vision — Grounding (Falcon Perception)

```bash
mere.run model pull vision-ground-falcon-perception
mere.run vision ground photo.jpg --query "person"
mere.run vision ground photo.jpg --query "person in red" --query "phone" -o grounded.png
mere.run vision ground photo.jpg --query "cat" --json-output grounded.json --mask-output-dir ./masks
```

Native Falcon Perception (grounded detection + segmentation). If `--model` is omitted, the command resolves the managed model id `vision-ground-falcon-perception`.

**Key options:** positional image path, `--query`/`--prompt` repeated grounding expressions, `-m`/`--model` managed id or local model root, `-o` annotated output image, `--json-output`, `--mask-output-dir`

**Output shape:** annotated image written to `<image>_grounded.<ext>` by default, JSON metadata written to `<image>_grounded.json`, optional per-detection mask PNGs under `--mask-output-dir`. JSON detections include `query`, normalized `xy`, normalized `hw`, derived `box`, optional `score`, and optional `maskPath`.

### Vision — Segmentation (SAM 3.1)

```bash
mere.run vision segment photo.jpg --prompt "a person" "a phone"
mere.run vision segment photo.jpg --box 100,100,300,400,person
mere.run vision segment photo.jpg --point 150,200,positive
mere.run vision segment photo.jpg --prompt "cat" -o annotated.png --mask-output-dir ./masks
```

Native SAM 3.1. Options: `--prompt` text prompts, `--box` x1,y1,x2,y2[,label], `--point` x,y,positive|negative, `-o` annotated output, `--json-output`, `--mask-output-dir`, `--threshold` (default 0.3), `--show-boxes`, `--multimask`

### Vision — Tracking (SAM 3.1)

```bash
mere.run vision track video.mp4 --prompt "the red car"
mere.run vision track-live --prompt "person"
```

Track objects through video or live camera feed.

### Vision — OCR

```bash
mere.run vision ocr document.png
```

LightOnOCR or GLM-OCR.

### Music Generation

```bash
mere.run music generate --caption "upbeat electronic track" --lyrics "verse lyrics here" -o song.wav
```

ACE-Step. Options: `--caption`, `--lyrics`, `-o` output

### Video Generation

```bash
mere.run video generate "a cinematic drone flythrough over snowy mountains" -o video.mp4
mere.run video generate "woman walking in neon rain" --image frame.png
mere.run video generate "city time-lapse" --width 768 --height 512 --num-frames 65
```

Native LTX pipeline. Options: `-o` output MP4, `--variant` distilled|unified-av, `--model-root`, `--width`/`--height`, `--num-frames` (must be 8n+1), `--fps` (default 24), `--seed`, `--image` (img2vid), `--image-strength`

### API Server (OpenAI-compatible)

```bash
mere.run api serve
mere.run api serve --engine text-chat-gemma4
mere.run api serve --engine text-chat-q35 -m ~/models/q35-nano
mere.run api serve -m ./model.gguf --port 11434
```

Serves at `http://localhost:8080` by default. Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (streaming supported).

**Engines:** `text-code` (default, GGUF/llama.cpp), `text-chat-gemma4`, `text-chat-q35`, `text-chat-klein`

**Key options:** `--port` (default 8080), `--host` (default 127.0.0.1), `-m` model path, `--engine`, `--context-size` (default 32768), `--kv-bits`/`--kv-quant-scheme` (Gemma4)

### Model Management

```bash
mere.run model list                    # Show all models and install status
mere.run model pull text-chat-gemma4   # Download a model
mere.run model pull vision-ground-falcon-perception
mere.run model remove text-chat-q35    # Remove a model
mere.run model info text-chat-gemma4   # Show manifest and validation
mere.run model repair-manifests        # Fix missing manifests
```

Models stored at `~/Library/Application Support/MereRun/models/<model-id>/`. Override with `--models-root` or `MERERUN_MODELS_DIR`.

## Additional Documentation

For deeper details on specific runtimes, architecture, or configuration, see the docs directory:

- [CLI Reference](docs/cli.md)
- [Getting Started](docs/getting-started.md)
- [Configuration](docs/configuration.md)
- [Model Sources](docs/model-sources.md)
- Runtime details: [Text](docs/runtime/text.md), [Image](docs/runtime/image.md), [Speech](docs/runtime/speech.md), [Vision](docs/runtime/vision.md), [Music](docs/runtime/music.md), [Video](docs/runtime/video.md), [API Server](docs/runtime/api-server.md), [Model Management](docs/runtime/model-management.md)
