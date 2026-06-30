# mere.run CLI

`mere.run` is the public command-line interface for the OSS `mere.run`
package. It exposes a modality-first command tree for image, text, speech,
vision, music, SFX, video, model management, local status snapshots, and local API
serving.

If you are looking for the broader docs set, start at [mere.run Documentation](/).

## Overview

```bash
swift run mere.run --help
```

Public tree:

- `mere.run guide`
- `mere.run image generate`
- `mere.run image train-lora`
- `mere.run image validate`
- `mere.run text chat`
- `mere.run text code`
- `mere.run text embed`
- `mere.run text anonymize`
- `mere.run speech synthesize`
- `mere.run speech transcribe`
- `mere.run speech profile { list, create, delete }`
- `mere.run vision caption`
- `mere.run vision inspect`
- `mere.run vision ground`
- `mere.run vision segment`
- `mere.run vision track`
- `mere.run vision track-live`
- `mere.run vision ocr`
- `mere.run music analyze`
- `mere.run music generate`
- `mere.run music realtime`
- `mere.run sfx generate`
- `mere.run video generate`
- `mere.run video export-latents`
- `mere.run model { list, capabilities, info, pull, remove, runtime, benchmark, repair-manifests }`
- `mere.run status`
- `mere.run api serve`
- `mere.run open-webui quickstart`
- `mere.run plugin { list, info, install, doctor }`
- `mere.run setup`
- `mere.run agent { onboard, install-pi, start }`

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

See [`model-sources.md`](./model-sources.md) for the full source story,
including which IDs are pullable from Hugging Face. The most common managed IDs
are:

- Images: `image-klein-nano`, `image-klein-base`, `image-klein-base-9b`, `image-klein-max`,
  `image-bonsai-binary`, `image-bonsai-ternary`, `image-zimage-nano`, `image-zimage-base`, `image-zimage-max`,
  `image-hidream-o1`, `image-hidream-o1-dev`, `image-krea2-raw`,
  `image-krea2-turbo`,
  `image-ideogram4-sdnq-uint4`
- Text chat: `text-chat-gemma4`, `text-chat-mebot`, `text-chat-psi-agent`, `text-chat-q36-nano`, `text-chat-lfm25-a1b-8bit`
- Text code / agents: `text-agent-qwen35-9b`, `text-agent-ornith-9b`, `text-agent-ornith-35b-mlx`, `text-agent-ornith-35b`, `text-code-north-mini`, `text-code-qwen3`
- Text embed: `text-embed-qwen3-0.6b`
- Text anonymize: `text-anonymize-privacy-filter`
- Speech TTS: `speech-tts-qwen3-nano`, `speech-tts-qwen3-customvoice`
- Speech ASR: `speech-asr-qwen3`, `speech-asr-parakeet`
- Vision OCR: `vision-ocr-lighton`, `vision-ocr-infinity-flash`,
  `vision-ocr-infinity-pro-int8`, `vision-ocr-infinity-pro`
- Vision segmentation / tracking: `vision-segment-sam31`
- Vision grounding: `vision-ground-falcon-perception`
- Music: `music-acestep`, `music-acestep-xl-turbo`, `music-acestep-xl-turbo-lm4b`, `music-magenta-rt2-small`, `music-magenta-rt2-base`
- SFX: `sfx-woosh-dflow`, `sfx-woosh-flow`
- Video: `video-ltx-av`, `video-ltx23-av-mlx`

For subsystem-specific implementation guides, see:

- [Image Runtime](./runtime/image.md)
- [Text Runtime](./runtime/text.md)
- [Speech Runtime](./runtime/speech.md)
- [Vision Runtime](./runtime/vision.md)
- [Music Runtime](./runtime/music.md)
- [SFX Runtime](./runtime/sfx.md)
- [Video Runtime](./runtime/video.md)

## Common workflows

### Pull and inspect models

```bash
swift run mere.run model list
swift run mere.run status
swift run mere.run model capabilities
swift run mere.run model runtime get text-chat-gemma4
swift run mere.run model pull image-zimage-nano
swift run mere.run model info image-zimage-nano
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

swift run mere.run model pull music-acestep-xl-turbo
swift run mere.run music generate \
  "upbeat electronic groove" \
  --model music-acestep-xl-turbo \
  --output ./xl-track.wav

swift run mere.run music analyze ./song.mp3 \
  --model music-acestep-xl-turbo-lm4b \
  --lm-subdirectory acestep-5Hz-lm-4B

swift run mere.run model pull music-magenta-rt2-small
swift run mere.run music realtime \
  "ambient modular synths with brushed drums" \
  --model music-magenta-rt2-small \
  --duration 4 \
  --output ./live.wav \
  --no-play
```

### Generate sound effects

```bash
swift run mere.run model pull sfx-woosh-dflow
swift run mere.run sfx generate \
  "metal wrench dropping onto concrete, bright clang and brief ring" \
  --model sfx-woosh-dflow \
  --duration 5 \
  --output ./wrench-clang.wav
```

### Generate video

```bash
swift run mere.run video generate \
  "a cinematic drone flythrough over snowy mountains" \
  --variant unified-av \
  --model-root ~/Library/Application\ Support/MereRun/models/video-ltx23-av-mlx \
  --output ./clip.mp4
```

### Serve a local API

```bash
swift run mere.run api serve --engine text-chat-gemma4
```

### Install a companion plugin

Official companion plugins are distributed outside the Swift package. The CLI
reads the live catalog from `sawfwair/mere-plugins`, prints the exact install
command by default, and only executes it when `--yes` is present.

```bash
swift run mere.run plugin list
swift run mere.run plugin info mere-runpod
swift run mere.run plugin install mere-runpod
swift run mere.run plugin install mere-runpod --yes
swift run mere.run plugin doctor mere-runpod
```

## Command reference

Model installation in the OSS repo is explicit. `mere.run model pull` uses cataloged Hugging Face snapshots only; local-path-only models must be supplied with command-specific `--model` or `--model-root` options. See [`configuration.md`](./configuration.md) and [`model-sources.md`](./model-sources.md).

### `mere.run plugin`

Discover and install official companion plugins from the live plugin catalog.
Plugins are separate executables, not code loaded into the `mere.run` process.

```bash
swift run mere.run plugin list
swift run mere.run plugin info mere-runpod
swift run mere.run plugin install mere-runpod --yes
swift run mere.run plugin doctor mere-runpod
```

Key options:

- `--catalog-url`: plugin catalog URL or local JSON path
- `--json`: emit catalog or plugin metadata as JSON
- `--channel`: install channel, defaulting to the catalog default
- `--yes`: execute the install command; omitted means dry-run preview
- `--force`: pass `--force` to `pipx install`

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
- `--steps`: override the model-specific step default
- `--cfg`: override the model-specific CFG default
- `--input`: image-to-image source. For FLUX.2 Klein, this is treated as a
  single reference image.
- `--ref-image`: repeatable FLUX.2 Klein or HiDream O1 reference image
- `--keep-original-aspect`: preserve one HiDream reference image's aspect ratio
- `--strength`: image-to-image/reference change strength
- `--structured-prompt`, `--json-prompt`: expand the prompt into a structured JSON caption with a local text chat model before image generation
- `--structured-prompt-model`: text chat model id for the adapter; defaults to `text-chat-gemma4-12b-4bit`
- `--structured-prompt-output`: write the generated structured JSON caption to a file
- `--lora`, `--lora-scale`
- `--quiet`

Unless `--quiet` is set, progress diagnostics on stderr include the native image
backend, for example `native MLX/Metal (default device: gpu)` on Apple Silicon.

Examples:

```bash
swift run mere.run image generate --prompt "a black cat on a red sofa"
swift run mere.run image generate --model image-zimage-nano --prompt "retro robot illustration" --output ./robot.png
swift run mere.run image generate --model image-bonsai-binary --prompt "sunlit greenhouse bonsai" --output ./bonsai.png
swift run mere.run image generate --model image-krea2-turbo --prompt "translucent portable speaker product photo" --steps 8 --output ./speaker.png
swift run mere.run image generate --prompt "turn this into a pencil sketch" --input ./photo.png --strength 0.6
swift run mere.run image generate --model image-ideogram4-sdnq-uint4 --prompt "a knight and a white horse in a sunny meadow" --structured-prompt --structured-prompt-output ./knight-prompt.json --output ./knight.png
swift run mere.run image generate \
  --model image-hidream-o1-dev \
  --prompt "put this subject in a studio portrait" \
  --ref-image ./subject.png \
  --output ./portrait.png
swift run mere.run image generate \
  --model image-klein-base \
  --prompt "a trading card character with the same sticker layout" \
  --ref-image ./card-reference.png \
  --output ./card.png
```

### `mere.run image train-lora`

Train a local text-to-image LoRA adapter. Krea 2 LoRAs are trained on
`image-krea2-raw` and can be used with `image-krea2-turbo` for fast inference.
FLUX.2 Klein LoRAs are trained on a Klein base model selected with `--model`
and can be used with Klein generation via `image generate --lora`.

```bash
swift run mere.run model pull image-krea2-raw
swift run mere.run model pull image-krea2-turbo
swift run mere.run image train-lora \
  --data ./style-dataset \
  --output ./style-krea2.safetensors \
  --recipe krea-cinematic-style \
  --quiet
swift run mere.run image generate \
  --model image-krea2-turbo \
  --prompt "a studio portrait in the trained style" \
  --lora ./style-krea2.safetensors \
  --output ./style-preview.png
```

For quick Krea smoke runs, `krea-fast-style` trains on `image-krea2-raw` for
100 steps with LR `0.0005`, 10-step warmup/cosine decay, 768 square, rank 32,
and alpha 32. Treat it as a proof pass and inspect images before trusting the
adapter. For stronger widescreen style datasets, `krea-cinematic-style` uses
200 steps, LR `0.0001`, 20-step warmup/cosine decay, 768x416, rank 32, alpha
32, and compiled-step disablement; override `--width`/`--height` for other
source aspects.

For practical Klein Base 9B style training, the `klein-fast-style` recipe uses
the undistilled BF16 `image-klein-base-9b` model, 1000 steps, LR `0.00005`,
max side `512`, the fast Klein target surface, disk-backed latent caching,
compiled-step disablement, and 250-step checkpoints:

```bash
swift run mere.run model pull image-klein-base-9b
swift run mere.run model pull image-klein-9b
swift run mere.run image train-lora \
  --data ./style-dataset \
  --output ./style-klein.safetensors \
  --recipe klein-fast-style \
  --quiet
swift run mere.run image generate \
  --model image-klein-9b \
  --prompt "TRIGGER_TOKEN a studio portrait in the trained style" \
  --lora ./style-klein.safetensors \
  --lora-scale 2.0 \
  --output ./style-preview.png
```

Dataset folders use image files with matching `.txt` captions:

```text
style-dataset/
  001.png
  001.txt
  002.jpg
  002.txt
```

Key options:

- `--data`: dataset directory with image + caption pairs
- `--output`: output `.safetensors` adapter path
- `--model`: Raw/base model id or local model path; defaults to `image-krea2-raw`
- `--width`, `--height`: fixed training resolution; must be divisible by 16
- `--training-steps`, `--steps`
- `--batch-size`
- `--learning-rate`, `--lr`
- `--rank`, `--alpha`
- `--max-text-length`
- `--scheduler-steps`
- `--caption-dropout`
- `--recipe`: named training recipe: `krea-fast-style`, `krea-cinematic-style`,
  or `klein-fast-style`
- `--lr-warmup-steps`, `--no-cosine-scheduler`, `--lr-min-factor`: Krea/Klein
  LR scheduler controls
- `--lite`: train only attention Q/V layers to reduce memory
- `--exclude-preview-images`
- `--checkpoint-interval`: save intermediate Klein LoRA adapters every N steps
- `--max-resolution`: preserve source aspect ratio up to a maximum side length
- `--low-ram`: use the Klein disk-backed latent cache
- `--no-compile`: disable compiled train-step graphs
- `--lora-target-preset`: exact Klein target preset, including `fal-klein-fast`
- `--lora-target-mode`: for FLUX.2 Klein, `suffix` or `transformer-linear-walk`
- `--quiet`

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

Run local text chat with the Gemma 4, Qwen3.6, LFM2, or Psi family.

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
- `--stream`
- `--thinking`
- `--stats`: includes Gemma4 MTP state and accept/draft counts when a Gemma4 model is used
- `--quiet`

Unless `--quiet` is set, diagnostics on stderr include the selected text
backend, for example native MLX/Metal for MLX models or llama.cpp/GGUF for GGUF
models.

Examples:

```bash
swift run mere.run text chat --prompt "What is classifier-free guidance?"
swift run mere.run text chat --model text-chat-q36-nano --prompt "Explain speculative decoding."
swift run mere.run text chat --model text-agent-ornith-9b --prompt "Write a compact Swift slugify helper."
swift run mere.run text chat --model text-chat-lfm25-a1b-8bit --prompt "Summarize LFM2 in one paragraph."
swift run mere.run text chat --stream --prompt "Write a short welcome message."
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

North Mini Code is managed as `text-code-north-mini` for local coding-agent
comparisons. It pulls the Unsloth GGUF quant and runs through the same native
Swift/llama.cpp code path as Qwen:

```bash
swift run mere.run model pull text-code-north-mini
swift run mere.run text code --model text-code-north-mini --prompt "Sketch a small Swift Result helper."
```

Ornith 35B is managed as `text-agent-ornith-35b` for larger local coding-agent
comparisons. It pulls DeepReinforce's Q4_K_M GGUF quant and runs through the
same native Swift/llama.cpp code path:

```bash
swift run mere.run model pull text-agent-ornith-35b
swift run mere.run text code --model text-agent-ornith-35b --prompt "Sketch a small Swift Result helper."
```

`text-agent-ornith-35b-mlx` is the native Swift/MLX lane for a locally converted
Ornith 1.0 35B Q4 directory. It is intentionally local-only until a converted
MLX snapshot is published:

```bash
swift run mere.run text chat --model text-agent-ornith-35b-mlx --prompt "Sketch a small Swift Result helper."
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

### `mere.run text anonymize`

Detect and redact PII with the native OpenAI Privacy Filter model.

```bash
swift run mere.run text anonymize "My name is Alice Smith and my email is alice@example.com"
swift run mere.run text anonymize --json --pretty "Phone: 555-1234"
cat notes.txt | swift run mere.run text anonymize --output redacted.txt
```

Key options:

- positional text arguments, or stdin when omitted
- `--model`
- `--max-tokens`
- `--replacement`: template supporting `{label}` and `{index}`
- `--json`
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
swift run mere.run vision caption ./cards/*.jpg \
  --output-dir ./captions \
  --prompt-file ./card-caption-prompt.txt \
  --focus "full card border" "printed title text" \
  --trigger-token cardstyle
```

- `--prompt`: caption instruction
- `--prompt-file`: read reusable caption instructions from a UTF-8 text file
- `--focus`: visible details the captioner should prioritize
- `--trigger-token`: prefix each saved caption with an exact LoRA trigger token

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
- `--threshold`: score cutoff, default `0.05`
- `--resolution`
- `--show-boxes`
- `--multimask`: emit up to three candidates per geometry-prompted object

Defaults:

- annotated image: `<image-stem>_segmented.<ext>`
- JSON metadata: `<image-stem>_segmented.json`

Notes:

- still-image runs accept text, box, and point prompts in the same invocation
- `--mask-output-dir` writes one PNG mask per exported detection candidate
- empty detection sets still produce annotated output plus JSON metadata

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
- `--threshold`: score cutoff, default `0.05`
- `--show-boxes`
- `--show-labels`

Defaults:

- annotated video: `<video-stem>_tracked.mp4`
- JSON metadata: `<video-stem>_tracked.json`

Notes:

- text prompts seed objects on `--init-frame`, then the native tracker reuses geometry prompts for later frames
- box and point prompts seed explicit tracked objects directly on the init frame
- `--mask-output-dir` writes per-frame mask PNGs under frame-named subdirectories
- empty prompt sets still produce an annotated video and JSON summary

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

- `--prompt`: one or more text prompts used to seed objects from the init frame
- `--camera`: camera device index
- `--duration-seconds`
- `--init-frame`: initial frame index used to seed tracking
- `--seed-search-frames`: additional frames to search when the init frame finds no objects
- `--output`: annotated video path
- `--json-output`: tracking metadata path
- `--threshold`: score cutoff, default `0.05`
- `--show-boxes`
- `--show-labels`

Notes:

- `track-live` currently records a camera clip first, then runs tracking over the recorded media
- live tracking searches a short warm-up window after the init frame so startup exposure or motion blur does not silently produce an unsegmented output
- live mode accepts text prompts only in the current implementation
- `--output` is required; `--json-output` is optional

### `mere.run vision ocr`

Extract text from one or more images.

```bash
swift run mere.run vision ocr <images...> [options]
```

Key options:

- `--backend`: `lighton`, `glm`, or `infinity`
- `--compare`: compare LightOn against the selected secondary backend; defaults
  to GLM when `--backend lighton`
- `--model`: managed id or path to the LightOn OCR root when using the LightOn backend
- `--glmocr-cli`, `--glm-config`
- `--infinity-runtime`: `native` or `external`; native uses the Swift Q35 runtime
- `--infinity-model`: native managed model id or local path; upstream model or
  server id when `--infinity-runtime external`
- `--infinity-parser-cli`: path to the Infinity-Parser2 `parser` executable for
  external runs
- `--infinity-backend`: external `vllm-server`, `vllm-engine`, or `transformers`
- `--infinity-api-url`, `--infinity-api-key`: external vLLM server settings
- `--infinity-task`: `doc2json`, `doc2md`, or `custom`
- `--infinity-output-format`: `md` or `json`
- `--max-tokens`
- `--quiet`

Examples:

```bash
swift run mere.run model pull vision-ocr-lighton
swift run mere.run model pull vision-ocr-infinity-flash
swift run mere.run model pull vision-ocr-infinity-pro-int8
swift run mere.run vision ocr ./page.png --backend lighton --model ~/Library/Application\ Support/MereRun/models/vision-ocr-lighton
swift run mere.run vision ocr ./page.png --backend glm
swift run mere.run vision ocr ./page.png --backend infinity --infinity-task doc2md
swift run mere.run vision ocr ./page.png --compare --backend infinity
swift run mere.run vision ocr ./page.png --backend infinity --infinity-runtime external --infinity-api-url http://127.0.0.1:8000/v1/chat/completions
```

### `mere.run music analyze`

Analyze an audio file with ACE-Step 5 Hz LM audio understanding and print JSON
metadata to stdout.

```bash
swift run mere.run music analyze "<audio>" [options]
```

Key options:

- `--model`, `-m`
- `--checkpoints-root`
- `--turbo-subdirectory`
- `--vae-subdirectory`
- `--lm-subdirectory`
- `--duration`: analyze the first N seconds instead of the full decoded input
- `--max-new-tokens`
- `--lm-temperature`
- `--lm-top-k`
- `--lm-top-p`
- `--include-raw-lm`
- `--include-audio-codes`
- `--quiet`

Examples:

```bash
swift run mere.run music analyze ./song.mp3 \
  --model music-acestep-xl-turbo-lm4b \
  --lm-subdirectory acestep-5Hz-lm-4B
swift run mere.run music analyze ./song.mp3 --duration 30 > ./song-analysis.json
```

### `mere.run music generate`

Generate music from a caption. The default model uses the native ACE-Step
pipeline with optional lyrics; Magenta RT2 models use the native Apple Silicon
Magenta bridge and ignore ACE-Step-only controls.

```bash
swift run mere.run music generate "<caption>" [options]
```

Key options:

- `--output`
- `--checkpoints-root`
- `--lyrics`
- `--lyrics-file`
- `--source-audio`: source song for ACE-Step cover conditioning; implies cover mode unless `--non-cover` is set
- `--analyze-source-audio`: use ACE-Step 5 Hz LM audio understanding to fill missing cover metadata from `--source-audio`
- `--reference-audio`: optional ACE-Step timbre reference audio file(s)
- `--duration`
- `--steps`
- `--use-lm`
- `--lm-subdirectory` (for example `acestep-5Hz-lm-4B` with `music-acestep-xl-turbo-lm4b`)
- `--text-subdirectory`
- `--seed`
- `--quiet`

Magenta RT2 options:

- `--style-conditioning`: `streaming` keeps the realtime C++ coarse style-token policy; `full` uses all MusicCoCa style tokens like the Python high-level generator
- `--temperature`
- `--top-k`
- `--cfg-musiccoca`
- `--cfg-notes`
- `--cfg-drums`
- `--drumless`
- `--unmask-width`
- `--seed-rotation`
- `--prefill-silence`
- `--prefill-duration`

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
swift run mere.run music generate \
  "dream-pop cover with soft vocals" \
  --source-audio ./song.mp3 \
  --analyze-source-audio \
  --lyrics-file ./cover-lyrics.txt \
  --audio-cover-strength 0.85 \
  --output ./cover.wav
swift run mere.run music generate \
  "ambient modular synths with brushed drums" \
  --model music-magenta-rt2-small \
  --duration 4 \
  --output ./magenta.wav
```

### `mere.run music realtime`

Run Magenta RealTime 2 generation. On macOS the command plays to the default
audio device by default; pass `--output` to capture a WAV file. Use `--no-play`
with `--output` and `--duration` for a headless smoke run.

```bash
swift run mere.run music realtime "<prompt>" [options]
```

Key options:

- `--model`: `music-magenta-rt2-small`, `music-magenta-rt2-base`, or a local Magenta RT2 root
- `--duration`
- `--output`
- `--play`, `--no-play`
- `--style-conditioning`: `streaming` keeps the realtime C++ coarse style-token policy; `full` uses all MusicCoCa style tokens like the Python high-level generator
- `--temperature`
- `--top-k`
- `--cfg-musiccoca`
- `--cfg-notes`
- `--cfg-drums`
- `--drumless`
- `--unmask-width`
- `--seed-rotation`
- `--prefill-silence`
- `--prefill-duration`
- `--interactive`: read live steering commands from stdin
- `--quiet`

Interactive commands:

- `prompt <text>`
- `style streaming|full`
- `temp <value>`, `topk <value>`
- `mc <value>`, `notes <value>`, `drums <value>`
- `noteon <0-131>`, `noteoff <0-131>`, `onset 0|1`
- `drumless on|off`, `unmask <value>`, `seed <value>`
- `reset`, `quit`, `help`

Examples:

```bash
swift run mere.run music realtime \
  "ambient pads with sub bass" \
  --model music-magenta-rt2-small \
  --duration 4 \
  --output ./live.wav

swift run mere.run music realtime \
  "drumless glassy arpeggios" \
  --model music-magenta-rt2-small \
  --duration 2 \
  --output ./smoke.wav \
  --no-play

swift run mere.run music realtime \
  "ambient modular synths" \
  --model music-magenta-rt2-small \
  --duration 30 \
  --interactive
```

### `mere.run sfx generate`

Generate a mono WAV sound effect from a text prompt. The default model uses the
native Sony Research Woosh DFlow path; the original Woosh Flow checkpoint is
also available as `sfx-woosh-flow`.

```bash
swift run mere.run sfx generate "<prompt>" [options]
```

Key options:

- `--model`: `sfx-woosh-dflow`, `sfx-woosh-flow`, or a local Woosh checkpoints root
- `--output`
- `--duration`
- `--steps`
- `--cfg`
- `--renoise`
- `--seed`
- `--quiet`

Examples:

```bash
swift run mere.run sfx generate \
  "metal wrench dropping onto concrete, bright clang and brief ring" \
  --model sfx-woosh-dflow \
  --duration 5 \
  --steps 4 \
  --cfg 4.5 \
  --output ./wrench-clang.wav
```

### `mere.run sfx ae`

Encode audio into normalized Woosh-AE latents or decode those latents back to a
mono WAV.

```bash
swift run mere.run sfx ae encode ./input.wav -o ./input-latents.npy
swift run mere.run sfx ae decode ./input-latents.npy -o ./input-roundtrip.wav
```

### `mere.run sfx condition text`

Export Woosh text-conditioning tensors for a prompt. The output safetensors file
contains `embeddings` and `mask` arrays.

```bash
swift run mere.run sfx condition text "glass breaking" -o ./glass-condition.safetensors
```

### `mere.run sfx clap score`

Score a text prompt against an audio file with the native Woosh-CLAP text and
PaSST audio towers. The command prints JSON to stdout.

```bash
swift run mere.run sfx clap score "glass breaking" ./glass.wav
```

### `mere.run sfx video generate`

Generate a mono WAV sound effect from a raw video file or precomputed
Synchformer video features. `.npy` feature inputs must have shape
`[frames, 768]` or `[1, frames, 768]`.

```bash
swift run mere.run model pull sfx-woosh-synchformer
swift run mere.run sfx video generate \
  "footsteps echoing in a hallway" \
  ./silent-hallway.mp4 \
  --model sfx-woosh-dvflow-8s \
  --duration 8 \
  --output ./hallway-footsteps.wav
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
- `--duration`
- `--fps`
- `--seed`
- `--image`
- `--image-strength`
- `--end-image`
- `--end-image-strength`
- `--quiet`

Environment:

- `MERERUN_VIDEO_LTX_MODEL_ROOT`
- `MERERUN_VIDEO_LTX_TEXT_ENCODER_ROOT` for an external
  `mlx-community/gemma-3-12b-it-4bit` checkout used by `video-ltx23-av-mlx`

For `--variant unified-av`, keep `--fps 24` unless you are deliberately making
a retimed clip. LTX 2.3 unified AV is trained around 24 fps; using 8 fps can
make generated motion look slow while audio remains normal. Use `--duration`
for clip length so the CLI can choose the nearest legal `8n+1` frame count.
Use the default `distilled` lane for faster video-only drafts. Use
`--variant unified-av --model video-ltx23-av-mlx` for the current high-quality
synchronized audio/video lane.

Examples:

```bash
swift run mere.run video generate \
  "a cinematic drone flythrough over snowy mountains" \
  --variant distilled \
  --model video-ltx23-av-mlx \
  --num-frames 65

swift run mere.run video generate \
  "two actors talking beside a window while a restrained orchestral score and distant city sirens play underneath" \
  --variant unified-av \
  --model video-ltx23-av-mlx \
  --duration 15 \
  --fps 24 \
  --output ./dialogue-score-sfx.mp4

swift run mere.run video generate \
  "a car drives from a bright morning street into a warm sunset road, smooth forward motion" \
  --variant distilled \
  --model video-ltx23-av-mlx \
  --image ./car-start.png \
  --end-image ./car-end.png \
  --num-frames 65 \
  --output ./car-start-to-end.mp4
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

List all managed model IDs, whether they are installed, and their resolved
payload size. Sizes follow symlinked payload directories in the model store.

```bash
swift run mere.run model list
```

### `mere.run status`

Show a quick local snapshot: whether the API server answers, which model it
reports as loaded through `/v1/models`, the active model-store path/source, and
which managed models are installed in that store.
When the server exposes the native runtime pool, JSON status also includes pool
entries, active request counts, request admission queue depth, the memory
snapshot, runtime capability flags, aggregate cache stats, per-model prefix KV
cache stats, per-model decode batching stats when enabled, aggregate benchmark
stats from completed native chat requests, and the runtime settings path.

```bash
swift run mere.run status
swift run mere.run status --host 127.0.0.1 --port 11434
swift run mere.run status --json
```

Useful options:

- `--host`: local API host to check, default `127.0.0.1`
- `--port`: local API port to check, default `8080`
- `--api-key`: bearer token for `/v1/models`, also read from `MERERUN_API_KEY`
- `--timeout-seconds`: network probe timeout
- `--json`: emit a structured snapshot for scripts and agents

### `mere.run model runtime get`

Read typed per-model API runtime settings from the active model store.

```bash
swift run mere.run model runtime get text-chat-gemma4
swift run mere.run model runtime get text-chat-gemma4 --json
```

Settings are stored at
`<active model store>/.mere-run/runtime-model-settings.json` and follow
`--models-root` / `MERERUN_MODELS_DIR`.

### `mere.run model runtime set`

Update typed API serving defaults for a managed API-capable model.

```bash
swift run mere.run model runtime set text-chat-gemma4 \
  --alias chat-default \
  --pinned \
  --ttl-seconds 3600 \
  --max-context-tokens 8192 \
  --max-tokens 1024 \
  --temperature 0.6 \
  --top-p 0.9 \
  --kv-cache-mode auto
```

Use the matching `--clear-*` flags to remove optional values. Engine overrides
are validated against the curated catalog. Gemma4 accepts `--kv-cache-mode`
values of `default`, `polar2`, and `auto`; non-Gemma4 models reject PolarKV
runtime modes. `--ttl-seconds` unloads idle loaded models during the runtime
pool's opportunistic eviction passes, while `--pinned` protects a model from
automatic TTL/LRU eviction without blocking explicit unload. Memory-pressure LRU
uses the API server's `--memory-guard` tier. The guard derives soft/hard
ceilings from process resident memory, host memory headroom, and a tier reserve;
elevated pressure evicts the least-recently-used idle unpinned model, while
critical pressure evicts every idle unpinned model.

### `mere.run model pull`

Download a managed Hugging Face snapshot into the local model store. The command checks
the model capability catalog and available disk space before downloading so unsupported
machines do not pull models they cannot run and tight disks fail with a useful cache path.

```bash
swift run mere.run model pull image-zimage-nano
swift run mere.run model pull --all
```

Use `--allow-unsupported` only when you intentionally accept the runtime risk.

### `mere.run model benchmark gemma4-kv`

Run a fixed-token real-checkpoint Gemma4 KV cache comparison. The command runs
the selected Gemma4 model twice in one process: default Gemma4 KV settings first,
then the runtime `polar2` mode, which uses model-default prefill with packed
`polar` 2-bit KV from token 0 for decode. It
disables EOS stopping so both variants decode exactly `--decode-tokens`, and
reports TTFT, prefill tok/s, KV conversion time, decode tok/s, end-to-end tok/s,
and process resident memory before and after each variant.

```bash
swift run mere.run model benchmark gemma4-kv \
  --model text-chat-gemma4-turbo \
  --decode-tokens 48 \
  --json
```

Use `--prompt`, `--prompt-file`, or `--prompt-repeat` to control prompt length.
Use `--prompt-repeat-values` and `--decode-token-values` with comma-separated
values to run a prompt-size/decode-length matrix for promotion evidence.
The default fixture prompt is deterministic and intended for local A/B
comparisons, not model-quality evaluation.

### `mere.run model benchmark gemma4-mtp`

Run a fixed-token real-checkpoint Gemma4 MTP comparison. The command runs the
selected Gemma4 model twice in one process: `baseline` with
`MERERUN_GEMMA4_MTP=0`, then `mtp` with `MERERUN_GEMMA4_MTP=1`. It defaults to
the practical 4-bit checkpoint and disables EOS stopping so both variants decode
exactly `--decode-tokens`.

```bash
swift run mere.run model benchmark gemma4-mtp \
  --model text-chat-gemma4-12b-4bit \
  --decode-tokens 48 \
  --json
```

Output includes prompt tokens, generated tokens, load time, prefill time, decode
time, TTFT, prefill tok/s, decode tok/s, end-to-end tok/s, process resident
memory, decode speedup, end-to-end speedup, and MTP counters for rounds, drafted
tokens, accepted tokens, rejected tokens, acceptance rate, and accepted tokens
per round.

Use `--prompt`, `--prompt-file`, or `--prompt-repeat` to control prompt length.
Use `--prompt-repeat-values` and `--decode-token-values` with comma-separated
values to run a prompt-size/decode-length matrix. `--mtp-block-size` and
`--mtp-min-prompt-tokens` are optional benchmark overrides for draft block size
and the activation threshold; leaving them unset uses the runtime policy
defaults. The default fixture is deterministic and intended for throughput
comparison, not model-quality evaluation.

### `mere.run model benchmark q36-mtp`

Run a requested-token real-checkpoint Qwen3.6 MTP comparison. The command runs
`text-chat-q36-nano` with three policies:

- `baseline`: MTP disabled with `MERERUN_Q35_MTP_SPECULATION=0`.
- `adaptive`: production long-context policy.
- `forced`: MTP enabled with `MERERUN_Q35_MTP_SPECULATION=1` and a configurable
  forced threshold.

```bash
swift run mere.run model benchmark q36-mtp \
  --prompt-repeat-values 8,80,150 \
  --temperature-values 0,0.7 \
  --decode-tokens 32 \
  --json
```

The runtime may still stop on EOS before `--decode-tokens`. Output includes
prompt tokens, generated tokens, load time, prefill time, decode time, TTFT,
prefill tok/s, decode tok/s, end-to-end tok/s, process resident memory, and
adaptive/forced speedups versus baseline. Greedy forced MTP uses the native
block verifier; non-greedy forced MTP stays on the exact probabilistic
speculative path. Use `--mtp-block-size` to test a different greedy draft block
cap and `--forced-mtp-min-prompt-tokens` to adjust the forced policy threshold.

### `mere.run model benchmark api-workload`

Replay streaming OpenAI-compatible chat requests against an already-running
`mere.run api serve` process. This is the serving-path benchmark for request
admission, prefix KV reuse, opt-in decode batching, and the eventual SSD KV
decision.

```bash
MERERUN_GEMMA4_PREFIX_KV_CACHE=0 \
swift run mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-turbo \
  --max-active-requests 1

swift run mere.run model benchmark api-workload \
  --model text-chat-gemma4-turbo \
  --json
```

To test the measured-work path, rerun the same workload with default prefix
reuse and opt-in batching enabled, then compare TTFT, wall-clock throughput, and
runtime status deltas:

```bash
MERERUN_GEMMA4_CONTINUOUS_BATCHING=1 \
swift run mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-turbo \
  --max-active-requests 4

swift run mere.run model benchmark api-workload \
  --model text-chat-gemma4-turbo \
  --concurrency 4 \
  --json
```

The built-in workload uses one stable system prefix and varied final user turns.
Output reports per-request TTFT, total latency, streamed chunk count, wall-clock
requests/sec, prefix KV hits/misses, reused prefix tokens, decode batched steps,
single decode steps, and whether SSD KV cache is available. Use
`--workload-file` to replay JSONL rows with either `{ "id", "user" }` or
`{ "id", "messages" }`.

### `mere.run model benchmark code`

Run a small real coding-eval slice against installed local coding models. The
default suite is `humaneval-slice`, a three-task HumanEval subset covering
`HumanEval/0`, `HumanEval/3`, and `HumanEval/8`. The default model comparison is
`text-agent-ornith-9b`, `text-code-north-mini`, and `text-code-qwen3`.

```bash
swift run mere.run model benchmark code \
  --allow-code-execution \
  --json
```

The command prompts each model once per task, combines the generated Python with
the task tests, and runs that candidate in a sandboxed `python3` subprocess with
a per-candidate timeout. Because scoring executes generated code locally, pass
`--allow-code-execution` for real runs or `--dry-run` to inspect the plan.
The default `--sandbox auto` uses `sandbox-exec` on macOS and `bubblewrap` on
Linux when available. Use `--sandbox none` only for a trusted local smoke where
timeout and temporary-directory hygiene are enough. The default generation cap is
`--max-tokens 1024`, and capped cases are reported as `reachedMaxTokens` in JSON
or `capped=true` in text output. Reasoning blocks are preserved separately as
`reasoningCharacters`/`reasoning_chars` and
`incompleteReasoning`/`reasoning_incomplete`, while only visible code is
executed. `reasoning_reopened=true` flags a second generated reasoning block,
which usually indicates a loop or phase restart. Use `--models` and `--tasks`
to narrow the slice while iterating. Use `--models text-agent-ornith-35b` for
the larger Ornith GGUF eval target.

For a larger slice from the official HumanEval data, download and decompress
`HumanEval.jsonl.gz`, then pass the JSONL file with `--humaneval-file`:

```bash
curl -L https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz \
  -o /tmp/HumanEval.jsonl.gz
gunzip -c /tmp/HumanEval.jsonl.gz > /tmp/HumanEval.jsonl
swift run mere.run model benchmark code \
  --humaneval-file /tmp/HumanEval.jsonl \
  --tasks HumanEval/0,HumanEval/1,HumanEval/2,HumanEval/3,HumanEval/4 \
  --allow-code-execution
```

### `mere.run model benchmark vlm`

Run a tiny synthetic VLM smoke, or use `lmms-eval` to compare an installed
vision-chat model against existing multimodal datasets.

```bash
swift run mere.run model benchmark vlm --json
```

The default synthetic suite compares `vision-chat-gemma4-12b` with the existing
`vision-inspect-qwen3-vl-2b` backend on deterministic color, location, and
counting fixtures.

For existing datasets, install or check out `lmms-eval`, then start with a
dry-run:

```bash
swift run mere.run model benchmark vlm \
  --dataset mathvista-testmini \
  --limit 16 \
  --lmms-eval-root ~/src/lmms-eval \
  --dry-run \
  --json
```

Preset dataset flags map to upstream task names:

| Dataset flag | lmms-eval task |
| --- | --- |
| `mathvista-testmini` | `mathvista_testmini` |
| `mmmu-val` | `mmmu_val` |
| `chartqa` | `chartqa` |
| `docvqa-val` | `docvqa_val` |
| `mme` | `mme` |

Use `--lmms-tasks` for raw upstream task names, `--external-endpoint --base-url`
for an already-running OpenAI-compatible server, and omit `--dry-run` to let the
command start a local `mere.run api serve` process per requested model.

### `mere.run model capabilities`

Show this machine's supported models, recommended setup package, chat winners
by RAM band, and a short summary of what each model does.

```bash
swift run mere.run model capabilities
swift run mere.run model capabilities --all
```

### `mere.run model info`

Inspect a canonical model ID or a local model root. The Storage section reports
the layout, resolved payload size, wrapper size when different, and symlink
counts.

```bash
swift run mere.run model info image-zimage-nano
swift run mere.run model info /path/to/model/root --components
swift run mere.run model info text-chat-gemma4
```

### `mere.run model remove`

Delete an installed managed model by canonical ID.

```bash
swift run mere.run model remove image-zimage-nano
swift run mere.run model remove image-zimage-nano --force
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
- `POST /v1/embeddings`
- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `POST /v1/audio/speech`
- `POST /v1/audio/transcriptions`
- `GET /runtime/status`
- `POST /runtime/models/{id}/load`
- `POST /runtime/models/{id}/unload`
- `GET/PATCH /runtime/models/{id}/settings`

Security defaults:

- loopback binds are local-first and do not require auth
- non-loopback binds require `--api-key` or `MERERUN_API_KEY`
- `POST /v1/chat/completions`, `POST /v1/embeddings`,
  `POST /v1/images/generations`, and `POST /v1/audio/speech` require
  `Content-Type: application/json`; `POST /v1/images/edits` and
  `POST /v1/audio/transcriptions` require `multipart/form-data`
- `--rate-limit-per-minute` applies basic request throttling to the
  OpenAI-compatible routes
- `--max-active-requests` controls fair FIFO chat admission; the default `1`
  preserves serialized local inference while exposing queue depth in status;
  queued client cancellations are removed from the FIFO instead of running later
- `--memory-guard` controls runtime memory pressure behavior. Accepted values
  are `off`, `safe`, `balanced`, `aggressive`, and `custom`; `custom` also
  requires `--memory-guard-custom-ceiling-gb`.
- elevated or critical memory pressure pauses extra concurrent admissions while
  letting one request run so the server can make progress
- Gemma4 and Qwen-family chat use chunked prefill checkpoints for long prompts.
- Gemma4 uses in-memory prefix KV reuse by default in `api serve`; set
  `MERERUN_GEMMA4_PREFIX_KV_CACHE=0` for a baseline. Runtime status reports
  entries, hits, and reused tokens when a Gemma4 model is loaded, including
  semantic chat-prefix checkpoints before the final message when token prefixes
  match exactly.
- Qwen-family chat uses text-only in-memory prefix KV reuse by default in
  `api serve`; set `MERERUN_Q35_PREFIX_KV_CACHE=0` for a baseline. Vision
  prompts are excluded from reuse, and text-only requests use the same semantic
  chat-prefix checkpoints as Gemma4.
- Managed Gemma4 12B text and vision pulls install a companion MTP assistant.
  When `MERERUN_GEMMA4_MTP` is not disabled, greedy serial decode can use that
  assistant on the decode tail after prefill; sampled requests, continuous
  batching, raw local model paths, and prefix-KV seeded requests use baseline
  decode
- Gemma4 and Qwen-family chat can opt into decode batching with
  `MERERUN_GEMMA4_CONTINUOUS_BATCHING=1` or
  `MERERUN_Q35_CONTINUOUS_BATCHING=1`; use `--max-active-requests` above `1` to
  allow overlap, and status reports actual batched decode steps and max observed
  batch size; Gemma4 full-attention rows stay same-position because that path
  still uses scalar RoPE/cache offsets, while Qwen-family full-attention rows use
  row-offset-aware ragged KV caches and Qwen-family linear rows use typed recurrent
  state so compatible Qwen-family rows can batch across decode positions; the scheduler
  services the earliest decode position first by batching compatible rows there
  or advancing one lower-offset row until it can join a compatible batch
- Gemma4 can opt into experimental packed PolarKV with
  `--kv-quant-scheme polar --kv-bits 2`; use it for memory-pressure and
  long-context synthetic decode testing. It is not the default until checkpoint
  benchmarks prove the end-to-end model path.
- Per-model runtime settings can also set `kvCacheMode` to `default`, `polar2`,
  or `auto` for Gemma4. `auto` keeps the default KV path below 1024 prompt
  tokens and switches to decode-deferred packed PolarKV at or above that
  threshold.
- `/runtime/status` and `mere.run status` aggregate prefix hits, reused tokens,
  batched decode steps, completed chat requests, generated tokens, and average
  load/prefill/decode timings across loaded models under `cacheStats` and
  `benchmarkStats`
- generation parameters are bounded before execution; for example, `max_tokens` must fit the configured context size, and native chat engines receive that same context cap for prompt truncation
- LoRA adapters for the API server are selected by the operator with `--lora`; request bodies cannot provide local LoRA paths

Engine values:

- `text-code`
- `text-chat-klein`
- `text-chat-gemma4`
- `text-chat-q36`
- `text-chat-lfm2`
- `text-chat-deepseek-v4-flash`

OpenAI chat compatibility:

- DS4 raw-proxies the full `/v1/chat/completions` body to `ds4-server`.
- Native engines decode the common OpenAI Chat request shape and reject
  unsupported high-impact fields with `invalid_request_error`.
- `max_completion_tokens`, `developer` messages, function tools, image content
  parts, structured JSON mode, `stop` sequences, and streaming usage are
  capability-gated by engine.
- `tool_choice` accepts `none`, `auto`, `required`, and specific function
  choices by narrowing the advertised tool list to the named function.

OpenAI embeddings compatibility:

- `POST /v1/embeddings` serves native `text-embed-qwen3-0.6b` embeddings.
- `input` may be a string or array of strings.
- `encoding_format` may be omitted or set to `float`; base64 encoding and
  dimension overrides are rejected.

OpenAI image/audio compatibility:

- `POST /v1/images/generations` serves native image generation models such as
  `image-zimage-nano`; it supports `prompt`, `size`, `n=1`, `response_format`
  `b64_json` or `url`, and local extensions such as `seed`.
- `POST /v1/images/edits` accepts multipart `image` uploads, Open WebUI-style
  `image[]` repeated uploads, an optional `mask`, and an edit `prompt`; it uses
  the same image runtime with input-image conditioning. Masks are accepted for
  client compatibility; current native edit models use whole-image conditioning.
- `POST /v1/audio/speech` serves `speech-tts-qwen3-nano` and returns WAV by
  default, with `mp3`, `opus`, `aac`, and `flac` available when `ffmpeg` is
  installed. OpenAI model names such as `tts-1` map to the local default.
- `POST /v1/audio/transcriptions` accepts multipart uploads for
  `speech-asr-parakeet` or `speech-asr-qwen3`, with `json`, `text`,
  `verbose_json`, `srt`, and `vtt` response formats. OpenAI model names such as
  `whisper-1` map to the local default.

Examples:

```bash
swift run mere.run api serve
swift run mere.run api serve --engine text-chat-gemma4
swift run mere.run api serve --engine text-chat-lfm2
swift run mere.run api serve --engine text-code --model ./Qwen3-Coder-Next-Q4_K_M.gguf
swift run mere.run api serve --host 0.0.0.0 --port 11434 --api-key "$MERERUN_API_KEY" --rate-limit-per-minute 120 --max-active-requests 1
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  --data '{"model":"text-embed-qwen3-0.6b","input":"local RAG"}'
curl http://127.0.0.1:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  --data '{"model":"image-zimage-nano","prompt":"a compact local AI workstation","size":"1024x1024"}'
curl http://127.0.0.1:8080/v1/images/edits \
  -F model=qwen-image-edit \
  -F prompt="make the workstation dusk-lit while preserving the layout" \
  -F image=@input.png
curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  --output speech.wav \
  --data '{"model":"speech-tts-qwen3-nano","input":"mere.run is online","voice":"nova","response_format":"wav"}'
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=speech-asr-parakeet \
  -F file=@speech.wav
```

After starting a server, run `swift run mere.run status` from another terminal
to confirm `/health`, `/v1/models`, the runtime pool, and the served model.

### `mere.run setup`

Choose the public onboarding path. The default interactive command offers the
local Mere agent powered by Pi, a bring-your-own-agent handoff prompt, or manual
commands.

```bash
swift run mere.run setup
swift run mere.run setup --mode agent --agent-model small --dry-run
swift run mere.run setup --mode agent --agent-model tier --install --start
swift run mere.run setup --mode byoa
swift run mere.run setup --mode manual
```

Agent model choices:

- `small`: `text-agent-qwen35-9b`, a Qwen3.5 9B Q4 GGUF setup agent for 16 GB machines
- `tier`: the best supported local tier for this machine, currently 9B, Qwen3.6 nano, Qwen3-Coder Next, or DeepSeek V4 Flash on 96 GB+ machines
- `premier`: `text-agent-deepseek-v4-flash`, the preferred managed 96 GB+ setup-agent tier served by the bundled DS4 engine

North Mini Code (`text-code-north-mini`) is available as a managed native GGUF
coding model. It is pullable through `model pull`, can be served with
`api serve --engine text-code --model text-code-north-mini`, and can be started
through the Pi-backed `agent start` path like other `text-code` models once the
installed llama.cpp runtime supports the `cohere2moe` architecture.

Ornith (`text-agent-ornith-9b`) is available as an experimental native
MLX/OptiQ coding-agent model. It uses the Qwen-family runtime, so serve it with
`api serve --engine text-chat-q36 --model text-agent-ornith-9b`.
The local converted Ornith 35B MLX target (`text-agent-ornith-35b-mlx`) uses
the same native Qwen-family serving engine when installed.
The larger Ornith 35B GGUF target (`text-agent-ornith-35b`) is also available
for explicit evals and runs through:

```bash
swift run mere.run api serve --engine text-chat-q36 --model text-agent-ornith-35b-mlx
swift run mere.run api serve --engine text-code --model text-agent-ornith-35b
```

BYOA prints a ready-to-paste Claude/Codex prompt. Manual mode prints the
commands for capabilities, model pulls, serving, and optional Pi installation.
Pi auto-install uses the published macOS release assets; on Linux, put a `pi`
binary on `PATH` or pass `--pi-path` and the agent runs in the current terminal.

### `mere.run agent onboard`

Lower-level agent plumbing used by `mere.run setup`. Print a guided setup
summary for the current machine. Optional flags can pull the
recommended supported model package, install Pi, and write a Pi provider
extension that points at `mere.run api serve`.

```bash
swift run mere.run agent onboard
swift run mere.run agent onboard --pull-recommended
swift run mere.run agent onboard --install-pi --configure-pi
swift run mere.run agent onboard --configure-pi --model text-agent-deepseek-v4-flash
swift run mere.run agent onboard --configure-pi --model text-code-north-mini --port 8080
swift run mere.run agent onboard --configure-pi --model text-agent-ornith-9b --port 8080
```

### `mere.run agent install-pi`

Install the latest `badlogic/pi-mono` release asset for the current macOS
architecture into the mere.run application-support directory.

```bash
swift run mere.run agent install-pi
```

### `mere.run agent start`

Start a local API server for a selected managed agent model and launch Pi
against the `mere-run` provider. GGUF code models use `--engine text-code`,
Qwen3.6 uses `--engine text-chat-q36`, and DeepSeek V4 Flash uses the DS4-backed
`--engine text-chat-deepseek-v4-flash`. If `--model` is omitted, `agent start`
uses the best installed startable setup agent first, then a valid persisted Pi
provider model, then the current machine's startable hardware tier. On 96 GB+
Apple Silicon Macs, DeepSeek V4 Flash is the preferred setup-agent tier; smaller
Qwen models are alternatives, not upgrades.

```bash
swift run mere.run model pull text-agent-deepseek-v4-flash
swift run mere.run agent install-pi
swift run mere.run agent start --model text-agent-deepseek-v4-flash
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
./scripts/e2e_smoke.sh --installed
```
