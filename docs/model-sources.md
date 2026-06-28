# Model Sources

This repo supports two model paths:

1. Hugging Face snapshots pulled into the local mere.run model store with `mere.run model pull`
2. Explicit local paths passed to commands with `--model`, `--model-root`, or equivalent command-specific options

The public repo no longer supports private model archives, R2 credentials, or a
packaged central model host.

The canonical local model store is:

```text
~/Library/Application Support/MereRun/models
```

Override that with `MERERUN_MODELS_DIR` or `--models-root`.

## Canonical Managed Model IDs

This is the authoritative public catalog list. It is kept in sync with
`ManagedModelCatalog.allSpecs`, and the test suite fails if the table drifts
from the runtime catalog used by `mere.run model list`,
`mere.run model capabilities --all`, and `mere.run model pull`.

<!-- managed-model-catalog:start -->
| Catalog category | Model ID |
| --- | --- |
| `image` | `image-klein-nano` |
| `image` | `image-klein-max` |
| `image` | `image-klein-9b` |
| `image` | `image-klein-base` |
| `image` | `image-klein-base-9b` |
| `image` | `image-klein-shared` |
| `image` | `image-bonsai-binary` |
| `image` | `image-bonsai-ternary` |
| `image` | `image-zimage-nano` |
| `image` | `image-zimage-max` |
| `image` | `image-zimage-base` |
| `image` | `image-hidream-o1` |
| `image` | `image-hidream-o1-dev` |
| `image` | `image-krea2-raw` |
| `image` | `image-krea2-turbo` |
| `image` | `image-ideogram4-sdnq-uint4` |
| `text-chat` | `text-chat-mebot` |
| `text-chat` | `text-chat-psi-agent` |
| `text-chat` | `text-chat-gemma4` |
| `text-chat` | `text-chat-gemma4-turbo` |
| `text-chat` | `text-chat-gemma4-12b` |
| `text-chat` | `text-chat-gemma4-12b-4bit` |
| `vision-chat` | `vision-chat-gemma4-12b` |
| `text-chat` | `text-chat-gemma4-nano` |
| `text-chat` | `text-chat-gemma4-max` |
| `text-chat` | `text-chat-q36-nano` |
| `text-code` | `text-agent-ornith-9b` |
| `text-code` | `text-agent-qwen35-9b` |
| `text-code` | `text-code-north-mini` |
| `text-code` | `text-agent-ornith-35b` |
| `text-chat` | `text-chat-q36-nano-gguf` |
| `text-chat` | `text-agent-deepseek-v4-flash` |
| `text-chat` | `text-chat-lfm25-a1b-8bit` |
| `speech-tts` | `speech-tts-qwen3-nano` |
| `speech-tts` | `speech-tts-qwen3-customvoice` |
| `speech-asr` | `speech-asr-qwen3` |
| `speech-asr` | `speech-asr-parakeet` |
| `text-code` | `text-code-qwen3` |
| `text-embed` | `text-embed-qwen3-0.6b` |
| `text-anonymize` | `text-anonymize-privacy-filter` |
| `vision-ocr` | `vision-ocr-infinity-flash` |
| `vision-ocr` | `vision-ocr-infinity-pro` |
| `vision-ocr` | `vision-ocr-infinity-pro-int8` |
| `vision-ocr` | `vision-ocr-lighton` |
| `vision-segment` | `vision-segment-sam31` |
| `vision-ground` | `vision-ground-falcon-perception` |
| `music` | `music-acestep` |
| `music` | `music-acestep-xl-turbo` |
| `music` | `music-acestep-xl-turbo-lm4b` |
| `music` | `music-magenta-rt2-small` |
| `music` | `music-magenta-rt2-base` |
| `sfx` | `sfx-woosh-dflow` |
| `sfx` | `sfx-woosh-flow` |
| `sfx` | `sfx-woosh-clap` |
| `sfx` | `sfx-woosh-synchformer` |
| `sfx` | `sfx-woosh-vflow-8s` |
| `sfx` | `sfx-woosh-dvflow-8s` |
| `video` | `video-ltx-av` |
| `video` | `video-ltx23-av-mlx` |
<!-- managed-model-catalog:end -->

Most catalog IDs have managed Hugging Face sources and can be installed with
`mere.run model pull`. A small number of legacy/local catalog IDs remain so
existing installs and explicit local paths keep working:

- `image-klein-shared`
- `text-chat-mebot`
- `text-chat-psi-agent`

`image-klein-shared` is an internal shared-component install shape, and the
text-chat IDs listed here remain local-path-only until they have public Hugging
Face sources.

`text-agent-qwen35-9b` is the low-memory setup-agent model. It uses the public
Hugging Face source `unsloth/Qwen3.5-9B-GGUF` and selects
`Qwen3.5-9B-Q4_K_M.gguf`.

`text-code-north-mini` installs the Unsloth GGUF quant of Cohere Labs' North
Mini Code 1.0 at the pinned catalog revision. North Mini Code is a 30B total /
3B active coding MoE with a 256K advertised context window; mere.run uses the
`North-Mini-Code-1.0-UD-Q4_K_M.gguf` file so the model runs through the same
native Swift/llama.cpp `text code` path as the existing Qwen coder. It requires
a llama.cpp runtime with `cohere2moe` architecture support.

`text-agent-ornith-9b` installs the public
`sahilchachra/ornith-1.0-9b-optiq-5bpw-mlx` snapshot at the pinned catalog
revision. Ornith is a DeepReinforce agentic coding model with Qwen3.5 text
architecture metadata; mere.run treats this MLX OptiQ quant as a native
Qwen-family runtime target for `chat`, `api serve`, and setup-agent experiments.

`text-agent-ornith-35b` installs DeepReinforce's public
`deepreinforce-ai/Ornith-1.0-35B-GGUF` Q4_K_M file at the pinned catalog
revision. It runs through the native Swift/llama.cpp `text code` path for
larger Ornith coding-agent comparisons and uses a 32K runtime context by
default to keep local evals predictable.

`text-agent-deepseek-v4-flash` is the preferred managed setup-agent tier on
96 GB+ Apple Silicon Macs. Smaller Qwen setup agents are lower-memory
alternatives, not upgrades from DeepSeek V4 Flash.

`text-chat-q36-nano` uses the public `mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit`
snapshot. That Hugging Face repo includes an MTP head (`mtp.safetensors`) for
OptiQ serving; mere.run loads that draft head when present, but only uses it for
adaptive speculative decode when the effective prompt and context window are
long enough. Short-context requests decode with the main chat weights. The dense
`mlx-community/Qwen3.6-27B-OptiQ-4bit` MTP quant also exists on Hugging Face,
but is not a managed native target until the dense Qwen3.6 layer path is
implemented.

`text-chat-q36-nano-gguf` installs the Unsloth
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` quant. It is the llama.cpp/GGUF companion to
the Apple Silicon MLX `text-chat-q36-nano` path and is the default chat model
for Linux CUDA hosts.

`text-chat-gemma4-12b` and `vision-chat-gemma4-12b` share Google's dense Gemma
4 12B-it checkpoint; the text id uses the native chat path, while the vision id
enables OpenAI image content parts through `api serve`. Pulling either managed
12B id also pulls the companion `google/gemma-4-12B-it-assistant` MTP drafter.
The native Swift runtime uses that assistant only for greedy decode-tail
speculation after text, image, or audio prefill has produced target hidden state
and shared KV; raw local model paths and sampled generations fall back to
baseline decode. `text-chat-gemma4` is the dense bf16 Gemma 4 31B alias and is
gated for larger machines. On 32 GB Apple Silicon Macs, use
`text-chat-gemma4-turbo`, which installs the MLX NVFP4 Gemma 4 26B-A4B-it MoE
snapshot and runs through the native Swift Gemma runtime. On smaller supported
machines, `text-chat-gemma4-12b-4bit` installs the
`mlx-community/gemma-4-12B-it-4bit` snapshot as the compact managed Gemma 12B
chat tier.

`text-chat-lfm25-a1b-8bit` uses the public
`LiquidAI/LFM2.5-8B-A1B-MLX-8bit` snapshot at the pinned catalog revision. It is
a text-only MLX 8-bit directory-root model with `config.json`,
`tokenizer.json`, `tokenizer_config.json`, and sharded `*.safetensors` weights.
mere.run runs it through the native Swift LFM2 runtime; no Python bridge is used.

Useful environment variables for that path:

- `MERERUN_HUB_CACHE`: override the native Hugging Face snapshot cache path

`image-klein-9b` installs the ungated `mlx-community/FLUX.2-klein-9B` mirror for
larger distilled Klein generation and reference-image workflows. It is distinct
from `image-klein-base-9b`, which pulls the undistilled Base 9B transformer plus
shared 9B components for higher-capacity LoRA training and research workflows.

`image-bonsai-binary` and `image-bonsai-ternary` map to PrismML Apple Silicon
Bonsai Image snapshots:

- `prism-ml/bonsai-image-binary-4B-mlx-1bit`
- `prism-ml/bonsai-image-ternary-4B-mlx-2bit`

The snapshot uses a FLUX.2 Klein transformer, but its component names are
upstream-specific. Managed or local roots are expected to contain:

- `manifest.json`
- `tokenizer/tokenizer_config.json`
- `text_encoder-mlx-4bit/config.json`
- `text_encoder-mlx-4bit/model.safetensors` or `model.safetensors.index.json`
- `transformer-packed-mflux/config.json`
- `transformer-packed-mflux/quantization_config.json`
- `transformer-packed-mflux/diffusion_pytorch_model.safetensors`
- `vae/config.json`
- `vae/diffusion_pytorch_model.safetensors`
- `scheduler/scheduler_config.json`

The binary manifest records the transformer as 1-bit g128 Prism packed affine
weights; the ternary manifest records 2-bit g128 MLX packed affine ternary
weights. Both keep the text encoder in the upstream 4-bit MLX layout and run
generation through the native Swift FLUX.2 Klein pipeline with four steps, CFG
1.0, and sigma shift 3.0 by default. The binary runtime path uses a native
Swift/Metal packed 1-bit affine matmul kernel, with a dequantized MLX fallback
for non-GPU or unsupported shapes while upstream `mlx-swift` lacks `bits=1`
quantized matmul.

`vision-segment-sam31` packages the native SAM 3.1 segmentation and tracking runtime used by `mere.run vision segment`, `mere.run vision track`, and `mere.run vision track-live`. Managed or local SAM roots are expected to contain:

- `config.json`
- `model.safetensors` or `model.safetensors.index.json`

Tokenizer files are optional for geometry prompts. Text prompts require
`tokenizer.json` and `tokenizer_config.json` in the SAM root or tokenizer
subdirectory.

The manifest for this package advertises both `vision_segmentation` and
`vision_tracking` capabilities.

`image-hidream-o1` and `image-hidream-o1-dev` map to the public HiDream O1
image checkpoints:

- `HiDream-ai/HiDream-O1-Image`
- `HiDream-ai/HiDream-O1-Image-Dev`

HiDream O1 uses a unified pixel-transformer root layout rather than a
VAE/text-encoder component tree. Managed or local roots are expected to contain:

- `config.json`
- `tokenizer_config.json`
- `tokenizer.json` or `vocab.json` plus `merges.txt`
- `preprocessor_config.json`
- `model.safetensors` or `model.safetensors.index.json`

The native Swift runtime validates this layout, decodes the typed root
configuration, prepares text/reference sample metadata, and runs generation
through the downloaded Qwen3-VL decoder, vision tower, timestep embedder, patch
embedder, generation-aware attention mask, and HiDream pixel head. Text-only
generation, one-reference instruction editing, and multi-reference subject
personalization share the same native path; reference modes additionally run
Qwen3-VL vision preprocessing and replace chat-template image placeholders
before denoising.

Runtime defaults come from the managed manifest:

- `image-hidream-o1-dev`: 28 steps, CFG 0.0, fixed flash FlowMatch schedule
- `image-hidream-o1`: 50 steps, CFG 5.0, shifted Flow UniPC schedule

Both checkpoints are large BF16 unified-transformer roots, about 33 GiB on disk
each before filesystem compression effects. Expect high unified-memory pressure
and prefer one-step smokes before full-quality 28/50 step runs.

`image-krea2-raw` and `image-krea2-turbo` map to Krea's public Krea 2
checkpoints:

- `krea/Krea-2-Raw`
- `krea/Krea-2-Turbo`

Managed or local roots are expected to contain the component Diffusers layout:

- `model_index.json`
- `tokenizer/tokenizer_config.json`
- `tokenizer/tokenizer.json`
- `text_encoder/config.json`
- `text_encoder/model.safetensors`
- `transformer/config.json`
- `transformer/diffusion_pytorch_model.safetensors.index.json`
- `transformer/diffusion_pytorch_model-*.safetensors`
- `vae/config.json`
- `vae/diffusion_pytorch_model.safetensors`
- `scheduler/scheduler_config.json`

Krea also publishes root-level `raw.safetensors` and `turbo.safetensors`
transformer copies for the official codebase. Managed pulls intentionally
exclude those files and pull the split transformer component instead, so the
model store does not download the same large transformer payload twice.

The native Swift runtime follows the public Krea 2 sampler shape: Qwen3-VL text
conditioning with the Krea system prefix, layer-selected hidden-state fusion,
single-stream MMDiT denoising, 16-pixel image-token alignment, FlowMatch Euler
steps, and Qwen Image VAE decoding. The wired public generation mode is
text-to-image with optional LoRA adapters; reference images and image-to-image
are not supported for this family yet. LoRA training uses `image-krea2-raw`;
inference uses `image-krea2-turbo`.

Krea publishes sample LoRA adapters such as
`krea/Krea-2-LoRA-retroanime` and `krea/Krea-2-LoRA-kidsdrawing`. Those
adapters are trained on Raw and loaded on Turbo with Diffusers
`lora_A` / `lora_B` keys. The native loader preserves Krea's `img_in`,
`txt_in`, `text_fusion`, `time_embed`, `time_mod_proj`,
`transformer_blocks`, and `final_layer` module names so those published
adapters can be used as compatibility references.

Runtime defaults come from the managed manifest:

- `image-krea2-raw`: 52 steps, CFG 3.5, FlowMatch shift/mu 1.15
- `image-krea2-turbo`: 8 steps, CFG 0.0, FlowMatch shift/mu 1.15

The component install is about 36 GiB before filesystem compression effects.
Expect high unified-memory pressure and prefer 96 GB+ Apple Silicon machines.

`image-ideogram4-sdnq-uint4` maps to WaveCut's public SDNQ uint4 Ideogram 4
snapshot:

- `WaveCut/ideogram-4-sdnq-uint4`

Managed or local roots are expected to contain:

- `model_index.json`
- `quantization_manifest.json`
- `tokenizer/tokenizer_config.json` or `tokenizer/tokenizer.json`
- `text_encoder/config.json`
- `transformer/config.json`
- `transformer/diffusion_pytorch_model.safetensors`
- `unconditional_transformer/config.json`
- `unconditional_transformer/diffusion_pytorch_model.safetensors`
- `vae/config.json`
- `vae/diffusion_pytorch_model.safetensors`
- `scheduler/scheduler_config.json`

The managed manifest records SDNQ asymmetric uint4 weights and the separate
positive and unconditional transformer branches used by Ideogram 4 guidance.
The current native support can pull, inspect, validate, decode SDNQ uint4
linear, embedding, and Conv2d weights, build Qwen3-VL concatenated text
features, pack Ideogram 4 text/image samples, run positive/unconditional CFG
denoising, and decode PNG output through the Flux2-style VAE. Text-to-image
`image generate` is wired; image-to-image, reference inputs, and LoRA are still
unsupported for this family.

## Hugging Face Cache

Hub snapshots use the shared cache location managed by the runtime. Override it
when you want large models on an external disk:

```bash
export MERERUN_HUB_CACHE=/Volumes/Models/huggingface
swift run mere.run model pull image-zimage-nano
```

Model pulls are resumable at the file level through the Hugging Face snapshot
cache. The CLI writes a managed-model symlink from the mere.run model store to
the prepared snapshot when needed.

## Hardware Support Checks

Managed pulls are gated by the local capability catalog before any download. The
check uses supported local runtimes plus memory thresholds for each model
family, then blocks models that are unlikely to run reliably on the current machine.

Inspect the local recommendation first:

```bash
swift run mere.run model capabilities
swift run mere.run model capabilities --all
```

If you are intentionally testing an unsupported setup, pass
`--allow-unsupported` to `mere.run model pull`.

## Model Store Behavior

The CLI resolves models in this order:

1. `--models-root` process override
2. `MERERUN_MODELS_DIR`
3. persisted local model-store setting
4. default `~/Library/Application Support/MereRun/models`

Examples:

```bash
# Pull into the default model store
swift run mere.run model pull image-zimage-nano

# Pull into a custom SSD-backed store
MERERUN_MODELS_DIR=/Volumes/Models swift run mere.run model pull text-chat-q36-nano
MERERUN_MODELS_DIR=/Volumes/Models swift run mere.run model pull text-chat-lfm25-a1b-8bit

# Inspect what is currently installed
swift run mere.run status
swift run mere.run model list
swift run mere.run model info image-klein-max
```

## Music, SFX, And Video Layouts

Some retained surfaces have more structure than a flat model root.

### `music-acestep`, `music-acestep-xl-turbo`, and `music-acestep-xl-turbo-lm4b`

The top-level model roots are:

```text
.../models/music-acestep
.../models/music-acestep-xl-turbo
.../models/music-acestep-xl-turbo-lm4b
```

Those roots may contain:

- `acestep-v15-turbo/`
- `acestep-v15-xl-turbo/`
- `acestep-5Hz-lm-1.7B/` or another supported LM subdirectory
- `acestep-5Hz-lm-4B/`
- `Qwen3-Embedding-0.6B/`
- `vae/`

Older local installs that still use `music-acestep-v15-turbo/` remain
supported.

`music-acestep-xl-turbo` pulls the ACE-Step 1.5 XL turbo DiT into
`acestep-v15-xl-turbo/` and reuses the base ACE-Step 1.5 VAE and Qwen3 text
encoder components. `music-acestep-xl-turbo-lm4b` adds the optional
`acestep-5Hz-lm-4B/` 5 Hz LM. The default ACE-Step LM component discovery
continues to prefer the smaller `acestep-5Hz-lm-1.7B/` when both LM directories
are present; pass `--lm-subdirectory acestep-5Hz-lm-4B` to force the 4B LM.

`mere.run music generate` and `mere.run music analyze` auto-discover these
layouts unless you override the root with `--checkpoints-root` or
`MERERUN_MUSIC_ACESTEP_ROOT`.

### `music-magenta-rt2-small` and `music-magenta-rt2-base`

Magenta RT2 models use exported runtime assets from
`google/magenta-realtime-2` at revision
`010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc`. The managed pull keeps only the
files needed by the native runtime:

```text
.../models/music-magenta-rt2-small
├── models/mrt2_small/mrt2_small.mlxfn
├── models/mrt2_small/mrt2_small_state.safetensors
├── resources/musiccoca/
└── resources/spectrostream/
```

The base model uses `models/mrt2_base/` with matching `mrt2_base` filenames.
Raw `checkpoints/*.safetensors` files are not a complete `mere.run` layout.

`mere.run music generate --model music-magenta-rt2-small` renders an offline
WAV. `mere.run music realtime --model music-magenta-rt2-small` plays on the
default macOS audio device and can capture to WAV with `--output`.

### `sfx-woosh-dflow`

The Woosh DFlow model root is:

```text
.../models/sfx-woosh-dflow
└── checkpoints/
    ├── Woosh-DFlow/
    ├── Woosh-AE/
    └── TextConditionerA/
        └── tokenizer/
```

The managed pull uses the `AEmotionStudio/woosh-models` Hugging Face mirror for
Sony Research Woosh v1.0.0 weights and mounts `FacebookAI/roberta-large`
tokenizer files under `checkpoints/TextConditionerA/tokenizer/`. The native
runtime exposes the text-to-SFX distilled DFlow path through
`mere.run sfx generate`.

### `sfx-woosh-flow`

The Woosh original Flow model root is:

```text
.../models/sfx-woosh-flow
└── checkpoints/
    ├── Woosh-Flow/
    ├── Woosh-AE/
    └── TextConditionerA/
        └── tokenizer/
```

The managed pull uses the same mirror and tokenizer mount as
`sfx-woosh-dflow`. The native runtime exposes the original text-to-SFX Flow
checkpoint through `mere.run sfx generate --model sfx-woosh-flow`; it generally
needs more denoise steps than the distilled model.

### Woosh CLAP and V2A

`sfx-woosh-clap` installs `checkpoints/Woosh-CLAP/` plus the mounted
RoBERTa tokenizer for native text/audio scoring through
`mere.run sfx clap score`.

`sfx-woosh-dvflow-8s` and `sfx-woosh-vflow-8s` install the distilled and
original video-to-audio checkpoint stacks from `AEmotionStudio/woosh-models`.
Both include `checkpoints/Woosh-AE/` and `checkpoints/TextConditionerV/`.

`sfx-woosh-synchformer` installs the companion
`mmaudio_synchformer_fp16.safetensors` visual extractor from
`Kijai/MMAudio_safetensors`. `mere.run sfx video generate` uses it when the
input is a raw video file; `.npy` inputs can still provide precomputed
Synchformer `synch_out` features directly.

### `video-ltx-av`

The unified AV model root is:

```text
.../models/video-ltx-av
```

`mere.run video generate --variant distilled --model video-ltx-av` is the
faster video-only draft path. `mere.run video generate --variant unified-av
--model video-ltx-av` remains available for the older synchronized AV root.
`MERERUN_VIDEO_LTX_MODEL_ROOT` can still point at this layout explicitly.

### `video-ltx23-av-mlx`

The LTX 2.3 MLX split model root is:

```text
.../models/video-ltx23-av-mlx
```

It pulls the distilled split checkpoint from
`dgrauet/ltx-2.3-mlx`, including `split_model.json`, connector weights,
separate video VAE/audio VAE/vocoder files, and the LTX 2.3 upscalers.
`mere.run model pull video-ltx23-av-mlx` also installs the hidden
`text-encoder-ltx-gemma3-12b-4bit` companion used for Gemma 3 prompt
conditioning. Set `MERERUN_VIDEO_LTX_TEXT_ENCODER_ROOT` only when pointing at an
external `mlx-community/gemma-3-12b-it-4bit` checkout.

The native Swift `unified-av` runtime has a split loader for this layout,
including the LTX 2.3 V2 connector, unified AV transformer, split video/audio
VAE files, BWE vocoder, and MLX-native tensor layouts. Use this model for the
current high-quality synchronized audio/video lane.
The Unsloth `LTX-2.3-GGUF` checkpoint family is a separate quantized GGUF lane
and is not loaded by the native MLX video runtime.
