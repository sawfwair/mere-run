# LTX Module

This directory owns native video generation for mere.run.

- `LTXDistilledLatentGenerator.swift`: distilled text-to-video and image-to-video path
- `LTXInferenceTimings.swift`: typed model-load and generation phase timings
- `LTXGemmaTextEncoder.swift`: text encoder support for LTX models
- `LTX25Resources.swift`: immutable official LTX 2.5 source pins and packed-component layout validation
- `LTXVideoMP4Writer.swift`: final MP4 assembly

Edit here when you are changing native LTX loading, denoising, decode, or export behavior. Do not mix unrelated CLI concerns into this directory; keep command parsing in `Sources/MereRunCLI/Commands/VideoCommand.swift`.

Before stopping, run `swift test`, `./scripts/check.sh`, and a smoke path if your change affects real model execution.

## LTX 2.5

The official LTX 2.5 distilled release runs through the native `LTXUnifiedAVGenerator` with no Python sidecar. `LTX25Resources` validates the packed Hugging Face layout; `LTXGemmaTextEncoder` reads the embedded tokenizer and Gemma 4 weights directly; and the unified generator loads the packed transformer, video VAE, audio VAE/BWE vocoder, and spatial upsampler. Stage one uses the release's ancestral Euler update and stage two uses its deterministic refinement schedule.

Pull the gated checkpoint into the managed model store:

```bash
mere.run model pull video-ltx25-distilled-bf16 --accept-model-license
```

The Hugging Face account behind `HF_TOKEN` must already have access. Then use
the managed ID (or pass an official checkout with `--model-root`):

```bash
mere.run video generate "a small robot walks across a wooden table" \
  --model video-ltx25-distilled-bf16 \
  --quality final \
  --output-mode audio-video \
  --fps 24
```

The first installed-checkpoint acceptance render used source revision `dd53cc2cd45bbeaa3563dfb575cba3f49cf44761` and the upstream `v1.2.0` implementation at `d151147788a9284cca791edc6ce898007e727fe6`. It produced 25 H.264 frames plus a 48 kHz stereo AAC track from the release binary.
