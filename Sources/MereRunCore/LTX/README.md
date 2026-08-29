# LTX Module

This directory owns native video generation for mere.run.

- `LTXDistilledLatentGenerator.swift`: distilled text-to-video and image-to-video path
- `LTXInferenceTimings.swift`: typed model-load and generation phase timings
- `LTXGemmaTextEncoder.swift`: text encoder support for LTX models
- `LTXPromptEmbeddingCache.swift`: bounded resident-session prompt embedding cache
- `LTXGuidanceProjectionCache.swift`: low-memory-safe full-guidance prompt projection reuse policy
- `LTX25Resources.swift`: immutable upstream and managed LTX 2.5 pins plus packed-component layout validation
- `LTXVideoMP4Writer.swift`: final MP4 assembly

Edit here when you are changing native LTX loading, denoising, decode, or export behavior. Do not mix unrelated CLI concerns into this directory; keep command parsing in `Sources/MereRunCLI/Commands/VideoCommand.swift`.

Before stopping, run `swift test`, `./scripts/check.sh`, and a smoke path if your change affects real model execution.

## LTX 2.5

The LTX 2.5 release runs through the native `LTXUnifiedAVGenerator` with no
Python sidecar. Both managed Sawfwair distributions store their BF16
transformers directly in mere.run's native module-key layout, so a pull does
not keep an official source transformer beside a generated optimization copy.
The Distilled distribution bundles a self-contained MLX Q4 Gemma 4 language
tower with its BF16 projection and tokenizer assets. The Full distribution
retains the official BF16 text tower and every full/dev parity component.
Stage one uses the release's ancestral Euler update and stage two uses its
deterministic refinement schedule.

Pull the public model into the managed model store after reviewing its terms:

```bash
mere.run model pull video-ltx25-distilled-bf16 --accept-model-license
```

Each model ID resolves to one inference-only repository and download; no local
re-keying, optimization, or quantization step is required. Then use the managed
ID (or pass a compatible official checkout with `--model-root`):

```bash
mere.run video generate "a small robot walks across a wooden table" \
  --model video-ltx25-distilled-bf16 \
  --quality final \
  --output-mode audio-video \
  --fps 24
```

The first installed-checkpoint acceptance render used source revision `dd53cc2cd45bbeaa3563dfb575cba3f49cf44761` and the upstream `v1.2.0` implementation at `d151147788a9284cca791edc6ce898007e727fe6`. It produced 25 H.264 frames plus a 48 kHz stereo AAC track from the release binary.
