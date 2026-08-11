# Muse Glimmer

Native Swift/MLX text-and-image inference for Meta Muse Glimmer 30B.

The audited converter defaults to a selective Q4 layout that leaves the token
embedding and complete vision tower in BF16. A smaller `compact` scope is
available for explicit low-memory experiments. The loader discovers quantized
modules from the checkpoint's `.scales` arrays and accepts both upstream and
common MLX Hub key layouts, so selective and compact artifacts share the same
native runtime.

Image preparation performs separable uint8 Lanczos-3 resizing with per-pass
rounding and clamping before normalization, matching the released Pillow and
torchvision path. Vision position geometry stays in float32. Exact byte and bit
pattern fixtures cover both contracts so a visually plausible preprocessing
change cannot silently alter model input.

- `MuseGlimmerConfig.swift`: typed upstream configuration, including per-layer
  NoPE/rotary selection and optional MLX quantization metadata.
- `MuseGlimmerModel.swift`: dense local/global decoder, gated attention,
  centered residual norms, ViT-G/14 perception encoder, and multimodal scatter.
- `MuseGlimmerImageProcessor.swift`: aspect-preserving patch preparation and
  prompt placeholder expansion.
- `MuseGlimmerTokenizerAndTemplate.swift`: the released ATEM chat/tool template.
- `MuseGlimmerAssistantModel.swift`: the official five-layer DFlash assistant,
  including target hidden-state projection and bidirectional sliding attention.
- `MuseGlimmerDFlashDecoder.swift`: lossless draft verification, rejection
  correction, and acceptance-aware target-only fallback.
- `MuseGlimmerGenerator.swift`: managed target/assistant loading and native decode.

The managed target is pinned to Sawfwair's selective MLX Q4 artifact at
`Sawfwair/Muse-Glimmer-30B-MLX-4bit@6532e898dc5c1a55b51b1b108cd36728b79be751`.
Its receipt pins the original Meta BF16 source at
`meta-models/Muse-Glimmer-30B@f84ecc3a0ea984a4c04542a84269e3d065350a6e`,
including every source and output hash. The automatically installed DFlash
companion is pinned to
`meta-models/Muse-Glimmer-30B-assistant@2c86316d689027b91123638739743fef1d425233`.
Inference never invokes Python. The converters under `scripts/model-conversion/`
are offline release tooling only.

Native DFlash captures target layers 1, 13, 25, 37, and 49 during prefill,
projects them through the assistant, drafts from one real anchor plus mask
tokens, and verifies the entire candidate block with one target forward. The
replacement selected at a rejected position becomes the next anchor, matching
the released algorithm without an extra recovery forward. Greedy output is
covered against serial target decode in `MuseGlimmerTests`.

The checkpoint permits 15 draft tokens per round, but the affine-Q4 target's
best measured MLX setting on an M4 Max is 3. In two interleaved 128-token
release runs, target-only decode averaged 14.73 tok/s and native DFlash averaged
20.07 tok/s with 54.9% draft acceptance: 1.36x, or 36.2% faster. This is a local
diagnostic, not a cross-machine guarantee. The runtime uses DFlash for output
budgets of at least 32 tokens and falls back losslessly after two rounds below
40% acceptance.

A paired 16-window, 8,192-token WikiText-2 candidate check measured 9.535
perplexity for selective Q4 and 9.554 for compact Q4. The selective-minus-compact
mean NLL delta was -0.0020 with a 95% bootstrap interval of -0.0035 to -0.0004.
That bounded sample supports selective Q4 as the default conversion scope; it is
not a substitute for a full release-quality evaluation.

The official BF16 assistant is preferred even beside a Q4 target. An optional
Q4 assistant measured slower at the same acceptance rate on this workload, so
its converter is retained for reproducibility rather than selected first.
