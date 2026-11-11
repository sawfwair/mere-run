# MiniMax-H3-Image

Core owns checkpoint discovery, loading, reference preparation, Qwen3-VL
conditioning, denoising orchestration, and PNG output. Computation lives in
`MereRunH3Model` beside the shared H3 layers.

The runtime loads the original checkpoint layout at
`MiniMaxAI/MiniMax-H3-Image@492fd8c4eca666f1085f57e30738c1e0d86b3439`.
It reads `transformer/`, `text_encoder/`, `vae/`, `tokenizer/`, and the fused
PDD heads. Each shard must match its index, and every model parameter must
be covered. The loader preserves FP32 embedding, time, and output tensors
and BF16 transformer weights. Original fused QKV rows are deinterleaved
before loading. The image VAE has 16 attention heads, not the video VAE's 32.

`MiniMaxH3ImageOptimizer` exports a separate native affine Q8/group-64 checkpoint.
`h3_image_mlx.json` declares its format, source identity, and exact modulation
cache. Export maps QKV rows before quantization, retains FP32 projections and
BF16 matrices whose input width does not align to 64, and omits unused Qwen
layers. The loader validates every shard and parameter; it never deinterleaves
already packed Q8 rows. The original VAE, tokenizer, license, and PDD heads stay
with the exported checkpoint. Publication uses a new directory and atomic rename.

The fixed six-step timestep cache preserves released-precision AdaLN outputs
and omits their large projection matrices. Rotary tables are prepared once per
request and cast to the Q/K activation dtype, as in upstream; FP32 tables must
not promote the BF16 attention and residual stream. Set
`MERERUN_H3_IMAGE_PROFILE=1` to retain per-block and large-sequence
QKV, attention, and feed-forward timings on stderr.

`MiniMaxH3ImageGenerator(retainModels: true)` serializes requests in an actor,
retains the transformer and VAE, and caches one exact text-only conditioning
result. A different root or manifest clears the retained state; references
always encode afresh. `unload()` releases all retained state. The default
generator remains per-request. `model benchmark h3-image` uses the retained
session for cold/warm comparisons, with optional blockwise BF16 materialization
and attention head/query limits and VAE tile-batch controls.

The image path packs text, ordered reference latents, and target latents.
Text and target share the current timestep. VAE references stay at 0.999.
Each of six Euler evaluations selects its own FP32 PDD output head. The
heads predict data-ward velocity, so the update adds
`(sigma_current - sigma_next) * velocity` in FP32. Trajectory fixtures call
the pinned upstream scheduler independently of this implementation.
The conditioner reads the unnormalized activation after Qwen decoder layer 50.
Large sequences split QKV projections and feed-forward evaluation into chunks
of at most 4096 rows. Native Q8 on `applegpu_g17s` uses at most 1024 queries
and 28 heads per attention kernel; other configurations default to 640 queries
and eight heads. Every query still attends to every key.
Reference labels are `Picture i: `, with no angle brackets; prompts refer
to them as `<Picture i>`.

The VAE uses overlapping 256-pixel tiles with at least 64 pixels of overlap.
Decode batches up to 32 existing tiles with the original FP32 VAE; encoding
remains sequential. Geometry, blending, and assembly order stay unchanged.
Its image decoder uses two-axis rotary embeddings, four learned register
tokens, and an extra zero token. Reference posterior samples round through
FP16 before latent normalization and patch alignment. The seeded native MLX
random streams differ from PyTorch CPU streams.

Run `H3ImageComputationTests`, `H3ImageIntegrationTests`, and the H3-Image CLI
capability test. Tiny numerical fixtures come from pinned upstream
Diffusers code through `scripts/validation/minimax-h3-image-reference.py`.
The [real-checkpoint report](../../../docs/benchmarks/minimax-h3-image-native-qualification-2026-11-10.md)
preserves the original 2K and 4K BF16 baseline. The
[Q8 follow-up](../../../docs/benchmarks/minimax-h3-image-q8-2026-11-10.md)
compares packed Q8 and resident BF16 compute, cold and warm requests, and
process memory. The [M5 acceleration qualification](../../../docs/benchmarks/minimax-h3-image-m5-nax-2026-11-10.md)
covers the scoped MLX dispatch repair, attention schedule, and batched decode.
Editing and broader quality remain unqualified.
Opt-in `H3ImageRealCheckpointTests` export real-weight
decoder and transformer probes when `MERERUN_H3_IMAGE_CHECKPOINT` and
`MERERUN_H3_IMAGE_PROBE_OUTPUT` are set. Use `MERERUN_TEST_MLX_DEVICE=gpu` for
Metal and `MERERUN_H3_IMAGE_PROBE_SIZE=4096` for the full-stack 4K token probe.
