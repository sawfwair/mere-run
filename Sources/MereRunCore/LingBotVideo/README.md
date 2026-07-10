# LingBotVideo

Native Swift/MLX text-to-video runtime for the LingBot-Video Dense 1.3B and
quantized 30B-A3B MoE checkpoints.

Read in this order:

1. `LingBotVideoPipeline.swift` owns prompt encoding, CFG, denoising, VAE decode, and output tensors.
2. `LingBotVideoResources.swift` is the typed Hugging Face checkpoint boundary.
3. `LingBotVideoPromptEncoder.swift` loads the Qwen3-VL language stack without retaining its vision tower.
4. `LingBotVideoPromptSample.swift` preserves upstream structured-caption and Auto Negative JSON serialization.
5. `LingBotVideoTransformer.swift` implements the Dense and routed-expert joint video/text DiT.
6. `LingBotVideoScheduler.swift` implements the upstream Flow-UniPC solver.
7. `LingBotVideoMoEQuantizer.swift` converts routed experts one shard at a time to MLX affine weights.

The converted MoE refiner runs on demand with `video generate --refiner`. It
reuses the prompt embeddings, bicubic-upscales and VAE-encodes the base video,
then executes the released threshold-plus-tail Flow-UniPC schedule. The
separate Qwen3.6 rewriter LoRA remains outside this runtime.

The T2V path matches the released prompt template, source-ordered compact JSON,
duration round-up, CFG, Flow-UniPC, latent normalization, and refiner defaults.
`video generate --temporal-probe` stops at a selected denoiser step, decodes the
predicted clean sample, and lets the CLI score temporal stability before a full
base or refiner run. Large steps report completed DiT blocks for each CFG branch.
The transformer uses fused MLX RMSNorm/LayerNorm and cached row-concatenated QKV
and gate/up projections. `MERERUN_LINGBOT_FUSED_PROJECTIONS=0` is the regression
kill switch. Masked batched CFG preserves per-sample text lengths and RoPE
positions; it is exposed as an opt-in because one Apple GPU still performs both
branches' arithmetic.
