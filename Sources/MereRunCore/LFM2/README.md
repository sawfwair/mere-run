# LFM2 Runtime

Native Swift MLX runtime for LiquidAI LFM2 text and vision-language checkpoints.

## Managed models

- `text-chat-lfm25-a1b-8bit` — `LiquidAI/LFM2.5-8B-A1B-MLX-8bit`
- `text-chat-lfm25-a1b-bf16` — `LiquidAI/LFM2.5-8B-A1B` with DSpark
- `text-chat-lfm25-1.2b-bf16` — `LiquidAI/LFM2.5-1.2B-Instruct`
- `text-chat-lfm25-2.6b-4bit` — the `4bit/` partition of
  `LiquidAI/LFM2.5-2.6B-MLX`
- `text-chat-lfm25-2.6b-bf16` — `LiquidAI/LFM2.5-2.6B` with DSpark
- `vision-chat-lfm25-3b-8bit` — `LiquidAI/LFM2.5-VL-3B-MLX-8bit`
- Serving engine: `text-chat-lfm2`

The runtime loads MLX-converted directory-root snapshots with:

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors` or `model.safetensors.index.json` plus shards
- `processor_config.json` for the vision-language checkpoint

## Architecture

`LFM2Model.swift` mirrors both the dense `lfm2` and sparse `lfm2_moe` MLX
layouts:

- token embedding with tied output projection
- hybrid decoder layers selected from `layer_types`
- full-attention layers with q/k RMSNorm, RoPE, GQA KV repetition, and `KVCache`
- short depthwise-conv layers with recurrent convolution state
- dense `w1`/`w2`/`w3` feed-forward layers for `Lfm2ForCausalLM`
- dense `gate_proj`/`up_proj`/`down_proj` layers for the MoE model's configured
  dense prefix
- sparse MoE feed-forward layers with router top-k, optional `expert_bias`, and
  `SwitchGLU` expert projections

`LFM2VLModel.swift` adds the packed-patch SigLIP2 NaFlex encoder, resized
positional embeddings, factor-2 pixel unshuffle, and multimodal projector used
by LFM2.5-VL. `LFM2VLImageProcessor.swift` mirrors the checkpoint's smart
resize and patch packing contract. Only the multimodal prefill uses those
components; token decode continues through `LFM2Model` and its typed caches.

`LFM2Generator.swift` is the chat entrypoint. It resolves managed installs or
downloads through `ManagedModelResolver`, loads tokenizer/template resources,
applies sharded safetensor weights with `HFSafetensorsWeightsLoader`, pre-fills
in cancellable chunks, then uses either the pipelined serial loop or the
row-compacting continuous decode scheduler selected by the serving runtime.

For text targets, the managed catalog also installs the matching pinned
`*-dspark` companion. `LFM2DSpark.swift` projects hidden states captured at
the inputs to the checkpoint's five configured target layers into a five-layer,
non-causal diffusion-attention drafter. It proposes up to nine tokens per round;
`LFM2DSparkDecoder.swift`
verifies the block with the target model, commits full acceptances, and replays
only the bounded committed prefix after a rejection because LFM2 short-conv
state cannot be sliced. Greedy generation preserves the target's exact token
sequence; sampled generation uses rejection sampling. Set
`MERERUN_LFM25_DSPARK=0` to disable the route or
`MERERUN_LFM25_DSPARK_PATH=/path/to/sidecar` to test an explicit checkpoint.

Small-route decode fuses the affine-8 gate and up gather-GEMVs with SwiGLU,
then keeps the native down projection. The fused path is enabled by default
and can be disabled with `MERERUN_LFM2_FUSED_AFFINE8_MOE=0` for a controlled
A/B or rollback. It applies only on Apple GPU execution with BF16/FP16
activations, affine group size 64, 8-bit weights, an input width divisible by
512, and an output width divisible by 8. Other quantization and tensor layouts
fall back to MLX's portable gather path.

## Notes

- This runtime is Swift-native and does not bridge to Python.
- The three `text-chat-*` checkpoints reject image content parts. The
  `vision-chat-lfm25-3b-8bit` catalog entry enables them through the same
  serving engine.
- API serving enables exact token-prefix KV reuse by default and enables
  continuous decode batching when `--max-active-requests` is greater than one.
  Ragged rows carry independent attention offsets and typed short-convolution
  state so compatible requests may share one model forward across positions.
  Requests using DSpark currently bypass prefix-cache reuse and continuous
  batching so the drafter and target caches remain synchronized.
- Cold model preparation is deduplicated by the serving pool. Residency epochs
  invalidate stale decode loops and explicit unloads without canceling another
  request that is waiting on the same shared preparation task.
- Public generator entrypoints establish a task-local MLX default stream before
  loading or evaluating the model, matching the other native MLX chat engines.
