# LFM2 Runtime

Native Swift MLX runtime for LiquidAI LFM2 text-chat checkpoints.

## Managed model

- Canonical id: `text-chat-lfm25-a1b-8bit`
- Upstream: `LiquidAI/LFM2.5-8B-A1B-MLX-8bit`
- Serving engine: `text-chat-lfm2`

The runtime loads MLX-converted directory-root snapshots with:

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors` or `model.safetensors.index.json` plus shards

## Architecture

`LFM2Model.swift` mirrors the `lfm2_moe` MLX layout:

- token embedding with tied output projection
- hybrid decoder layers selected from `layer_types`
- full-attention layers with q/k RMSNorm, RoPE, GQA KV repetition, and `KVCache`
- short depthwise-conv layers with recurrent convolution state
- dense feed-forward layers for the configured dense prefix
- sparse MoE feed-forward layers with router top-k, optional `expert_bias`, and
  `SwitchGLU` expert projections

`LFM2Generator.swift` is the chat entrypoint. It resolves managed installs or
downloads through `ManagedModelResolver`, loads tokenizer/template resources,
applies sharded safetensor weights with `HFSafetensorsWeightsLoader`, pre-fills
in cancellable chunks, then decodes serially.

## Notes

- This runtime is Swift-native and does not bridge to Python.
- LFM2 is currently text-only. API requests with image content parts are rejected
  by the `text-chat-lfm2` capability profile.
- Prefix KV reuse and continuous decode batching are not implemented for LFM2 yet.
