# Q35

Qwen 3.5/3.6/3.8 dense and hybrid MoE text and vision-language runtime.

- `Q35Config.swift`: typed text/vision configuration.
- `Q35TokenizerAndTemplate.swift`: checkpoint-native chat-template rendering,
  image-token expansion, and tokenization.
- `Q35Model.swift`: native model entry point.
- Attention and MoE files own model math only.

Keep tokenizer/tool template compatibility isolated here; model layers should
not know about CLI or managed-model concerns.

Official Hugging Face Qwen 3.5/3.8 checkpoints store zero-centered RMSNorm
offsets, while converted MLX checkpoints store direct scales. The loader detects
the checkpoint layout from embedded MTP keys or PyTorch Conv1d shapes and keeps
both conventions compatible with the native offset RMSNorm module.

The official Qwen3.8 27B shards embed a dense one-layer MTP head. The loader
reads only the shards that contain `mtp.*` tensors, maps its dense SwiGLU layout,
and also discovers a bare `model.safetensors` MTP component under `mtp/`. The
managed 4-bit lane pairs the MLX Fast reference target with a matching
4-bit/group-64 proposal head. `MERERUN_Q35_MTP_SPECULATION=1` enables greedy
speculation from short prompts. It stays opt-in because multi-token target
verification can choose a different greedy path from serial target decode.
Hybrid MoE Qwen models keep the existing adaptive long-context threshold.

Greedy Qwen3.8 MTP uses a proposal-only compact vocabulary projection containing
the first 98,304 tokenizer rows and the official control-token rows. A fused
Metal reduction maps its argmax back to the full tokenizer without materializing
the unused vocabulary tail. A request-local MTP cache is primed from up to 4,096
prompt hidden states and retains only target-confirmed transitions; speculative
rows execute on a disposable fork. The exact target projection still verifies
every emitted token. Per-request acceptance estimates adapt the draft depth and
can fall back to target-only rounds when proposals stop paying for their repair
cost. Sampled MTP keeps the full-vocabulary probability path.
