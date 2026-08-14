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
and also discovers the same pinned official shard under `mtp/` beside the
managed 4-bit target. `MERERUN_Q35_MTP_SPECULATION=1` enables greedy speculation
from short prompts. It stays opt-in because multi-token target verification can
choose a different greedy path from serial target decode. Hybrid MoE Qwen models
keep the existing adaptive long-context threshold.
