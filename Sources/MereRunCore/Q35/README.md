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
Qwen3.6 hybrid MoE keeps the existing adaptive long-context threshold.

Ornith 1.5's official MLX quants omit the MTP tensors advertised by their
configuration. Managed Q4/Q6/Q8/BF16 pulls therefore install one shared,
revision-pinned final shard from the authoritative Ornith base checkpoint; the
runtime reads only its `mtp.*` tensors. Routed-expert gate/up fusion is prepared
one decoder layer at a time after target weights load. Each evaluated fused
stack replaces its two source arrays before the MLX cache is cleared, bounding
transient preparation memory instead of retaining a model-wide duplicate.
Verified Q4 measurements showed a short-prompt decode win, so managed Ornith
1.5 targets enable MTP from token zero while preserving target verification.

Greedy Qwen3.8 MTP uses a proposal-only compact vocabulary projection containing
the first 98,304 tokenizer rows and the official control-token rows. A fused
Metal reduction maps its argmax back to the full tokenizer without materializing
the unused vocabulary tail. A request-local MTP cache is primed from up to 4,096
prompt hidden states and retains only target-confirmed transitions; speculative
rows execute on a disposable fork. The exact target projection still verifies
every emitted token. Per-request acceptance estimates adapt the draft depth and
can fall back to target-only rounds when proposals stop paying for their repair
cost. Sampled MTP keeps the full-vocabulary probability path.

Qwen3.8 target verification marks its model forward explicitly. On macOS, only
that marked path may use the fused BF16 GDN prework kernel, and only for batch
one, a four-tap depthwise convolution, 128-wide key/value heads, and sequence
widths 3–9. The kernel replaces convolution-state concatenation, depthwise
Conv1d, SiLU, q/k/v preparation, RMS normalization, scaling, and next-state
capture. Decode, prefill, unsupported shapes, and non-Metal platforms retain
the composed operations. Metal tests require bit-exact parity for widths 3, 4,
5, 7, and 9.

The streaming tool-call parser walks the Qwen XML structure rather than using
delimiter search. A closing tag is accepted only at its structural position,
so strings containing tag-like text remain parameter data. Streaming reparses
are bounded; EOS still receives a final structural parse.

Qwen3.8-Flash-Next uses the `qwen4_exp` text architecture. Its native path
implements four-stream gated residual hyper-connections, PLE n-gram hashing and
embedding lookup, grouped RMS normalization, sigmoid gated-delta output, and
the published hybrid full/linear-attention schedule. PLE lookup keeps only the
two preceding token IDs in each incremental cache; its 128 logical embedding
parts are installed from the physical safetensors shards without duplicating
the table. Four-layer evaluation boundaries keep the 48-layer lazy graph below
the macOS Metal watchdog while preserving each hybrid block.

QSA indexers run in every full-attention layer. Raw index keys are mean-pooled
in four-token blocks in FP32, normalized, and rotated at the block's first
position. Normalized/rotated index queries score those blocks by summed ReLU
head scores. Future blocks are masked before selecting the best 512 complete
blocks; the current partial block is always included. At or below the 2,048-token
budget every visible token is selected, so the original causal SDPA path remains
exact. Above it, 16-query tiles bound gathered KV temporaries without allocating
a dense sequence-squared attention matrix. The model's 262,144-token context
limit remains subject to total KV, MTP-history, and model residency.

`Q38QSACache` snapshots raw index keys and rotary positions alongside the main
KV cache, including prefix forks, accepted-prefix MTP rollback, and right-padded
ragged batching. Quantized main KV retains unquantized indexer history. Tests
cover selector causality, GQA dense-mask parity, pool-before-norm/first-position
RoPE, chunk/serial parity, cache lifecycle, and the published 2,048-token boundary.

Qwen4Exp's bundled one-layer MTP head consumes the target's four-stream
hidden state, drafts through its trained hyper-connection/full-attention/MoE
block with the same QSA history, and leaves every emitted token under exact
target verification. MTP history priming uses 256-token chunks. Managed
Flash-Next models enable this path from short prompts; API startup warmup runs a
representative eight-token decode so its lazy target and draft graphs are paid
before the server reports healthy. `MERERUN_Q35_MTP_SPECULATION=0` keeps the
target-only path available for comparison or memory pressure.
