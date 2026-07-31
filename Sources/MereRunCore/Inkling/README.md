# Inkling

Inkling-Small text generation runs in the native Swift/MLX runtime.

- `InklingResources.swift` pins the official BF16 checkpoint and mere.run's
  mixed MLX conversion: routed experts are affine 2-bit/group-128 and all
  non-routed weights remain BF16.
- `InklingModel.swift` implements Inkling's relative-position attention,
  short-convolution residuals, and sigmoid-routed shared-expert MoE.
- `InklingGenerator.swift` owns managed loading, prompt encoding, chunked
  prefill, autoregressive decode, and native LoRA application.
- `InklingTextModelLoader.swift` is the shared checkpoint loader for chat and
  training.
- `InklingTextSFTTokenizer.swift` applies the released message format and
  masks loss to assistant output tokens.
- `InklingTextLoRAInjector.swift`, `InklingTextLoRATrainingPipeline.swift`, and
  `InklingTextLoRAAdapter.swift` implement attention, MLP, expert, and
  unembedding fine-tuning, held-out evaluation, adapter serialization, and
  runtime loading. Routed and shared experts use shared-outer factors: the
  hidden-dimension factor is shared while the intermediate factor remains
  expert-specific.

The released checkpoint accepts text, images, and audio. This first mere.run
lane intentionally loads only `language_model.*` weights and exposes text chat;
the vision and audio towers remain out of scope.

The architecture advertises 1,048,576 tokens. mere.run defaults to 32,768
because KV-cache and attention-mask residency still grow with context on the
128 GB machines targeted by this conversion.

Train and apply a native adapter with:

```bash
mere.run text train-lora \
  --model text-chat-inkling-small \
  --data ./inkling-sft.jsonl \
  --eval ./inkling-heldout.jsonl \
  --reasoning-effort 0.2 \
  --output ./inkling-assistant.safetensors

mere.run text chat \
  --model text-chat-inkling-small \
  --lora ./inkling-assistant.safetensors \
  --reasoning-effort 0.2 \
  --prompt "Answer using the fine-tuned behavior."
```

Training preserves the mixed checkpoint: routed experts stay affine
2-bit/group-128, non-routed weights stay BF16, and the optimizer updates only
the injected LoRA parameters. Inkling defaults to q/k/v/o attention,
gate/up/down MLPs (including routed and shared experts), and `lm_head`; pass
`--target-modules` to override that surface. The training path uses the generic
differentiable mask graph instead of the inference-only Metal mask kernel and
stops gradients through discrete expert indices. Use the same
`--reasoning-effort` value for training and inference.
