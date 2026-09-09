# Native Qwen model layers

This library owns Qwen 3.5, 3.6, and 3.8 configurations, attention and expert
layers, hybrid caches, MTP heads and committed draft history, Flash-Next PLE
row lookup, and the vision tower. It depends on MLX and the shared tensor,
attention-cache, and text-encoder libraries.

Core owns managed-model selection, resource resolution, checkpoint policies,
tokenizers, prompts, generation scheduling, and request cleanup. It re-exports
this library's public API. The vision loader in Core selects checkpoint shards
and passes mapped arrays to the model for installation.

Preserve serial/verification arithmetic, hybrid-cache rollback, independent
fork wrappers, QSA pooling state, PLE token history, and request-scoped compiled
graph ownership. MTP proposals never authorize output: target verification
remains in the Core generator.

Runtime hooks use package access. `Q35SwitchLinear` and its call method are open
so Core's Inkling LoRA adapter can retain its subclass; construction remains
package-scoped. Run the independent `QwenRuntimeTests` target before the full
repository gate. See [Qwen runtime boundaries](../../docs/internals/qwen-runtime-boundaries.md)
for the generator stages and validation limits.
