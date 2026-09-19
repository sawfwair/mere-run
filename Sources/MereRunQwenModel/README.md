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

Bonsai 2 uses `Q35PrismConfiguration` and the packed Prism projection and
embedding modules. Schema-2 module records bind each transformed tensor to its
explicit sign vector and block size. Projections apply the signed normalized
Hadamard transform before affine 2-bit matmul; embeddings apply the inverse
transform after row lookup. GDN activations are already grouped in the pack.
Do not permute them or fuse rotated projections through the ordinary affine
projection path. Core validates complete text parameter coverage and preserves
the pack's FP16 vision weights. `Bonsai2Tests` covers transform math, pack
rejection, and native loading; full-checkpoint parity is a separate gate.

`Q35PrismFusion` shares the signed transform across attention Q/K/V,
linear-attention QKV/Z, and dense MLP gate/up projections when their block
sizes and sign vectors match. The cache stays outside the parameter tree and
invalidates on module replacement. Set `MERERUN_Q35_PRISM_FUSION=0` for the
reference separate-projection path, or `MERERUN_Q35_PRISM_FUSION=1` to also
combine each group into one row-concatenated packed matmul. The latter retains
about 4.35 GiB of additional packed weights for Bonsai 2. The default shares
transforms without copying weights. `Bonsai2FusionTests` checks numerical
agreement, incompatible rotations, and cache invalidation.

On Metal, the validated 1024-element Prism transform uses one FP32 butterfly
kernel that also applies the sign vector and activation casts. Other block
sizes and CPU execution use the reference MLX operations. Set
`MERERUN_Q35_PRISM_METAL=0` to select the reference transform for comparison.
Run `Bonsai2FusionTests` with `MERERUN_TEST_MLX_DEVICE=gpu` for exact transform
comparisons in both directions and all supported floating-point dtypes.
`Bonsai2PerformanceTests` is an opt-in component profile: set
`MERERUN_BONSAI2_PROFILE` to the installed checkpoint directory and select the
GPU test device. Its synchronized timings include launch overhead and are not
full-model decode throughput.
