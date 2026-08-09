# Shared MLX kernels

This directory contains reusable native MLX and Metal execution primitives.
Model-specific graph code remains in the model's own module and calls these
primitives only after validating its tensor layout and admission policy.

## Dynamic sparse attention

`DynamicSparseAttention.swift` implements batch-one, head-dimension-128
block-sparse attention. It summarizes 64-token blocks, constructs
query-dependent routes, preserves neighboring and prefix blocks exactly, and
evaluates the selected blocks with an online-softmax Metal kernel. The dense
route gate compares the custom kernel with MLX fused SDPA before a tensor shape
may use sparse execution.

`DynamicSparseAttentionRuntime.swift` owns the reusable step, layer, sequence,
and numerical admission checks. Z-Image enables it only for long sequences at
the measured crossover. MiniMax-H3 keeps its model-specific policy in the H3
generator and transformer.

## Editing rules

- Keep the shared kernel independent of model tensor layouts and masks.
- Preserve FP32 outputs when any Q, K, or V input is FP32.
- Add a real Metal parity test for kernel changes.
- Require model-specific warm timing and artifact validation before enabling a
  new caller by default.
- Keep unsupported head dimensions, masks, and batch shapes on fused dense
  SDPA rather than approximating their semantics.
