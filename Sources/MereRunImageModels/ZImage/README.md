# ZImage model configuration

Typed checkpoint configuration, model metadata, and the sparse-attention
admission policy. Transformer computation lives in `Transformer/`; Core owns
filesystem loading and runtime resource selection.

`DynamicSparseAttentionRuntime.swift` qualifies shapes against the dense
reference and schedules sparse execution. Shared kernels live in `MereRunTensor`.
