# Native Nemotron 3.5 Bolt model

This target loads the original 90-shard mixed IQ2_XXS/Q8_0/NVFP4 checkpoint.
`BoltConfig` validates typed heterogeneous block metadata. `BoltCheckpoint`
reads safetensors without interpreting E4M3 scale bytes as floating-point
weights. `BoltPacked` multiplies original GGML blocks directly on Metal,
including gather multiplication over selected experts. A scalar reference
path supports portable arithmetic tests.

`BoltModel` owns FP32 Mamba-2 recurrent state, attention caches without RoPE,
sigmoid routing, latent expert projections, and strict tensor inventories.
`BoltVision` reuses the repository's C-RADIO architecture and pixel shuffle.
Core owns media decoding, prompt planning, and user-facing results.

The IQ2 codebook is copied from ggml-org/llama.cpp at
`b76b13379d3b7d67ab0bc424516bb6fd2522bd8e`, `ggml/src/ggml-common.h`.
See `LICENSE.ggml`. The pinned NVIDIA architecture, configuration fixture,
and model materials use OpenMDW-1.1; see `LICENSE.OpenMDW-1.1`.

The runtime doesn't expand all experts to BF16 or invoke upstream Python.
LiLiCorr and video pruning require separate implementations and qualification.
