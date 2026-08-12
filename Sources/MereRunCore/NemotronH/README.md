# Nemotron-H runtime

This directory implements NVIDIA Nemotron 3.5 Lightning 30B-A3B and its DSpark
speculative companion directly in Swift/MLX.

The target alternates 23 Mamba-2 blocks, six grouped-query attention blocks,
and 23 routed-MoE blocks. Mamba recurrent state remains FP32, attention uses no
RoPE, and each MoE block selects six of 128 routed experts alongside the shared
expert. Converted ModelOpt NVFP4 matrices retain their native E2M1 values,
E4M3 block scales, and FP32 global scales. Released FP8 Mamba projections are
materialized as BF16 once by the offline converter.

DSpark is a separate six-layer Qwen3-style checkpoint. It consumes six target
hidden-state taps, proposes masked blocks with sliding-window attention and the
released Markov head, and shares the target LM head. The decoder applies exact
target verification and residual-distribution replacement for sampled output.
Because Nemotron-H Mamba state cannot be sliced from a rejected candidate,
recovery replays only the bounded committed prefix from an untouched cache.
After two rounds below the configured acceptance threshold, decoding continues
serially on the already-correct target cache.

Conversion is deliberately outside runtime code:

- `scripts/model-conversion/convert_nemotron35_lightning_mlx.py`
- `scripts/model-conversion/convert_nemotron35_dspark_mlx.py`

Both converters reject source drift using exact revision, byte-count, tensor
inventory, and SHA-256 pins. Generated `MERERUN_CONVERSION.json` receipts record
the converted artifact hashes.
