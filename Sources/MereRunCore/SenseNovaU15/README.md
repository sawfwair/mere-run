# SenseNova U1.5

Native Swift/MLX inference for the official `sensenova/SenseNova-U1.5-8B-MoT` checkpoint.

The runtime implements the checkpoint's dual-expert Qwen3 transformer, raw-pixel flow-matching image path,
resolution-aware noise schedule, text CFG, and multi-image editing. It does not invoke Python and does not use
an external VAE or text encoder.

The model is intentionally pinned in the managed catalog. Keep checkpoint key mapping limited to layout changes
required by MLX (PyTorch convolution kernels are OIHW; MLX kernels are OHWI).
