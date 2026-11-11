# H3-Image numerical fixtures

Generated with `scripts/validation/minimax-h3-image-reference.py` from the
Apache-2.0 Diffusers transformer, image VAE, and scheduler at commit
`f598e6e6a99528858b962c1399c45b59a6c25484`. The script verifies source SHA-256
hashes before executing the upstream classes. It uses PyTorch 2.15.0 and
Diffusers 0.42.1 on CPU.

The fixtures contain synthetic weights, three text rows, an optional
reference row, and four target rows. The BF16 case retains FP32 timestep,
input, and output projections. Outputs cover all six PDD heads and complete
text-only and reference-conditioned Euler trajectories computed by the upstream
scheduler, including H3's data-ward velocity sign. The two-layer image
decoder fixture uses a non-square latent grid to distinguish two-axis rotary
positions from the video decoder's three-axis positions.

Checkpoint mappings reverse Diffusers' SwiGLU half swap and fuse its separate
Q/K/V projections into the runtime's global slabs. These small fixtures do
not contain released model weights or establish image quality or throughput.

For exact GPU fixture comparisons on M5 and later, launch tests with
`MLX_ENABLE_TF32=0 MERERUN_TEST_MLX_DEVICE=gpu`. MLX otherwise permits TF32
matmuls, whose rounding differs from these FP32 CPU reference values.
