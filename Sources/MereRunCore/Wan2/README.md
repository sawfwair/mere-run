# Wan2 native runtime

This directory owns the native Swift/MLX Wan2.2 TI2V-5B runtime. The managed
model root is `video-wan22-ti2v-5b-mlx` and contains:

- `config.json`: typed Wan2.2 TI2V-5B architecture and inference defaults.
- `model.safetensors`: single 5B diffusion transformer in MLX layout.
- `t5_encoder.safetensors`: UMT5-XXL encoder weights in MLX layout.
- `tokenizer.json`: local UMT5 tokenizer; inference must not fetch it at runtime.
- `vae.safetensors`: float32 Wan2.2 48-channel VAE encoder and decoder weights.

The model is derived from the official `Wan-AI/Wan2.2-TI2V-5B` checkpoint.
`Wan2Resources` pins both the official source revision and the managed converted
snapshot used by `mere.run model pull`.

## Runtime boundaries

`Wan2TI2VGenerator` is a reusable warm runtime. Reusing one instance retains the
tokenizer, UMT5 encoder, diffusion transformer, VAE, prompt embeddings, and the
latest terminal-frame latent. `Wan2WorldSession` owns that runtime behind an
actor and exposes serial world transitions without passing mutable MLX tensors
to callers or launching a process per move.

World transitions carry a stable camera-control record with XYZ translation and
rotation. The base TI2V checkpoint currently applies that control through the
motion prompt and first-frame conditioning (`textAndFirstFrame`). A future
DreamX-derived camera adapter can consume the same control as causal camera
latents and report `causalCameraLatents` without changing the session request or
receipt schema.

The session writes one MP4 and one terminal PNG per transition. The next
transition consumes the terminal latent directly in memory; the PNG is an
observable state artifact, not the internal chaining path.

## Verified parity

The real-checkpoint harnesses compare native Swift/MLX against independent
upstream implementations:

- UMT5 output against Wan's PyTorch encoder.
- One-block and full 30-block transformer output against the public MLX Wan
  implementation.
- VAE image latents against Wan's official PyTorch VAE.
- Two consecutive world transitions with warm model reuse and terminal-latent
  chaining.

Tiny 128-pixel renders are structural tests only. Quality acceptance should use
at least a 512x320 spatial canvas; Wan geometry is divisible by 32 and frame
counts follow 4n+1.
