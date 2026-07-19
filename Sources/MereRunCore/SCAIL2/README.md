# SCAIL-2 native runtime

This directory owns the native Swift/MLX SCAIL-2 implementation. The runtime
consumes the separately packaged MIT-licensed model root named
`video-scail2-14b-mlx`; it never starts Python or ComfyUI.

The local root contains:

- `config.json`: the typed SCAIL-2 14B architecture and generation defaults.
- `model.safetensors` or a sharded `model.safetensors.index.json`: the 40-block
  SCAIL-2 diffusion transformer.
- `clip.safetensors`: the visual tower of OpenCLIP XLM-R ViT-H/14.
- `t5_encoder.safetensors` or a sharded index: UMT5-XXL text encoder weights.
- `tokenizer.json`: the local UMT5 tokenizer.
- `vae.safetensors`: the Wan 2.1 16-channel video VAE.
- `LICENSE`, `README.md`, and `MERERUN_CONVERSION.json`: the MIT model terms,
  model card, source revision, conversion receipt, and artifact hashes.

Conversion and publication are owned by the model repository rather than the
mere.run source tree. `mere.run model pull video-scail2-14b-mlx` installs the
immutable `Sawfwair/SCAIL-2-14B-MLX` snapshot pinned in
`SCAIL2Resources.managedRevision`.

The input contract is one reference image and seven-color reference mask, a
driving video and matching seven-color mask video, plus an animation or
replacement mode. Mask colors are packed in this order: white, red, green,
blue, yellow, magenta, cyan. Long videos use 81-frame segments with five clean
history frames, matching upstream SCAIL-2.
