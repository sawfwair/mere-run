# FLUX 3 checkpoint metadata and VAE fixtures

`upstream-artifacts.json` records the pinned Hugging Face revision, original
file sizes, Git blob IDs, and LFS SHA-256 digests. No pretrained weights are
included. The public Qwen3-VL configuration and templates were compared with
the FLUX package's Git blob IDs; all three files are byte-identical.

`text-encoder-config.json` is the verified public configuration.
`vae-source-shapes.json` is the full-size original autoencoder's parameter schema,
generated on PyTorch's meta device from the pinned BFL source.

The VAE safetensors fixtures use synthetic weights and reduced channel widths
of 32 at every resolution. They preserve the original 32 latent channels,
2x2 packing, batch normalization, attention, quantization convolutions, and
encode/decode functions. Regenerate these files with
`scripts/validation/flux3-klein-reference.py` using its pinned dependencies.

These fixtures validate computation and tensor mapping. They do not qualify
the released checkpoint's text conditioning, image quality, or memory use.
