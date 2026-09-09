# Shared audio model layers

Use this library for audio model computation shared across runtime families.
`MMAudioBigVGAN` provides weight-normalized convolutions, alias-free activation,
residual blocks, and waveform upsampling for MMAudio and MiniMax-H3.

The target depends only on MLX libraries. It has no media I/O, audio codec,
model-catalog, tokenizer, or Core dependency. Checkpoint loading and key mapping
remain in Core extensions. The H3 audio/video boundary tests cover a small
vocoder's temporal expansion, batch isolation, output range, and dtype.
