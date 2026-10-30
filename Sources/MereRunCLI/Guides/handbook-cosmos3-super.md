# Cosmos3 Super 4-Step (NVIDIA)

`image-cosmos3-super-t2i-4step` installs the published MLX affine INT4/group-64
export of NVIDIA Cosmos3-Super-Text2Image-4Step. It generates still images through
the native Cosmos3 command. The image API and `image generate` do not serve this
checkpoint.

## Example to adapt

```bash
mere.run model pull image-cosmos3-super-t2i-4step
mere.run video cosmos3 "A red cube on a clean neutral background" \
  --model image-cosmos3-super-t2i-4step --mode text-to-image \
  --width 768 --height 768 --steps 4 --guidance-scale 1 --seed 7 \
  --output cosmos3-super.png
```

To use an existing copy, register its root with
`mere.run model location bind image-cosmos3-super-t2i-4step /path/to/checkpoint`.
The root must contain the converted transformer, VAE, scheduler, text tokenizer,
and distilled schedule. Edge reasoner and vision assets are not required.

## Controls and variants

- Use `--mode text-to-image`. This export does not support video, image
  conditioning, actions, sound, or the Edge reasoner.
- Keep four steps and guidance scale 1. The checkpoint supplies its distilled
  sigma schedule; custom shifts and schedules are unsupported. Negative prompts
  have no effect at guidance scale 1.
- Width and height must be divisible by 16. Start at 768 × 768, the published
  qualification size. Larger sizes need separate memory and quality checks.
- The catalog pins the 37.03 GB converted artifact. It does not download the
  original BF16 checkpoint. The catalog's 64 GB minimum and 128 GB recommendation
  are planning estimates, not minimum-memory qualification results.
- `video-cosmos3-edge-mlx` is a separate checkpoint with video, action, and
  reasoner modes. Its assets and default generation recipe differ.

## Sources and validation

- [Published MLX export](https://huggingface.co/Sawfwair/Cosmos3-Super-Text2Image-4Step-MLX-4bit),
  pinned to `4ef3565cbc05bf3e2867ced9c2cb761df52c6105`.
- [NVIDIA source checkpoint](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image-4Step),
  pinned by the converter to `aa0d5a57b7b045d68daa60fbacd84ec723c7cb7b`.
- The export reports a bounded 768 × 768 generation on an M4 Max with 128 GB
  memory. This does not establish broad image quality or other hardware limits.
- Catalog tests validate asset discovery, typed configuration, quantization,
  the four-step schedule, and manifest metadata. They have not been validated
  with model inference as part of this catalog addition.
- The checkpoint uses OpenMDW-1.1; consult the artifact's license before use.
