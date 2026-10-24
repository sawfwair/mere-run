# FLUX 3 Klein (Black Forest Labs)

Use `image-flux3-klein` for the experimental native Swift and MLX implementation of the
distilled FLUX 3 Image Klein model. It loads the original BF16 transformer,
Qwen3-VL-4B text encoder, and autoencoder directly. Python and checkpoint
conversion aren't required for inference.

## Install

Review the [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/8454017f9ebaada30c998fccd9f67a80c7580f68/LICENSE.md)
and obtain access to the gated Hugging Face repository. Configure your Hugging
Face token, then acknowledge the model terms when downloading:

```bash
mere.run model pull image-flux3-klein --accept-model-license
```

The managed download pins revision
`8454017f9ebaada30c998fccd9f67a80c7580f68`. It excludes the FP8, NVFP4, and
Diffusers copies of the transformer and VAE. The license has use and
distribution conditions; consult the linked terms for the Qualifying User
commercial-output exception.
Qualifying Users must satisfy the license's filtering or output-review
conditions and applicable AI-disclosure requirements.

## Example to adapt

```bash
mere.run image generate --model image-flux3-klein \
  --prompt "A brass camera on a walnut table, soft window light" \
  --width 1024 --height 1024 --steps 16 --seed 42 --output camera.png
```

## Controls and variants

The defaults are 16 steps and CFG 1. The distilled model predicts the flow
between the current and next timestep. It uses the publisher's rational
schedule with alpha 6.93 and retains Float32 Euler state.

Use sides divisible by 16, from 64 to 4096 pixels. The requested width and
height set the output canvas, including reference edits. Large canvases and
multiple references require more memory. Bounded 512/1024 generation and one-
and two-reference edits pass checkpoint smoke checks; 4K and ten-reference
workloads remain unqualified.

### Edit with references

```bash
mere.run image generate --model image-flux3-klein \
  --prompt "Put the camera from the first image on the desk in the second" \
  --ref-image camera.png --ref-image desk.png \
  --width 1024 --height 1024 --output composite.png
```

Supply up to 10 ordered references. `--input` supplies the first reference.
Single references are resized and reflect-padded; multiple references are
resized under one megapixel each and center-cropped to the 16-pixel grid.
References enter the image autoencoder, rather than the text encoder's vision
tower. Text conditioning retains padded tokens from eight Qwen3-VL layers.

Send bounding-box instructions in the prompt using the
[publisher's prompt format](https://github.com/black-forest-labs/flux-3-image-klein/tree/45e57c9fb06f9751cdced29a65c5d2171003be05).
There is no separate bounding-box file option or automatic prompt upsampler.

Negative prompts, CFG values other than 1, custom sigma schedules, edit strength,
LoRA inference or training, the base checkpoint, FP8, and NVFP4 are unsupported.
An identical seed reproduces native noise; it doesn't reproduce PyTorch's RNG.

## Sources and validation

Small Float32 and BF16 transformer fixtures compare the native implementation
with pinned BFL reference code, including reference conditioning and a complete
short Euler trajectory. Tiny VAE fixtures also compare encoding, decoding, and
PNG channel order with the reference implementation. The dated
[checkpoint qualification report](https://github.com/sawfwair/mere-run/blob/main/docs/benchmarks/flux3-klein-native-qualification-2026-10-15.md)
includes the generated images, exact prompts, checksums, elapsed times, and
process footprints for bounded 512/1024 generation and reference editing.
These cases do not establish broad quality, 4K operation, or ten-reference capacity.
