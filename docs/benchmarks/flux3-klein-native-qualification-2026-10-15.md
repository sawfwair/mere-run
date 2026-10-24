# FLUX 3 Klein native checkpoint qualification

This report gives contributors the bounded checkpoint evidence for the native
Swift/MLX FLUX 3 Klein runtime. All four cases pass execution and visual review.
The images are AI-generated outputs from these qualification runs.

## Checkpoint and runtime

The managed model `image-flux3-klein` loads the original distilled BF16 package
without conversion or a Python inference process. The source is
[black-forest-labs/flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein/tree/8454017f9ebaada30c998fccd9f67a80c7580f68)
at revision `8454017f9ebaada30c998fccd9f67a80c7580f68`. The implementation follows
[BFL's reference](https://github.com/black-forest-labs/flux-3-image-klein/tree/45e57c9fb06f9751cdced29a65c5d2171003be05)
at `45e57c9fb06f9751cdced29a65c5d2171003be05`.

All four weight files match their pinned SHA-256 hashes. The transformer has
359 BF16 tensors; the autoencoder has 251 tensors and matches the upstream
schema. The text encoder uses the original Qwen3-VL-4B shards. The
[JSON receipt](/qualification/flux3-klein-native-2026-10-15.json) records the
weight hashes, inference-source hashes, prompts, input order, output hashes,
and process measurements.

The runs use an Apple M3 Max with 36 GiB unified memory and macOS 26.6.2.
Normal shared admission remains enabled. Components load and release in stages:
text encoder, optional reference autoencoder, transformer, and output autoencoder.
Every case uses 16 steps, CFG 1, and seed 42.

## Results

Generation seconds come from durable-run timestamps at one-second resolution
and exclude admission waiting. The receipt also preserves raw process wall time,
which includes queue delays. These shared-host runs are not isolated throughput
benchmarks. Peak footprint is the operating system's process measurement, not
an MLX allocation counter or a minimum supported RAM claim.

| Case | Canvas | References | Generation seconds | Peak process footprint (GiB) |
| --- | --- | ---: | ---: | ---: |
| camera-512-s42 | 512 × 512 | 0 | 47 | 12.69 |
| camera-1024-s42 | 1024 × 1024 | 0 | 116 | 26.77 |
| blue-camera-edit-512-s42 | 512 × 512 | 1 | 65 | 12.81 |
| two-camera-edit-768x512-s42 | 768 × 512 | 2 | 100 | 19.40 |

### Text-to-image

Both sizes produce a coherent brass camera on a walnut table with soft window
lighting. The 1024 case preserves detail without visible corruption. Seed 42
produces different compositions at different sizes. Small markings contain
invented or malformed text; these cases do not qualify typography accuracy.

![AI-generated brass camera at 512 pixels.](/qualification/flux3-klein-native-2026-10-15/camera-512-s42.png)

![AI-generated brass camera at 1024 pixels.](/qualification/flux3-klein-native-2026-10-15/camera-1024-s42.png)

### Single-reference edit

The 512-pixel brass camera is supplied with `--input`. The prompt changes the
metal body to cobalt blue while retaining the camera, black grip, lens, pose,
table, and lighting. The output passes the visual checks recorded in the receipt.

![AI-generated cobalt blue camera from the single-reference edit.](/qualification/flux3-klein-native-2026-10-15/blue-camera-edit-512-s42.png)

### Two-reference composition

Two ordered `--ref-image` arguments supply the brass original and the blue edit.
The prompt requests the brass camera on the left and the blue camera on the right.
The output passes the composition and color checks recorded in the receipt.
This exercises the multi-reference preprocessing and reference-coordinate path.

![AI-generated composition using the brass and blue camera references.](/qualification/flux3-klein-native-2026-10-15/two-camera-edit-768x512-s42.png)

## Reproduce the cases

Obtain gated checkpoint access, configure your Hugging Face token, and review
the FLUX Kommunity License before acknowledging it during installation:

```bash
mere.run model pull image-flux3-klein --accept-model-license
mere.run image generate --model image-flux3-klein \
  --prompt "A photograph of a brass camera on a walnut table, soft window light, clear realistic materials." \
  --width 512 --height 512 --steps 16 --cfg 1 --seed 42 --output camera-512-s42.png
```

Repeat with `--width 1024 --height 1024` for the larger case. For the single edit,
use `--input camera-512-s42.png` with the exact edit prompt in the JSON receipt.
For the composition, pass `--ref-image camera-512-s42.png` followed by
`--ref-image blue-camera-edit-512-s42.png`, and use a 768 × 512 canvas.
Add `--run-dir` with a different empty directory for each case to retain
settings, inputs, output hashes, and completion state.

## Validation boundary

Small independent Float32 and BF16 fixtures compare transformer behavior,
reference coordinates, flow-map times, and an Euler trajectory against the
pinned upstream source. Tiny VAE fixtures compare encoding, decoding, and PNG
channel order. Catalog, capability, install validation, input ordering, limits,
and scheduler tests cover the public workflow.

These four generated images qualify only bounded execution and visible prompt
adherence. They do not establish broad image quality, full-checkpoint numerical
parity with BFL, text-rendering accuracy, 4K operation, or ten-reference capacity.
The single seed and related camera images are not an independent quality corpus.
Native seeds are reproducible within this runtime and do not reproduce PyTorch's RNG.
Hardware execution on macOS 15, Linux/CUDA, and iOS remains unqualified by these
Mac runs. Hosted CI and application distribution are separate gates.
