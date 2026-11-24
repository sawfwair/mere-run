# Hand off World captures and 3D assets

Use VFX Tools to move model output into an editing or 3D workflow with the
files and settings needed to inspect it. World handoff packages an ordered
camera sequence as media. IDOL handoff packages a Gaussian splat from one
image. These workflows produce different assets and require different models.

## Before you begin

Follow [Set up companion workflows](./plugin-workflows.md), install VFX Tools,
and check the selected native runtime:

```bash
mere.run plugin install mere-vfx-tools --yes
mere.run plugin doctor mere-vfx-tools
```

For World, use a compatible DreamX or Cosmos3 checkpoint and local server. For
IDOL, obtain the required publisher access and prepare local IDOL, DINOv3
ViT-H+/16, and FLUX.2-dev VAE directories. Review source-image rights and all
checkpoint terms. See [World setup](../runtime/world.md) and
[IDOL setup](../runtime/vision.md#idol-object-splats).

## Package a World camera sequence

1. Start a dedicated, empty World session. In Studio, open **Server → World**,
   or run the server in a separate terminal from the capture directory:

   ```bash
   mere.run world serve --backend cosmos3 --state-directory "$PWD/world-state"
   ```

2. Save the [lighthouse request example](https://github.com/sawfwair/mere-run-plugins/blob/main/examples/world/lighthouse-request.json)
   as `world-request.json`. Set `inputs.image` to your still and
   `options.stateDirectory` to the server's state directory. If your local
   server requires a key, provide `MERE_WORLD_API_KEY` in the capture shell.

3. Preview without contacting the server, then capture in a fresh directory:

   ```bash
   mere.run plugin run mere-vfx-tools -- world-handoff      --request-json ./world-request.json --output-dir ./world-preview --dry-run
   mere.run plugin run mere-vfx-tools -- world-handoff      --request-json ./world-request.json --output-dir ./world-edit
   ```

The `world/` folder includes numbered MP4s and PNGs. `world/handoff.json`
records camera instructions, effective sampling settings, seeds per move,
state sequence, and media hashes. `continuationFrame` identifies the last PNG
that can start another session. DreamX's exact inference checkpoints remain
part of the live server session; the media folder isn't a saved server.

Play each clip and verify camera order, frame counts, seams, and the final
still. Move a copy of `world/` to the editor and check that its files open
without the server. After a failed capture, inspect `world/capture.json` for
completed clips and job IDs. Use a fresh dedicated session and output directory
for another capture.

## Export a Gaussian splat from one image

Save this as `idol-request.json`, replacing the absolute checkpoint paths:

```json
{
  "inputs": {"image": "/absolute/path/to/object.png"},
  "options": {
    "reconstructionMode": "native-idol",
    "model": "/absolute/path/to/idol",
    "dino": "/absolute/path/to/dinov3",
    "vae": "/absolute/path/to/FLUX.2-dev/vae",
    "inputMode": "synthetic",
    "numGaussians": 262144,
    "seed": 42
  }
}
```

Run the handoff in a fresh directory:

```bash
mere.run plugin run mere-vfx-tools -- image-to-3d   --request-json ./idol-request.json --output-dir ./idol-review
```

The plugin checks the native PLY, prepared image, and camera against their
manifest. Open `native-idol/object.ply` in a Gaussian-splat viewer. This is a
splat, not a rigged mesh. Inspect the back, silhouette, scale, and missing
surfaces before using it in a scene.

`image-to-3d.json` records checkpoint identities, settings, conventions, and
hashes. The splat is y-up in normalized object-space units; the camera retains
its z-up convention, with angular values in radians. Preserve those conventions
when importing. See the [VFX Tools reference](https://plugins-docs.mere.run/plugins/vfx-tools)
for the complete handoff contract.
