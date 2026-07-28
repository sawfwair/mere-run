# Persistent World Runtime

A generated place you can walk around in. Send a camera move and get back a
video chunk; send another and it continues from where the last one ended — same
corridor, same light, same scene. The session is a navigation graph rather than
a one-way chain, so an exact inverse move replays the edge you arrived on
instead of inventing a new room on the way back.

It runs as a long-lived local HTTP server, loopback-first, on either of two
native Swift/MLX backends:

- `dreamx`: Wan 2.2 TI2V resources plus a converted DreamX causal checkpoint
- `cosmos3`: the official Cosmos3-Edge checkpoint with learned action
  conditioning and the `camera_pose` action domain

## Commands

| Command | What it does |
| --- | --- |
| `mere.run world serve` | Serve one warm native world-model session over HTTP. |

## Required models

DreamX combines two managed roots:

- `video-wan22-ti2v-5b-mlx` supplies the tokenizer, text encoder, VAE, and Wan
  TI2V base resources.
- `video-dreamx-world-5b-ar-mlx` supplies the converted DreamX causal weights
  for learned camera conditioning, block-causal attention, and persistent
  attention caches.

Both roots must already be installed or supplied as local directories. The
server never downloads model components while starting or serving requests.

Cosmos3 uses one complete managed snapshot:

- `video-cosmos3-edge-mlx` supplies the generation/understanding transformer,
  Wan VAE, tokenizer, scheduler, packed SigLIP2 vision encoder, and projector.

Install it explicitly after reviewing the OpenMDW-1.1 terms:

```bash
mere.run model pull video-cosmos3-edge-mlx --accept-model-license
```

## Start a session

```bash
mere.run world serve \
  --base-model video-wan22-ti2v-5b-mlx \
  --model video-dreamx-world-5b-ar-mlx \
  --state-directory ./world-state \
  --prepare
```

The default listener is `127.0.0.1:8791`. `--prepare` loads and warms the
models before the server accepts transitions. Without it, the runtime prepares
on the first explicit prepare or transition request.

Start an action-conditioned Cosmos3 session with:

```bash
mere.run world serve \
  --backend cosmos3 \
  --model video-cosmos3-edge-mlx \
  --state-directory ./vesper-world \
  --prepare
```

The default state directory is:

```text
~/Library/Application Support/MereRun/world-sessions/default
```

Binding to a non-loopback address requires `--api-key`. Authenticated requests
use `Authorization: Bearer <token>`.

## HTTP lifecycle

The server exposes:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Process health; does not load a model. |
| `GET` | `/v1/world/session` | Read phase, transition count, state IDs, and retained-state flags. |
| `POST` | `/v1/world/session/prepare` | Load and warm the session. |
| `POST` | `/v1/world/session/transitions` | Queue one transition and return `202 Accepted`. |
| `GET` | `/v1/world/jobs/{id}` | Poll job status, progress, receipt, or error. |
| `DELETE` | `/v1/world/jobs/{id}` | Request cancellation of an active transition. |
| `POST` | `/v1/world/session/reset` | Clear causal state, optionally seeding a new source image. |
| `POST` | `/v1/world/session/unload` | Release the warm runtime. |

Only one transition can be active in a session. A competing transition,
prepare, reset, or unload request fails explicitly instead of racing the active
generation.

## Queue a transition

The first transition requires `sourceImage`. Later transitions reuse the
previous terminal state unless a new source is supplied.

```bash
curl -X POST http://127.0.0.1:8791/v1/world/session/transitions \
  -H 'content-type: application/json' \
  -d '{
    "prompt": "continue forward through the same stone corridor",
    "camera": {
      "motion": "forward",
      "translationMeters": [0, 0, 1],
      "rotationDegrees": [0, 0, 0]
    },
    "sourceImage": "./corridor.png",
    "output": "./world-state/forward.mp4",
    "width": 512,
    "height": 320,
    "num_frames": 17,
    "steps": 40,
    "guidance_scale": 5,
    "shift": 5,
    "seed": 42,
    "fps": 24
  }'
```

Camera motion values are `hold`, `forward`, `backward`, `strafeLeft`,
`strafeRight`, `yawLeft`, `yawRight`, and `custom`. Translation and rotation
arrays are XYZ values in meters and degrees.

Omitted sampling fields use backend-specific recipes. DreamX defaults to
512x320, 17 frames, 40 steps, CFG 5, shift 5, seed 42, and 24 fps. Cosmos3
uses 320x176, 17 frames, 30 steps, CFG 1, shift 3, seed 0, and 30 fps. Every
interactive camera primitive uses NVIDIA's default 16-action cadence and the
same pinned per-action translation and rotation rates. This keeps each
generative chunk local enough to preserve scene identity. Explicit `num_frames`
still overrides the default. Each continued chunk uses `base_seed + chunk_index`,
matching NVIDIA's autoregressive sampling policy. Cosmos3 frame counts must be
`4n+1`.

The session is a small navigation graph rather than a write-only frame chain.
An exact inverse control at the same magnitude replays the current path edge in
reverse, pops it, and restores its source state, model-space action, and
generation depth. Repeated backtracking can therefore traverse multiple parent
edges without spending stochastic chunks or compounding avoidable scene drift.

For Cosmos3, the semantic camera request selects a normalized model-space
trajectory. Forward motion uses the exact 60x9 NVIDIA reference sequence;
rotations are emitted in the published column-based rotation-6D form. These
values are not meters or per-frame physical deltas. A caller that already has
a normalized trajectory can provide it directly as `model_space_actions`.
The server validates nine finite values per row, fits the sequence to the
requested frame count without renormalizing it, and reports `action_space`,
`action_domain`, and `model_space_actions` in the receipt (`raw_actions`
remains a compatibility alias).

For exact replay, add the trajectory beside the semantic camera intent:

```json
{
  "prompt": "continue through the same station",
  "camera": {
    "motion": "forward",
    "translationMeters": [0, 0, 1],
    "rotationDegrees": [0, 0, 0]
  },
  "model_space_actions": [
    [-1.0418892, -0.0325990, 0.1265574, 0.9999865, -0.0008900, -0.0051085, 0.0008920, 0.9999995, 0.0003920]
  ]
}
```

A production replay normally supplies all 60 rows. A shorter explicit list is
extended by repeating its final row; a longer list is truncated to
`num_frames - 1`.
The generated terminal frame seeds the next transition, so a second request
can omit `sourceImage` and continue moving through the same world. The runtime
follows NVIDIA's autoregressive recipe: it extracts the public terminal frame,
re-encodes that image at the start of the next chunk, and continues the
normalized translation trajectory from the prior terminal action. It does not
transplant the terminal latent from the end of one causal VAE timeline into
frame zero of another. The exact prior public state image is also placed at
frame zero of the next clip for a pixel-stable player handoff.

A completed receipt includes the previous and new state IDs, transition index,
MP4 output, terminal-frame image, camera request, conditioning mode, and seed.
Persisted files are the public handoff; callers never receive mutable MLX
tensors.

The session response also includes a top-level `backend` field whose value is
`dreamx` or `cosmos3`. Consumers should use it to select the matching
interactive camera recipe. Older servers can be identified from the session's
`conditioning_mode`.

## Reset or unload

```bash
curl -X POST http://127.0.0.1:8791/v1/world/session/reset \
  -H 'content-type: application/json' \
  -d '{"sourceImage":"./new-scene.png"}'

curl -X POST http://127.0.0.1:8791/v1/world/session/unload
```

Reset preserves the server but clears the transition chain. Unload releases
model resources and returns the session to a cold phase.

## Runtime entrypoints

### CLI and HTTP server

- `Sources/MereRunCLI/Commands/WorldCommand.swift`

### Runtime

- `Sources/MereRunCore/Wan2/Wan2WorldSession.swift`
- `Sources/MereRunCore/Wan2/Wan2CausalWorldGenerator.swift`
- `Sources/MereRunCore/Wan2/Wan2CameraConditioning.swift`
- `Sources/MereRunCore/Cosmos3/Cosmos3WorldSession.swift`
- `Sources/MereRunCore/Cosmos3/Cosmos3EdgeGenerator.swift`
- `Sources/MereRunCore/Cosmos3/Cosmos3Action.swift`
