# Persistent World Runtime

`mere.run world` runs a long-lived, local conditioned-video session. The current
public surface is a loopback-first HTTP server backed by Wan 2.2 TI2V resources
and a converted DreamX causal checkpoint.

## Public surface

- `mere.run world serve`

## Required models

The session combines two managed roots:

- `video-wan22-ti2v-5b-mlx` supplies the tokenizer, text encoder, VAE, and Wan
  TI2V base resources.
- `video-dreamx-world-5b-ar-mlx` supplies the converted DreamX causal weights
  for learned camera conditioning, block-causal attention, and persistent
  attention caches.

Both roots must already be installed or supplied as local directories. The
server never downloads model components while starting or serving requests.

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
    "height": 288,
    "seed": 42,
    "fps": 24
  }'
```

Camera motion values are `hold`, `forward`, `backward`, `strafeLeft`,
`strafeRight`, `yawLeft`, `yawRight`, and `custom`. Translation and rotation
arrays are XYZ values in meters and degrees.

A completed receipt includes the previous and new state IDs, transition index,
MP4 output, terminal-frame image, camera request, conditioning mode, and seed.
Persisted files are the public handoff; callers never receive mutable MLX
tensors.

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
