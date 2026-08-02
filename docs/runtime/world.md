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

The macOS Studio app exposes the same resident server in Advanced → Operations
with typed backend, model, state-directory, warmup, host, port, and
authentication controls. API keys are injected with `MERERUN_API_KEY` so they
do not appear in the child process argument list.

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

Install it explicitly from NVIDIA's public OpenMDW-1.1 repository:

```bash
mere.run model pull video-cosmos3-edge-mlx
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
| `POST` | `/v1/world/session/source` | Decode an uploaded image body, normalize it to PNG, and reset the world origin. |
| `POST` | `/v1/world/session/transitions` | Queue one transition and return `202 Accepted`. |
| `POST` | `/v1/world/session/rollouts` | Queue a multi-block DreamX `action_seq` rollout and return `202 Accepted`. |
| `GET` | `/v1/world/jobs/{id}` | Poll job status, progress, receipt, or error. |
| `GET` | `/v1/world/jobs/{id}/media/chunks/{index}` | Stream a decoded DreamX block as soon as it appears in the job. |
| `GET` | `/v1/world/jobs/{id}/media/output` | Stream the completed rollout MP4. |
| `GET` | `/v1/world/jobs/{id}/media/terminal-frame` | Fetch the completed rollout's terminal PNG. |
| `DELETE` | `/v1/world/jobs/{id}` | Request cancellation of an active transition. |
| `POST` | `/v1/world/session/reset` | Clear causal state, optionally seeding a new source image. |
| `POST` | `/v1/world/session/unload` | Release the warm runtime. |

Only one transition can be active in a session. A competing transition,
prepare, reset, or unload request fails explicitly instead of racing the active
generation.

## Queue a DreamX rollout

The rollout surface preserves the released DreamX AR request dialect. Actions
can compose translation and rotation keys: `WASD` move and `IJKL` pitch/yaw.
`action_speed_list` contains relative duration weights, not physical speed.
Model-space motion uses DreamX's fixed default rate of `1.5`; the runtime does
not reinterpret UI meters or degrees as trajectory speed.

```bash
curl -X POST http://127.0.0.1:8791/v1/world/session/rollouts \
  -H 'content-type: application/json' \
  -d '{
    "prompt": "continue through the same coherent station",
    "action_seq": ["w", "wj", "wl"],
    "action_speed_list": [4, 6, 6],
    "source_image": "./station.png",
    "width": 1280,
    "height": 704,
    "num_output_frames": 63,
    "speed": 1.5,
    "seed": 42,
    "fps": 16
  }'
```

As in upstream DreamX, `num_output_frames` is the latent-frame count and must
be divisible by three. The public pixel-frame count is
`(num_output_frames - 1) * 4 + 1`: `21` produces `81` frames and `63` produces
`249`. The current upstream one-minute recipe ends at 252 latent / 1,005 pixel
frames (62.8 encoded seconds at 16 fps, including the initial frame), and
resolution may not exceed the official 1280x704 geometry. The defaults are
1280x704, 21 latent frames, speed 1.5, seed 42, and 16 fps.

A browser or app can seed the first frame without knowing a path on the runtime
host:

```bash
curl -X POST http://127.0.0.1:8791/v1/world/session/source \
  -H 'content-type: image/jpeg' \
  --data-binary @station.jpg
```

Polling the returned job exposes `chunks` while generation is still active.
Each entry carries its pixel-frame range and a same-origin `media_path`. For a
fresh causal session, the first three-latent block emits nine pixel frames and
each following block emits 12 new frames after removing the shared boundary.
When continuing an already-active causal session, the first streamed chunk also
contains the prior terminal boundary frame, so clients must trust each chunk's
reported `pixel_frame_count` instead of assuming nine. A completed job also
includes a `rollout_receipt` with the final output and terminal-frame media paths. Browser
clients may fetch and queue those MP4 chunks in order instead of waiting for
the full rollout. The server supplies CORS headers for local product clients;
Bearer authentication still applies when configured.

Session snapshots expose total `generated_latent_frame_count` separately from
the bounded `retained_latent_frame_count`. DreamX keeps only the three clean
latents required for the next VAE decode window while its causal attention cache
continues at the full generated position.

They also expose `current_world_pose`, `scene_memory_mode`,
`scene_memory_frame_count`, `scene_memory_retrieval_count`, and
`scene_memory_recycled_frame_count`. Rollout receipts report the terminal pose
and per-rollout memory counts so clients can distinguish an actual non-local
revisit from a normal causal continuation.

## Revisit-aware scene memory

DreamX model conditioning remains byte-for-byte compatible with the released
chunk-relative trajectory path. In parallel, the session composes those moves
into a persistent global camera pose and stores at most 96 predicted-clean
latents with their original causal positions. Retrieval requires the paper's
minimum temporal separation heuristic plus its revisit thresholds: at most 2
degrees of yaw error and 0.1 model-space translation distance. A retrieved
latent contributes an 8-percent residual anchor during denoising; movement
without a qualifying revisit receives no anchor.

The public mode name is `paper_reconstructed_revisit_anchor` on purpose. The
DreamX 1.0 paper describes training with packed memory/recent/target tokens,
camera-overlap retrieval, retained original temporal positions, and residual
recycling. The released upstream AR code and weights do not contain that
memory-trained pathway. This implementation reconstructs geometry retrieval
and a conservative inference-time anchor; it does not claim weight-level
parity with unreleased training code. Exact causal checkpoints include the
global pose and scene-memory index, so branching cannot leak memories from a
discarded future.

## Lock and branch an exact causal state

DreamX graph locks can capture the live model state, including every bounded
self-attention and projective-camera cache window, the retained clean latents,
the global causal position, and the public terminal frame:

```bash
curl -X POST http://127.0.0.1:8791/v1/world/session/checkpoints \
  -H 'content-type: application/json' \
  -d '{"name":"station gate"}'
```

Restore the returned `checkpoint_id` before generating a different branch:

```bash
curl -X POST \
  http://127.0.0.1:8791/v1/world/session/checkpoints/$CHECKPOINT_ID/restore
```

`GET /v1/world/session/checkpoints` lists live locks, and
`GET /v1/world/session/checkpoints/:id/media/frame` returns a lock's PNG for a
browser client. `DELETE /v1/world/session/checkpoints/:id` releases it. Locks
are exact within the current loaded session and intentionally cleared by
`reset` or `unload`; they are not mislabeled restart-persistent checkpoints.
Artifact numbering remains collision-free across restores, logical resets, and
existing files in the state directory, while the next rollout's
`previous_state_id` and causal token position fork from the locked state.

## Paper-aligned world evaluation

Run the captured 5-second, exact 63-latent/249-frame parity, approximately
30-second, out-and-back, translation/rotation, and rectangular-loop suite
against a prepared server:

```bash
python3 scripts/reference-parity/run_dreamx_world_eval.py \
  --base-url http://127.0.0.1:8791 \
  --source /absolute/path/to/world-origin.jpg \
  --prompt "preserve this playable world and its landmarks" \
  --output /tmp/dreamx-world-eval
```

The runner saves requests, jobs, receipts, MP4s, terminal PNGs, `ffprobe`
truth, pose closure, and scene-memory telemetry. Add deterministic PSNR/SSIM
scores with:

```bash
uv run --script scripts/reference-parity/score_dreamx_world_eval.py \
  --report /tmp/dreamx-world-eval/report.json
```

LPIPS, DINO-Sim, VPR-Sim, SP-Match, and CLIP-Video remain explicitly unscored
until their pinned learned-metric lanes are present. A passing pose threshold
alone is structural evidence, never a quality claim.

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
same pinned translation-magnitude envelope. Rotation intent is divided into a
constant per-frame pose delta. This keeps each generative chunk local enough to
preserve scene identity. Explicit `num_frames` still overrides the default.
Each continued chunk uses `base_seed + chunk_index`,
matching NVIDIA's autoregressive sampling policy. Cosmos3 frame counts must be
`4n+1`.

The session is a small navigation graph rather than a write-only frame chain.
An exact inverse control at the same magnitude replays the current path edge in
reverse, pops it, and restores its source state, model-space action, and
generation depth. Repeated backtracking can therefore traverse multiple parent
edges without spending stochastic chunks or compounding avoidable scene drift.

For Cosmos3, the semantic camera request selects a normalized model-space
trajectory. NVIDIA defines every 9D `camera_pose` row as the relative pose delta
between consecutive visual states: XYZ translation plus column-based
rotation-6D. The released 60x9 trajectory is retained byte-for-byte as a parity
fixture, but it is an arbitrary camera sample rather than a canonical forward
path. Semantic translation controls preserve that sample's per-step magnitude
envelope while placing each delta on the requested camera-relative axis;
stationary and yaw controls use zero translation. Request translation values
scale the normalized envelope and are not a calibrated physical displacement.
A caller that already has a normalized trajectory can provide it directly as
`model_space_actions`.
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
re-encodes that image at the start of the next chunk, and compiles a fresh
relative-delta trajectory for the next request. It does not accumulate the
prior chunk's final delta as though it were an absolute pose, and does not
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
