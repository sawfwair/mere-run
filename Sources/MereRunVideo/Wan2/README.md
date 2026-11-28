# Wan2 native runtime

This directory owns the native Swift/MLX Wan2.2 TI2V-5B runtime. The managed
model root is `video-wan22-ti2v-5b-mlx` and contains:

- `config.json`: typed Wan2.2 TI2V-5B architecture and inference defaults.
- `model.safetensors`: single 5B diffusion transformer in MLX layout.
- `t5_encoder.safetensors`: UMT5-XXL encoder weights in MLX layout.
- `tokenizer.json`: local UMT5 tokenizer; inference must not fetch it at runtime.
- `vae.safetensors`: float32 Wan2.2 48-channel VAE encoder and decoder weights.

The model is derived from the official `Wan-AI/Wan2.2-TI2V-5B` checkpoint.
`Wan2Resources` pins both the official source revision and the managed converted
snapshot used by `mere.run model pull`.

## Runtime boundaries

`Wan2TI2VGenerator` is a reusable warm runtime. Reusing one instance retains the
tokenizer, UMT5 encoder, diffusion transformer, VAE, prompt embeddings, and the
latest terminal-frame latent. `Wan2WorldSession` owns that runtime behind an
actor and exposes serial world transitions without passing mutable MLX tensors
to callers or launching a process per move.

World transitions carry a stable camera-control record with XYZ translation and
rotation. The base TI2V checkpoint applies that control through the motion
prompt and first-frame conditioning (`textAndFirstFrame`). When a pinned DreamX
camera-adapter file is supplied, the same request is compiled into relative
camera extrinsics and intrinsics and the session reports
`projectiveCameraLatents`. Supplying the converted
`video-dreamx-world-5b-ar-mlx` checkpoint selects the released DreamX causal
transformer, persistent block-causal and cross-attention caches, four-pass
AR-forcing schedule, three-latent-frame chunks, chunk-relative PRoPE camera
conditioning, and `causalCameraLatents` session mode.

DreamX camera conditioning is a learned parallel self-attention branch in every
Wan transformer block. It applies PRoPE projection matrices to Q/K/V, performs
attention in camera-projective space, applies the inverse output transform, and
joins the base self-attention residual. It is not prompt steering.

The causal session writes MP4 chunks, a combined rollout MP4, and one terminal
PNG while keeping only the three latent frames needed by the next incremental
VAE decode plus the rolling 12-frame attention window in memory. A separate
monotonic latent-frame position keeps cache appends correct after that decode
window is compacted. The PNG is an observable artifact, not the chaining path.
The official DreamX color-statistics pass runs after VAE decode to stabilize
each generated chunk against the preceding world state.

The runtime also tracks a global world-to-camera chain independently of the
released model's chunk-relative PRoPE inputs. A bounded scene-memory index keeps
predicted-clean latents with those global poses. Once a candidate is separated
by at least `max(3, floor(0.2 * currentFrame))` latent frames, a revisit within
2 degrees yaw and 0.1 model-space distance can supply a weak clean-latent
residual anchor. A path returning within 0.01 degrees and 0.001 model-space
distance restores its first clean latent exactly instead of feeding causal
appearance drift back into the world. The first origin entry is retained even
when the bounded index evicts older non-origin views. Ordinary forward motion
is unchanged because it retrieves no candidate. This is an explicitly labeled inference reconstruction of the
DreamX 1.0 paper's geometry retrieval and residual-recycling idea. The released
AR repository and checkpoint do not include the paper's memory-trained packed
`[memory | recent | target]` pathway, so the native runtime does not claim that
unreleased trained feature.

`mere.run world serve` owns one such session in a long-lived process. Its
loopback HTTP API exposes cold/warm snapshots, asynchronous transition jobs,
multi-block `action_seq` rollouts, decoded block media, pollable denoising
progress, cancellation, reset, unload, receipts, and opaque state IDs:

```bash
mere.run world serve --prepare --scene-memory-strength 0.08
curl http://127.0.0.1:8791/v1/world/session
curl -X POST http://127.0.0.1:8791/v1/world/session/transitions \
  -H 'Content-Type: application/json' \
  --data '{
    "prompt":"A continuous first-person view of the same corridor.",
    "camera":{"motion":"forward","translation_meters":[0,0,0.25],"rotation_degrees":[0,0,0]},
    "source_image":"/absolute/path/to/seed.png",
    "width":512,"height":288
  }'

curl -X POST http://127.0.0.1:8791/v1/world/session/rollouts \
  -H 'Content-Type: application/json' \
  --data '{
    "prompt":"A continuous first-person view of the same corridor.",
    "action_seq":["w","wj","wl"],
    "action_speed_list":[4,6,6],
    "num_output_frames":63
  }'
```

The rollout API uses upstream semantics: `action_speed_list` weights action
duration, composed `WASD+IJKL` keys move and rotate together, the default
model-space rate is 1.5, and `num_output_frames` counts latent frames. Thus 63
latent frames produce 249 public frames in 21 three-latent causal blocks.
Source images follow DreamX's exact fixed-dimension Pillow bilinear resize
contract rather than an aspect-preserving crop; the native separable path is
byte-gated against both released downsample and upsample results.

Revisit memory defaults to a conservative `0.08` clean-latent recycling
strength. `--scene-memory-strength`, `--scene-memory-max-frames`,
`--scene-memory-minimum-gap`, `--scene-memory-max-yaw`, and
`--scene-memory-max-translation` tune ordinary retrieval. The
`--scene-memory-exact-yaw` and `--scene-memory-exact-translation` tolerances
make the exact-return policy reproducible;
`--disable-scene-memory` provides the matched no-memory control.

Non-loopback binds require `--api-key`. The converted causal model is a
local-only managed artifact because the public upstream checkpoint is FP32 and
does not match the streamed BF16 MLX layout; `mere.run` will not silently pull
that incompatible file. The base Wan model is declared as its companion for
UMT5, tokenizer, and VAE resources.

## Verified parity

The real-checkpoint harnesses compare native Swift/MLX against independent
upstream implementations:

- UMT5 output against Wan's PyTorch encoder.
- One-block and full 30-block transformer output against the public MLX Wan
  implementation.
- VAE image latents against Wan's official PyTorch VAE.
- Two consecutive world transitions with warm model reuse and terminal-latent
  chaining.
- DreamX trajectory, latent-frame alignment, intrinsics/extrinsics, and PRoPE
  Q/K/V/output transforms against the upstream PyTorch implementation.
- The pinned upstream AR composed-action and chunk-relative camera fixture runs
  in the default test suite rather than requiring an opt-in environment file.
- A real released block-0 camera-attention output against native Swift/MLX and a
  full 30-block projective-camera transition using the extracted 300-tensor
  adapter.
- The released DreamX AR checkpoint across initial, same-block recomputation,
  and appended cached blocks. Measured cosine similarities are 0.99943,
  0.99793, and 0.99988 respectively.
- A real two-move causal session with a 1.5-2.2 RGB-level encoded boundary
  difference, plus a clean 512x288 nine-frame quality candidate.

Tiny renders remain a responsiveness tier rather than the quality tier. On the
reference M4 Max / 128 GB host, a warm 384x224 three-latent move produced nine
frames in 6.11 seconds, with the first and only block at 5.91 seconds. A warm
640x352 six-latent composed forward+yaw move produced 21 frames in 29.31
seconds, with its first block at 13.40 seconds. These are measured responsive
local generation results, not realtime playback claims; product clients should
show immediate input state, stream each completed causal block, and use the
atlas for already-proved routes.

Live graph locks use `Wan2CausalWorldCheckpoint`, which preserves the bounded
attention windows, retained clean latents, prompt cross-attention state, and
global causal position by immutable MLX array reference. The global pose,
bounded scene-memory entries, and memory telemetry are part of the same lock.
Restoring a lock forks
the exact in-memory model state; reset and unload release all locks. The HTTP
surface exposes explicit create, list, restore, frame-media, and discard
operations so product clients can distinguish an exact live lock from a
terminal-PNG restart seam.

`scripts/reference-parity/run_dreamx_world_eval.py` drives the checked-in
`dreamx_eval_suite.json` through a live server. It captures every job receipt,
MP4 and terminal frame, verifies encoded frame counts/geometry with `ffprobe`,
and applies the paper's revisit pose gate. Run
`uv run --script scripts/reference-parity/score_dreamx_world_eval.py --report
<report.json>` for pinned PSNR/SSIM scoring. The learned lane is
`score_dreamx_world_eval_learned.py`; it requires pinned evaluation-only
checkouts of MutualVPR and LightGlue plus the hash-pinned MutualVPR checkpoint,
then scores matched-baseline LPIPS, DINO-Sim, VPR-Sim, SP-Match, and
CLIP-Video. Pose closure is never presented as visual parity.
