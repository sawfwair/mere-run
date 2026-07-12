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

The causal session writes one MP4 and one terminal PNG per transition while
keeping the complete latent history and rolling 12-frame attention window in
memory. The PNG is an observable artifact, not the chaining path. Its official
DreamX color-statistics pass runs after VAE decode to stabilize each generated
chunk against the preceding world state.

`mere.run world serve` owns one such session in a long-lived process. Its
loopback HTTP API exposes cold/warm snapshots, asynchronous transition jobs,
pollable denoising progress, cancellation, reset, unload, receipts, and opaque
state IDs:

```bash
mere.run world serve --prepare
curl http://127.0.0.1:8791/v1/world/session
curl -X POST http://127.0.0.1:8791/v1/world/session/transitions \
  -H 'Content-Type: application/json' \
  --data '{
    "prompt":"A continuous first-person view of the same corridor.",
    "camera":{"motion":"forward","translation_meters":[0,0,0.25],"rotation_degrees":[0,0,0]},
    "source_image":"/absolute/path/to/seed.png",
    "width":512,"height":288
  }'
```

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
- A real released block-0 camera-attention output against native Swift/MLX and a
  full 30-block projective-camera transition using the extracted 300-tensor
  adapter.
- The released DreamX AR checkpoint across initial, same-block recomputation,
  and appended cached blocks. Measured cosine similarities are 0.99943,
  0.99793, and 0.99988 respectively.
- A real two-move causal session with a 1.5-2.2 RGB-level encoded boundary
  difference, plus a clean 512x288 nine-frame quality candidate.

Tiny 128-256 pixel renders are structural tests only. Quality acceptance starts
at 512x288. On the reference 128 GB Apple Silicon host, one 512x288 move took
1,012.6 seconds: model passes began at 34.7s, 210.9s, 388.2s, 564.9s, and
739.4s; decode began at 917.7s. Product navigation must therefore use warm
sessions, queued graph edges, speculative neighboring moves, and cached clips
instead of presenting quality generation as an immediate keypress response.
