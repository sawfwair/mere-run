# UniverSR general-audio super-resolution implementation report

## Decision

The official general-audio UniverSR profile is admitted as a separate native
audio enhancement runtime. It supports speech, music, and sound effects with
effective 8, 12, 16, or 24 kHz input bandwidth and always writes 48 kHz mono
audio. It is described as audio super-resolution, not a complete mastering
chain.

## Source and license admission

| Item | Frozen identity | License/admission result |
| --- | --- | --- |
| Source | `woongzip1/UniverSR@26dc21c44e11f9f19e823f02b0d4641dd5ea5af2` | MIT |
| Official checkpoint | `woongzip1/universr-audio@1c3294844285af851b6ffa56cbde4e43cd41fc2b` | CC BY 4.0 |

The source-code and checkpoint licenses are intentionally recorded separately.
The 229,072,395-byte `pytorch_model.bin` has SHA-256
`eb99f98943cc32fa82226c2da14b32b5d890416070af4946acbce442b30dc20b`.
The 533-byte source `config.yaml` and 1,406-byte official model card are also
pinned by exact byte count and SHA-256. Runtime readiness fails if any of the
three artifacts changes.

## Native graph

The typed graph contains 394 float32 tensors and 57,231,302 scalars. It follows
the published general-audio profile:

- deterministic integer-ratio, windowed-sinc interpolation to 48 kHz;
- symmetric-Hann, 1,024-point complex STFT with a 512-sample hop;
- power-law magnitude compression with alpha 0.2, beta 1, and epsilon 1e-4;
- conditional ConvNeXt V2 U-Net dimensions `[96, 192, 384, 768]`, depths
  `[2, 2, 4, 2]`, 384 conditioning channels, and a 256-dimensional time path;
- rate-conditioned low-frequency observations and reconstruction of the
  remaining spectrum;
- deterministic noise seeding and Euler, midpoint, or RK4 ODE integration;
- inverse compression and native ISTFT cropped back to the source duration.

PyTorch Conv2d and ConvTranspose2d tensors are transformed into the MLX kernel
layouts only after the restricted state-dict reader verifies archive metadata,
key set, shape, dtype, tensor count, scalar count, byte count, and whole-file
hash.

## Runtime boundary

`mere.run audio enhance --model audio-enhance-universr-audio` selects this
runtime. Native 8/12/16/24 kHz files determine their effective rate directly;
bandwidth-limited audio stored in a 48 kHz container requires `--input-rate` so
the runtime does not guess. The published defaults are midpoint integration,
four steps, guidance scale 1.5, and seed 42, all recorded in the output
provenance manifest.

The installed-model gate and manual smoke use the public command to produce a
real WAV. Finite, non-silent output and spectral energy above the declared
input boundary prove the native path executed; they are not a perceptual
quality benchmark.
