# AP-BWE speech bandwidth-extension implementation report

## Decision

AP-BWE's 16 kHz to 48 kHz profile is admitted as a separate native audio
enhancement runtime. It is intentionally not presented as music mastering or
general audio super-resolution.

## Source and license admission

| Item | Frozen identity | License/admission result |
| --- | --- | --- |
| Source | `yxlu-0102/AP-BWE@751710f22404c27e5bcc983248f8b856a04b8422` | MIT |
| Official checkpoint | Google Drive file `1HYkD_5ha9GMrjzbTiSFiO1uQQwxXET15` | Upstream states weights use the same MIT license |
| Managed transport | `rsxdalv/AP-BWE@fa3f46d233cbc1d75cec9321188f86db627ba239` | Public mirror used only for transport |

The managed `g_16kto48k.zip` has 119,097,961 bytes and SHA-256
`305a05dcab7dc29ffba09d32692d7a34550fc8fbdf338013641ff5d39a3cb285`.
Those bytes match the official Drive checkpoint exactly. The source config,
code license, and weights-license files are also independently pinned by byte
count and SHA-256. Runtime readiness fails if any of the four changes.

## Native graph

The typed graph contains 162 float32 tensors and 29,760,515 scalars. It follows
the published 16→48 kHz profile:

- mono 16 kHz decoding and deterministic 3x band-limited interpolation;
- centered 1,024-point STFT with a periodic 320-sample Hann window padded to
  the FFT size and an 80-sample hop;
- magnitude and phase input convolutions from 513 bins to 512 channels;
- eight paired ConvNeXt blocks per stream using depthwise kernel-7 convolution,
  layer norm, exact GELU, 3x channel expansion, layer scale, and residuals;
- the upstream sequential magnitude/phase cross-add update;
- residual log-magnitude prediction, atan2 phase reconstruction, and native
  ISTFT back to the input duration.

PyTorch Conv1d tensors are transposed from `[out, in, kernel]` to MLX's
`[out, kernel, in]` layout. The loader rejects key, shape, dtype, tensor-count,
scalar-count, byte-count, and whole-file hash drift.

## Runtime boundary

`mere.run audio enhance` is a new top-level audio command because the same
surface can later host general audio super-resolution without conflating it
with music stem separation. This PR adds only AP-BWE; UniverSR remains a
separate stacked review.

The installed-model gate produces a real WAV artifact through the public
command. The manual checkpoint smoke additionally verifies a 48 kHz mono,
finite, non-silent output and reports energy above the original 8 kHz
narrowband boundary. These checks prove execution and artifact integrity, not
a perceptual benchmark.
