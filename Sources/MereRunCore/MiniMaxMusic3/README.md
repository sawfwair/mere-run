# MiniMax Music 3

This directory contains the native Swift/MLX inference path for MiniMax Music
3. The runtime follows the upstream checkpoint contract in four stages:

1. a Qwen3 global language model and local RVQ depth decoder generate eight
   audio codes plus the hidden-state conditioning for every 25 Hz frame;
2. the condition encoder aligns those frame states to the 44.1 kHz latent
   timeline;
3. a 1D flow-matching transformer denoises overlapping 200-frame windows;
4. the DAC-style vocoder decodes and stitches stereo waveform chunks.

Keep the prompt tokens, code offsets, chunk overlap, Euler schedule, and
weight-name mapping in parity with the pinned upstream revision declared by
`MiniMaxMusic3Resources`.

The default `staged` loading strategy releases each stage before loading the
next one and clears the MLX cache between stages. `resident` loads the complete
stack once for lower repeated-request latency. Both strategies execute the same
model math and generation schedule. Optimized autoregressive generation also
returns completed variable-length attention buffers to MLX every 64 frames;
live model weights, KV state, and generated conditioning remain resident.

The optimized language-model head retains only EOS plus the 16,384 reachable
semantic rows, but sampling restores those logits to their original positions
in a masked full-vocabulary view before the categorical draw. The residual
depth decoder intentionally recomputes its eight-token prefix with separate
projections. Cached or fused depth evaluation is numerically close, but a late
codebook can cross a categorical boundary and change every later semantic
frame. Seeded code trajectories and lyric transcription, not logit cosine
alone, are the admission checks for changing either path.

The released flow and seed recipes remain `sequential` and `legacy`.
`overlap-average` is an opt-in whole-song flow experiment that averages
overlapping window velocities at each Euler step, then uses bounded DAV decode
with retained-center stitching. `stage-separated-v1` independently derives the
autoregressive and flow random streams. The pipeline validates finite output,
signal level, peak, and stereo integrity before returning audio.

On an M4 Max with 128 GB unified memory and MLX 0.32.1, the PocketAiHub-style
direct INT8 ConvRot projection was slower than the converted BF16 dense path at
every measured 2,048-by-2,048 shape: 0.757 vs 0.285 ms at 100 rows, 1.048 vs
0.555 ms at 689 rows, and 1.772 vs 1.108 ms at 1,380 rows. Output cosine was
0.999991 or better. ConvRot therefore remains a tested research oracle rather
than a production execution path.

A real 251-frame Q8/draft checkpoint run measured sequential flow/vocoder at
48.08/4.91 seconds and overlap-average at 46.52/1.08 seconds. End-to-end totals
were 82.14 and 86.67 seconds respectively because the second run's common
autoregressive stage thermalized by roughly 10.5 seconds; compare isolated
profile stages, not that single noisy total, when evaluating the flow strategy.

A matched forced 3,001-frame (120-second) Q8/draft run measured sequential and
overlap-average flow at 454.31 and 464.38 seconds respectively, so the
whole-song experiment remains an opt-in continuity recipe rather than a speed
default. Sequential and overlap-average decode took 15.95 and 9.43 seconds.
Sample jumps at the exact flow and DAV boundaries stayed below each render's
global 99.99th-percentile adjacent-sample difference.

The same 3,001-frame sequential run exposed duration-scaled MLX buffer-cache
growth that process RSS did not report: `/usr/bin/time -l` measured a 110.20 GB
peak physical footprint. Returning completed autoregressive buffers every 64
frames reduced the matched peak to 23.78 GB (a 78.4% reduction) with zero swaps. The
autoregressive stage changed from 144.75 to 151.05 seconds, and the complete
float32 WAV remained byte-identical at SHA-256
`5ec23504131e7b74cef86f368d1a9164f9ac96d071d9c0f25615c5525fc559be`.
