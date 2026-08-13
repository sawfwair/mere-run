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
model math and generation schedule.
