# MiniMax Music 3 on M4 Max

This receipt records installed-checkpoint MiniMax Music 3 performance work on a
16-core CPU / 40-core GPU M4 Max MacBook Pro with 128 GB unified memory. Tests
used the immutable managed `MiniMaxAI/MiniMax-Music3` snapshot, staged loading,
44.1 kHz stereo PCM24 export, seed 37, guidance 1.7, and 30 flow steps unless a
row says otherwise.

## Matched 250-frame result

| Runtime | Wall time | Max resident | Relative speed |
| --- | ---: | ---: | ---: |
| mere.run 0.37.0 reference | 143.29 s | 18.67 GB | 1.00x |
| optimized BF16 | 105.46 s | 7.14 GB | 1.36x |
| Q8 autoregressive turbo | 52.66 s | 4.36 GB | 2.72x |
| Q4 autoregressive turbo | 51.40 s | 3.19 GB | 2.79x |

Q8 is the recommended turbo tier. Its first-step installed-checkpoint semantic
logits measured cosine similarity 0.99985 and retained 98 of the BF16 top 100
candidates. Q4 measured 0.99166 and retained 80 of 100; its small additional
speedup comes with a larger sampling-distribution change.

The BF16 optimized graph uses a 16,385-row reachable semantic head instead of
projecting the full 200,000-token vocabulary at every 25 Hz frame, fused QKV
and gate/up projections, incremental residual-depth KV caches, cached rotary
and zero conditioning, batched conditional/unconditional flow evaluation, and
fewer forced graph evaluations. `--performance-mode reference` preserves the
released graph for parity investigations.

## MLX 0.32.1 follow-on

The next pass rebased onto mere.run PR #297 after it landed. The tested chain
pins `mlx-swift` at `3e6df6d8163a8f212061d15739eeeec12d5b89e3`, its embedded
MLX at `b57bd7640f3f7c743b76a58478faaf1e8ee084f2`, and MLX core 0.32.1.

Stage profiling, a fixed-capacity autoregressive KV cache, and precomputed flow
input projections were added independently. These matched 100-frame Q8 runs
used the same prompt, checkpoint, seed, 30-step schedule, and one flow chunk:

| Graph | Total | Autoregressive | Flow | Flow transformer | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Refreshed dependency baseline | 30.90 s | 13.16 s | 16.88 s | 16.31 s | baseline |
| Static KV cache | 22.21 s | 7.25 s | 14.61 s | 14.29 s | 28.1% less total |
| Static KV + fused flow inputs | 19.24 s | 6.70 s | 12.21 s | 11.81 s | 37.7% less total |

The baseline and static-cache WAVs were byte-identical. Combining the flow
preprocess, residual, and input projections changes floating-point association;
its comparison against the unfused path measured 59.77 dB waveform PSNR and
0.999974 cosine similarity.

The public `--sampling-tier` presets keep the model and Euler solver fixed while
changing only the number of flow evaluations. `--steps` remains the exact
override. A clean 100-frame series from the final graph measured:

| Tier | Flow steps | Profiled total | Autoregressive | Flow | Wall time | Total speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `quality` | 30 | 20.03 s | 6.79 s | 12.89 s | 22.92 s | 1.00x |
| `fast` | 20 | 15.04 s | 5.52 s | 9.19 s | 15.13 s | 1.33x |
| `draft` | 16 | 13.41 s | 5.61 s | 7.47 s | 13.51 s | 1.49x |

Profiler JSON is opt-in with `--profile-output`. It synchronizes stage
boundaries so its internal totals are useful for hotspot attribution, while
unprofiled `/usr/bin/time` remains the wall-clock authority.

The finished reference path was also rendered from the optimized release
binary and compared with a clean 0.37.0 build. Both 250-frame PCM24 files had
the identical SHA-256
`6aee35f704ae5621f3bbc9b4b2ec04080931f8ec4a7db1e9099cf4ac4aaf5890`.
This comparison covers the seeded autoregressive trajectory, serial flow
guidance, vocoder, normalization, fade, and deterministic dither—not only
isolated tensor fixtures.

## Rejected experiments

- Whole-flow MLX compilation was waveform-identical but took 37.92 s versus
  35.63 s eager on a 100-frame, one-chunk, 30-step run: a 6.4% regression.
- Affine Q8 flow weights retained 0.99941 cosine on a fixed transformer fixture
  but increased a 251-frame end-to-end run to 60.10 s and did not reduce the
  run's peak resident memory. The production turbo modes therefore keep flow
  BF16.
- Serial flow CFG took 18.43 s versus 17.38 s for batched CFG on a matched
  100-frame, one-chunk, 30-step Q8 run. Batched CFG stays as the default.
- Compiling only the residual-depth graph was waveform-identical, but an ABBA
  series regressed from a 1.851 s eager median to a 2.157 s compiled median,
  16.5% slower. The fixed-capacity cache remains eager.
- A production-shape ConvRot W8A8 Metal oracle retained 0.999929 cosine and
  0.01192 relative RMSE, but took 57.64 ms versus 3.47 ms for BF16 GEMM, 16.6x
  slower on this M4 Max. The research test remains env-gated; the runtime does
  not ship a slower W8A8 path.
- A real, derived Q8 checkpoint repack occupied 9.01 GB and produced a
  byte-identical seeded WAV, but its ABBA load median was 3.883 s versus
  1.849 s from the released shards. Its first proof also reached 9.22 GB max
  resident versus 4.36 GB. The derived artifact and public repack surface were
  removed; source shards remain authoritative.

## ComfyUI comparison

The current ComfyUI implementation validates several architectural choices:
its autoregressive model uses a fixed KV cache, prunes the semantic projection
to 16,385 reachable rows, and has optional prefetch/graph support for the
residual-depth blocks. Its flow transformer still constructs the aligned input
inside each forward; mere.run now precomputes the invariant condition side and
fuses the latent-side projections. See ComfyUI's pinned
[autoregressive implementation](https://github.com/Comfy-Org/ComfyUI/blob/7fe8a6138504f90ff7be82f3babf416da32876b1/comfy/ldm/minimax_music/ar.py)
and [flow transformer](https://github.com/Comfy-Org/ComfyUI/blob/7fe8a6138504f90ff7be82f3babf416da32876b1/comfy/ldm/minimax_music/dit.py).

The public Comfy workflow offers pruned ConvRot INT8 checkpoints for the text
encoder and either FP16 or INT8 ConvRot flow weights. That is weight
quantization, not evidence that activation-quantized W8A8 is faster on M4-class
Apple GPUs. See the pinned
[workflow template](https://github.com/Comfy-Org/workflow_templates/blob/d9e66019b85da231b7c936ad9cb7ff08cec16557/templates/audio_minimax_music_3.json).

A frequently cited public timing log is also easy to misread: it generates
1,501 frames, approximately 60 seconds of audio, in about 170 seconds before
decode on a 24 GB AMD/HIP GPU. It is not a three-minute output and it is not an
MPS result. See the
[full ComfyUI run log](https://github.com/OrsoEric/HOWTO-ComfyUI/blob/602c8ef74cc297e530d8b905389a7b58e4f05bd1/logs/2026-08-14a-minmax3-music.md).

## Duration-floor correction

The model emits semantic frames at 25 Hz, but the vocoder emits whole
512-sample hops. At 44.1 kHz, 250 semantic frames decode to 440,832 samples, or
9.99619 seconds. A requested 10-second minimum now selects 251 frames and
decodes to 10.04263 seconds. This is why second-based floors must not use only
`duration * 25` truncation.

Long-form Q8 floor runs from the same release binary confirm both the corrected
duration contract and approximately linear scaling:

| Requested minimum | Frames / flow chunks | Wall time | Max resident | Peak footprint | Decoded duration |
| --- | ---: | ---: | ---: | ---: | ---: |
| 60 s | 1,501 / 15 | 457.54 s | 4.30 GB | 19.19 GB | 60.10485 s |
| 180 s | 4,501 / 45 | 1,413.81 s | 4.50 GB | 41.64 GB | 180.26812 s |

The 180-second receipt used native 44.1 kHz stereo PCM24 export. Its audio
SHA-256 was
`03726d0e39b0d6b36184d28862818a183cc80fddf1b5c7488bc81066a069289b`.

## Physical limit

The autoregressive stack emits one semantic frame plus seven residual codes per
25 Hz output frame. It is serial across time, so a three-minute song requires
at least 4,501 semantic iterations when a strict decoded-duration floor is
requested. No kernel can parallelize those dependent iterations without
changing the model.

The original BF16 8B semantic stack is primarily memory-bandwidth bound. An
idealized 26.2 GB of BF16 parameter traffic per output frame against the
[M4 Max's 546 GB/s memory bandwidth](https://www.apple.com/macbook-pro/specs/)
is already about 48 ms per frame before KV traffic, sampling, depth decode,
synchronization, or flow matching. Q8 reduces the dominant autoregressive
parameter traffic, after which the unquantized 4.86B flow transformer and its
30 serial Euler evaluations per chunk become the main tail. Long-song latency
therefore remains approximately linear in semantic frames and overlapping flow
chunks even after launch overhead is reduced.

For scale, halving that idealized autoregressive traffic for Q8 gives an
optimistic bandwidth-only floor near 24 ms per frame, or about 108 seconds for
4,501 frames. Streaming 4.86B BF16 flow parameters once per evaluation adds an
idealized roughly 24 seconds for 30 steps across 45 chunks, or about 13 seconds
at 16 steps. That places the absolute bandwidth-only floor around two minutes
before KV traffic, activations, kernel inefficiency, loads, sampling, overlap,
and vocoder work. It is a physical lower bound, not an achievable forecast.

Applying the measured `draft` flow ratio to the installed 180-second quality
receipt projects roughly 14–16 minutes on this machine. That projection is the
remaining practical ceiling from schedule reduction alone; materially crossing
it requires faster flow kernels or a changed/distilled model, not more launch
cleanup.

## Reproduction

```bash
/usr/bin/time -lp mere.run music generate \
  "128 BPM progressive deep-house instrumental, rubbery sub bass, shuffled hats, glassy minor-seventh stabs, hypnotic analog lead, polished club mix" \
  --model music-minimax-music3 \
  --instrumental \
  --duration 10 \
  --max-frames 250 \
  --sampling-tier quality \
  --seed 37 \
  --guidance-scale 1.7 \
  --memory-mode staged \
  --performance-mode q8 \
  --sample-rate 44100 \
  --export-format pcm24 \
  --normalize peak \
  --target-peak-db=-1.0 \
  --profile-output /tmp/minimax-music3-q8-profile.json \
  --output /tmp/minimax-music3-q8-10s.wav
```

Use `--minimum-duration 10` without `--max-frames` to benchmark the strict
decoded-duration floor instead; that intentionally generates 251 frames.
