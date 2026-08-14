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

## Reproduction

```bash
/usr/bin/time -lp mere.run music generate \
  "128 BPM progressive deep-house instrumental, rubbery sub bass, shuffled hats, glassy minor-seventh stabs, hypnotic analog lead, polished club mix" \
  --model music-minimax-music3 \
  --instrumental \
  --duration 10 \
  --max-frames 250 \
  --steps 30 \
  --seed 37 \
  --guidance-scale 1.7 \
  --memory-mode staged \
  --performance-mode q8 \
  --sample-rate 44100 \
  --export-format pcm24 \
  --normalize peak \
  --target-peak-db=-1.0 \
  --output /tmp/minimax-music3-q8-10s.wav
```

Use `--minimum-duration 10` without `--max-frames` to benchmark the strict
decoded-duration floor instead; that intentionally generates 251 frames.
