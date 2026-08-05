# MiniMax-H3 BF16 on M4 Max

This receipt records a real local end-to-end MiniMax-H3 FL2VA generation on an
M4 Max with 128 GiB unified memory. It is an artifact benchmark, not a
transformer-only extrapolation.

## Locked workload

- checkpoint: managed `video-minimax-h3-fl2va-bf16-mlx`
- mode: resident BF16 transformer, `maximum` acceleration
- geometry: 832x480, 124 frames, 24 fps
- schedule: 20 points / 19 model evaluations
- seed: `20260804`
- packed rows: 14,928
- execution: blockwise compiled
- policy: six complete 50-block evaluations and thirteen nine-block cached-tail
  evaluations, with native start, refresh, and endpoint evaluations

The command is reproducible with the repository benchmark harness:

```bash
scripts/h3-end-to-end-benchmark.sh target-maximum
```

## Result

| Phase | Time |
| --- | ---: |
| Text encoding | 3.847 s |
| Transformer preparation | 2.359 s |
| Denoising | 1,526.533 s (25:26.533) |
| Video VAE decode | 119.544 s |
| Audio decode | 0.892 s |
| End to end | **1,653.711 s (27:33.711)** |

The six full evaluations averaged 184.268 seconds. The thirteen cached-tail
evaluations averaged 32.372 seconds with a 32.191-second median. Peak reported
Metal memory was 43.41 GiB and active memory remained approximately 38.24 GiB.

The output is an H.264/AAC MP4 at 832x480, 24 fps, with synchronized 32 kHz
stereo audio. Its SHA-256 is
`270544693662769fd8f90971d1ccabce3405fe00ff21885f96b9b0e218cb921f`.
Visual acceptance confirmed a coherent caped figure and yellow umbrella,
wet-street reflections, levitating bus, and luminous dragon motion without the
spatial lattice rejected in earlier aggressive cache experiments.

## Duration boundary

MiniMax-H3 frame counts follow `17*n+5`. The locked 124-frame artifact is
5.167 seconds at 24 fps. It proves the near-30-minute target for this workload;
it is not a ten-second runtime claim. The first valid frame count beyond ten
seconds is 243 frames (10.125 seconds), which requires a separate measured
receipt.

## True-768 one-evaluation boundary

A separate release-mode probe locks the physical 768-line target without
pretending that a two-point sampler is a quality result:

```bash
scripts/h3-end-to-end-benchmark.sh probe-768
```

- geometry: 1344x768, 124 frames, 24 fps
- schedule: two points / one complete 50-block model evaluation
- packed rows: 37,794
- execution: dense BF16, blockwise compiled, 1,024-query attention chunks
- acceleration: exact `quality` (no tail reuse)

| Phase | Time |
| --- | ---: |
| Text encoding | 4.218 s |
| Transformer preparation | 4.661 s |
| Denoising | **953.485 s (15:53.485)** |
| Video VAE decode | 289.586 s (4:49.586) |
| Audio decode | 0.873 s |
| End to end | **1,253.425 s (20:53.425)** |

The full model evaluation itself measured 953.192 seconds. Peak reported Metal
memory was 48.77 GiB, so this workload is compute-bound rather than blocked by
unified-memory capacity. The valid H.264/AAC output is 1344x768 at 24 fps with
32 kHz stereo audio and a 5.167-second duration. Its SHA-256 is
`f74e0d2ab0cf9a58233e954560a9aa6c05a1f905fd78914350a077b70f7e5848`.

The two-point artifact is intentionally noise-like and is not a visual quality
gate. Using its full-evaluation time only as an arithmetic bound, a 20-point
exact run projects to about 5.11 hours end to end. The existing 417-block
`maximum` policy projects to about 2.29 hours. Those are projections, not
measured 768p render receipts; a near-30-minute quality result requires fewer
model/block evaluations rather than another small scheduling adjustment.
