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

The locked MP4 predates the audio-VAE bias-loading correction documented in
the kernel lab. Its video remains the accepted visual and timing receipt, but
its hissy soundtrack is not an audio-quality reference. The corrected decoder
matches the released FP32 reference at `rel_l2=0.0000194`; a future long render
must replace the soundtrack acceptance receipt rather than reusing this file.

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
- execution: dense BF16, blockwise compiled, 768-query single-evaluation attention chunks
- acceleration: exact `quality` (no tail reuse)

| Phase | Baseline | Optimized exact |
| --- | ---: | ---: |
| Text encoding | 4.218 s | 4.078 s |
| Transformer preparation | 4.661 s | 4.510 s |
| Denoising | 953.485 s | **775.336 s (12:55.336)** |
| Video VAE decode | 289.586 s | 242.004 s |
| Audio decode | 0.873 s | 0.829 s |
| End to end | 1,253.425 s | **1,027.310 s (17:07.310)** |

The full model evaluation itself fell from 953.192 to 775.062 seconds, a
1.230x throughput improvement and 18.7% less wall time. End to end, the boundary
is 3:46.115 faster (18.0%). Peak reported Metal memory remained 48.77 GiB, so
this workload is compute-bound rather than blocked by unified-memory capacity.
The valid H.264/AAC output is 1344x768 at 24 fps with 32 kHz stereo audio and a
5.167-second duration. Its SHA-256 is
`f74e0d2ab0cf9a58233e954560a9aa6c05a1f905fd78914350a077b70f7e5848`—exactly
the same as the baseline artifact, proving the optimized execution did not
change model output.

The two-point artifact is intentionally noise-like and is not a visual quality
gate. Using its full-evaluation time only as an arithmetic bound, a 20-point
exact run now projects to about 4.16 hours end to end. The existing 417-block
`maximum` policy projects to about 1.87 hours. Those are projections, not
measured 768p render receipts; a near-30-minute quality result requires fewer
model/block evaluations rather than another small scheduling adjustment.

## VAE tile frontier

A later fresh-process sweep of the released BF16 video decoder changed the
production spatial tile from 256 to 320 pixels. At 832x480 and 124 frames, the
matched decode fell from 96.091 to 76.421 seconds (1.257x) while reported peak
Metal memory fell from 9.85 to 8.91 GiB. A 480-pixel arm reached 77.577 seconds
and lost. At the true 1344x768 shape, the accepted 320-pixel arm completed in
213.005 seconds at a 15.12 GiB reported peak; the 480-pixel arm crossed 252
seconds without completing and was stopped.

This is a non-quantized runtime change: the released decoder, precision,
causal tiling, and overlap blend remain intact. The larger tile performs fewer
overlapping tile evaluations and gives each evaluation more spatial context.
At true 1344x768 spatial geometry, an order-balanced 22-frame screen confirmed
that 320 pixels also beats the tempting overlap-count breakpoints: a hot-state
416-versus-320 pair measured 33.756 versus 32.321 seconds, and 432- and
496-pixel arms were materially slower. A two-temporal-chunk submission
prototype produced no repeatable speedup and raised reported peak Metal memory
from 13.49 to 21.40 GiB, so the accepted path keeps serial causal chunks.
The earlier end-to-end table remains the immutable artifact receipt; replacing
its 242.004-second decode with 213.005 seconds would be an arithmetic projection,
not a newly measured end-to-end result.

## Ten-second physical boundary

The first valid frame count beyond ten seconds is 243 frames (10.125 seconds),
which yields 72 video latent frames and 73,470 packed rows for the locked prompt.
The accepted exact attention schedule splits the 56 independent heads into
eight-head submissions with 640 query rows per kernel. An order-balanced
full-block gate measured 32.249 seconds for the previous 768-query/56-head
schedule and 25.173 seconds for the new schedule, a 1.281x speedup with
bit-identical BF16 output (`rel_l2=0`). That puts one exact 50-block model
evaluation near 21 minutes. Scaling the measured 124-frame VAE decode only as
an arithmetic estimate adds roughly seven minutes, placing the
single-evaluation 10-second boundary near 28 minutes.

A deterministic full-block dtype gate at these exact 73,470 rows measured
29.679 seconds in BF16 and 30.204 seconds in FP16. The FP16 output remained
numerically close for one block (`rel_l2=0.00243324`), but it was 1.7% slower;
the shorter 14,958- and 37,966-row screens were also slower. The runtime keeps
BF16 because changing precision does not unlock additional M4 throughput for
this workload.

That number is the physical one-evaluation boundary, not a quality render.
A 20-point exact schedule still performs 19 complete model evaluations and
therefore projects to roughly 6.6 hours of denoising before decode. The
417-block `maximum` schedule projects to roughly 2.9 hours before decode.
Reaching near 30 minutes with comparable visual quality consequently requires
a materially lower-evaluation model or sampler, such as an upstream distilled
checkpoint; the exact head-scheduling win does not remove the remaining
evaluation-count multiplier.
