# MiniMax-H3 BF16 on M4 Max

This receipt records a real local end-to-end MiniMax-H3 FL2VA generation on an
M4 Max with 128 GiB unified memory. It is an artifact benchmark, not a
transformer-only extrapolation.

## Locked workload

- **Checkpoint:** Managed `video-minimax-h3-fl2va-bf16-mlx`.
- **Mode:** Resident BF16 transformer, historical scheduled-tail `maximum`.
- **Geometry:** 832 x 480 pixels, 124 frames, 24 frames per second.
- **Schedule:** 20 points and 19 model evaluations.
- **Seed:** `20260804`.
- **Packed rows:** 14,928.
- **Execution:** Blockwise compiled.
- **Policy:** Six complete 50-block evaluations and thirteen nine-block cached-tail
  evaluations, with native start, refresh, and endpoint evaluations

This historical artifact is reproducible on the runtime with
`MERERUN_H3_CACHE_STRATEGY=scheduled-tail`. Production `maximum` selects
the adaptive first-block policy in the adaptive cache comparison.

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

The output is an H.264/AAC MP4 at 832 x 480 pixels and 24 frames per second,
with synchronized 32 kHz
stereo audio. Its SHA-256 is
`270544693662769fd8f90971d1ccabce3405fe00ff21885f96b9b0e218cb921f`.
Visual acceptance confirmed a coherent caped figure and yellow umbrella,
wet-street reflections, levitating bus, and luminous dragon motion without the
spatial lattice rejected in earlier aggressive cache experiments.

## Adaptive first-block cache comparison

A later release-build A/B held checkpoint, prompt, seed, geometry, schedule,
weight residency, and packed layout fixed while changing only the cache
decision. This practical proxy used resident BF16 at 416 x 256 pixels, 107 frames,
20 schedule points and 19 evaluations, and 3,768 packed rows.

| Policy | Full and cached evaluations | Executed blocks | Denoising | End to end |
| --- | ---: | ---: | ---: | ---: |
| Scheduled-tail baseline | 6 / 13 | 417 | 222.220 s | 257.814 s |
| Adaptive first-block `maximum` | 7 / 12 | 362 | **163.953 s** | **194.174 s** |

The adaptive policy reduced denoising time by 26.2%, end-to-end time by 24.7%,
and transformer block work by 13.2% relative to the fixed scheduled-tail
baseline. It was run third rather than first, and macOS reported no thermal or
performance warning at its start. A deliberately conservative 0.12/0.18
calibration was also measured and rejected: it admitted only 7 cache hits,
executed 607 blocks, and took 328.364 seconds to denoise.

Both accepted MP4s contain H.264 416 x 256 video at 24 frames per second and AAC 32 kHz stereo
audio. Contact-sheet inspection confirmed the same coherent bus-lift-to-dragon
trajectory without lattice collapse. The baseline SHA-256 is
`02fb184e6cc8213197e52ecaf4b6b344eb4b292b6233213605bc36d5d50cb66b`; the
adaptive output SHA-256 is
`97409a6aada0e21aca2c6917adad582f86c2d82bfaee05195da79303174920dc`.

## Resident sliding-window and frame-injection receipts

A real two-window FL2VA generation then exercised the long-form path with the
same managed BF16 checkpoint. It generated 56 frames at 416 x 256 pixels using 39-frame
windows, an 18-frame video-and-audio overlap, nine schedule points per window,
and the adaptive `maximum` policy. The second window reused the resident Qwen
conditioner and transformer: both model-load phases measured 0.000 seconds.

| Window | Full and cached evaluations | Executed blocks | Denoising | Generation |
| --- | ---: | ---: | ---: | ---: |
| First | 5 / 3 | 253 | 43.268 s | 67.796 s |
| Continuation | 5 / 3 | 253 | 51.273 s | 63.421 s |

The resulting H.264/AAC MP4 contains exactly 56 frames at 24 frames per second and 32 kHz
stereo audio, both 2.333 seconds long. At the window boundary, frame-to-frame
RGB mean absolute difference was 4.128, below every adjacent comparison in the
local inspection range (6.418–7.609). Audio sample jumps at the same boundary
were 0.00954 and 0.00616, below the local channel p95 deltas of 0.02284 and
0.02128. Contact-sheet and spectrogram inspection found neither a visual cut
nor an audio-click impulse. The artifact SHA-256 is
`ac51d8eb9623e9f7ea996204a415370058241f912aadffc058dde3a2525f25de`.

A separate release-mode FL2VA generation injected an exact image at output
frame 11 of a 22-frame, 416 x 256 shot. The Qwen multimodal presentation and
video-VAE condition path both consumed the frame; the run completed in 46.944
seconds with 28.532 seconds of denoising. The valid H.264/AAC output has
SHA-256
`cea064abd0c99c0350629f8207e0e1055a2b96f0b2e2f9b76bda2effb17cf940`.
Contact-sheet inspection confirmed a coherent approach to and departure from
the injected composition without collapse.

These receipts prove FL2VA execution. Ref2VA now has a separate real-checkpoint
validation receipt in
[`minimax-h3-ref2va-mlx-8bit.md`](./minimax-h3-ref2va-mlx-8bit.md), in addition
to the typed continuation-layout coverage here.

The locked MP4 predates the audio-VAE bias-loading correction documented in
the kernel lab. Its video remains the accepted visual and timing receipt, but
its hissy soundtrack is not an audio-quality reference. The corrected decoder
matches the released FP32 reference at `rel_l2=0.0000194`; a later long render
must replace the soundtrack acceptance receipt instead of reusing this file.

## Duration boundary

MiniMax-H3 frame counts follow `17*n+5`. The locked 124-frame artifact is
5.167 seconds at 24 frames per second. It proves the near-30-minute target for this workload;
it is not a ten-second runtime claim. The first valid frame count beyond ten
seconds is 243 frames (10.125 seconds), which requires a separate measured
receipt.

## Turbo four-evaluation receipt

The optional `minimax-h3-turbo-4step` adapter was validated separately with a
real release-mode generation rather than extrapolating its tiny-shape smoke
test:

- **Checkpoint:** Managed `video-minimax-h3-fl2va-bf16-mlx`.
- **Adapter:** Checksum-pinned `minimax-h3-turbo-4step`, strength `0.9`.
- **Geometry:** 1280 x 768 pixels, 124 frames, 24 frames per second.
- **Schedule:** Five points and four complete model evaluations.
- **Seed:** `20260804`.
- **Packed rows:** 36,047.
- **Execution:** Dense BF16, blockwise compiled, exact `quality` acceleration.

| Phase | Time |
| --- | ---: |
| Text encoding | 4.066 s |
| Transformer and adapter preparation | 4.231 s |
| Denoising | 3,373.421 s (56:13.421) |
| Video VAE decode | 272.737 s (4:32.737) |
| Audio decode | 0.987 s |
| Native generation total | **3,656.608 s (60:56.608)** |
| Process wall time | **3,659.420 s (60:59.420)** |

The four evaluations measured 842.253, 829.334, 825.381, and 876.346 seconds.
Peak reported Metal memory was 51.37 GiB, the process reported zero swaps, and
the final H.264/AAC MP4 contained 124 unique frames with synchronized 32 kHz
stereo audio. Its SHA-256 is
`bf651365864295b8de793af7b4311e8b2e84269fd280bda88f4a38068fd19665`.

Visual acceptance confirmed a stable umbrella and caped figure, wet-street
reflections, a levitating bus, and a luminous dragon developing through the
shot without the rejected spatial lattice. The final dragon remains slightly
soft and translucent, so this is a practical four-evaluation receipt rather
than a claim of parity with the full 19-evaluation trajectory. The prompt
requested driving rain, so its broadband soundtrack is not a clean-noise or
hiss acceptance reference.

## Dynamic sparse Metal receipt

The Apple-GPU dynamic-sparse path was accepted with a real synchronized-video
generation, not from random-tensor timing alone:

- **Checkpoint:** Managed `video-minimax-h3-fl2va-bf16-mlx`.
- **Adapter:** Checksum-pinned `minimax-h3-lightx2v-4step`, strength `1.0`.
- **Geometry:** 768 x 448 pixels, 124 frames, 24 frames per second, and 13,085 packed rows.
- **Schedule:** Five points and four transformer evaluations.
- **Seed:** `20260808`.
- **Dense arm:** Resident BF16, exact `quality`.
- **Sparse arm:** Resident BF16, attention-only `maximum`, tau `1.0`, and no cache reuse.
- **Prompt:** A continuous rain-soaked railway-platform two-shot with a detective,
  a paramedic, a recorder handoff, a passing train, and two spoken lines

| Phase | Dense | Dynamic sparse |
| --- | ---: | ---: |
| Conditioner preparation/load | 0.332 s | 0.311 s |
| Text encoding | 4.077 s | 4.066 s |
| Transformer preparation | 14.218 s | 14.093 s |
| Evaluation 1, protected dense | 211.479 s | 108.017 s |
| Evaluation 2, sparse plus one-time gate | 154.554 s | 107.196 s |
| Evaluation 3, sparse steady state | 131.430 s | 101.294 s |
| Evaluation 4, protected dense | 134.311 s | 127.645 s |
| Denoising | **631.826 s** | **444.216 s** |
| Video/audio decode | 85.632 s | 72.503 s |
| Native generation total | **736.285 s** | **535.386 s** |
| Process wall time | 737.28 s | 537.28 s |

This is a 29.7% measured denoise reduction and 27.3% measured native
end-to-end reduction. Both processes reported zero swaps and approximately
68.0 GB peak footprint. The dense arm paid a larger first-evaluation compile
cost, while the later sparse process could reuse system Metal compilation
state. Therefore the complete-run delta is an observed receipt, not a claim
that every second came from sparsity. The most comparable late dense arm took
131.430 and 134.311 seconds; sparse steady state took 101.294 seconds. The
sparse process's final protected dense evaluation took 127.645 seconds, giving
an in-process thermal control for the 20.6% faster sparse step.

Before sparse execution was admitted, the custom all-routes-dense Metal sample
matched fused SDPA at `max_abs=0.10026`, `mean_abs=0.00219996`, and
`rel_l2=0.0027471`. Both outputs then passed the artifact boundary: 768 x 448
H.264, exactly 124 frames at 24 frames per second and 5.167 seconds, plus stereo AAC at
32 kHz. Eight matched frame samples retained both actor identities, the
recorder handoff, coherent faces, train movement, and the final composition.
Native Parakeet transcribed both soundtracks exactly as
`You kept the recording? Every second.`

The dense artifact SHA-256 is
`5d1b0ed616a7527ab8fefc1982ac88a21e23ab7087abb51fc574372229dca3ed`.
The dynamic-sparse artifact SHA-256 is
`96bdcbe41bbd65fde2784384150c9b9b6ec867dcf61456f8a74b74b0a52c132d`.

## True-768 one-evaluation boundary

A separate release-mode probe locks the physical 768-line target without
pretending that a two-point sampler is a quality result:

```bash
scripts/h3-end-to-end-benchmark.sh probe-768
```

- **Geometry:** 1344 x 768 pixels, 124 frames, 24 frames per second.
- **Schedule:** Two points and one complete 50-block model evaluation.
- **Packed rows:** 37,794.
- **Execution:** Dense BF16, blockwise compiled, with 768-query single-evaluation attention chunks.
- **Acceleration:** Exact `quality` with no tail reuse.

| Phase | Baseline | Optimized exact |
| --- | ---: | ---: |
| Text encoding | 4.218 s | 4.078 s |
| Transformer preparation | 4.661 s | 4.510 s |
| Denoising | 953.485 s | **775.336 s (12:55.336)** |
| Video VAE decode | 289.586 s | 242.004 s |
| Audio decode | 0.873 s | 0.829 s |
| End to end | 1,253.425 s | **1,027.310 s (17:07.310)** |

The full model evaluation itself fell from 953.192 to 775.062 seconds, a
1.230 times the throughput and 18.7% less wall time. End to end, the boundary
is 3:46.115 faster (18.0%). Peak reported Metal memory remained 48.77 GiB, so
this workload is compute-bound rather than blocked by unified-memory capacity.
The valid H.264/AAC output is 1344 x 768 at 24 frames per second with 32 kHz stereo audio and a
5.167-second duration. Its SHA-256 is
`f74e0d2ab0cf9a58233e954560a9aa6c05a1f905fd78914350a077b70f7e5848`, exactly
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
production spatial tile from 256 to 320 pixels. At 832 x 480 pixels and 124 frames, the
matched decode fell from 96.091 to 76.421 seconds (1.257 times) while reported peak
Metal memory fell from 9.85 to 8.91 GiB. A 480-pixel arm reached 77.577 seconds
and lost. At the true 1344 x 768 shape, the accepted 320-pixel arm completed in
213.005 seconds at a 15.12 GiB reported peak; the 480-pixel arm crossed 252
seconds without completing and was stopped.

This is a non-quantized runtime change: the released decoder, precision,
causal tiling, and overlap blend remain intact. The larger tile performs fewer
overlapping tile evaluations and gives each evaluation more spatial context.
At true 1344 x 768 spatial geometry, an order-balanced 22-frame screen confirmed
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
schedule and 25.173 seconds for the accepted schedule, a 1.281-times speedup with
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

Inference-only solver screens did not change that conclusion. On the locked
416 x 256 proxy, 12-point variable-step Adams-Bashforth was less faithful to the
20-point Euler artifact than plain 12-point Euler (0.667 versus 0.694 SSIM).
A two-evaluation Heun solve reached the desired compute class—45.548 seconds of
proxy denoise—but produced tiled chromatic noise at 0.327 SSIM. Both candidates
were removed; these are rejection receipts, not available runtime modes.

### Arithmetic roof

At 73,470 packed rows, the released 50-layer transformer performs about 211.390
TFLOP per block: 154.767 TFLOP of dense attention and 56.624 TFLOP of linear
projections. The accepted 25.173-second block is therefore about 8.40 effective
TFLOP/s. For scale, even treating 36 TFLOP/s as a perfectly attainable
end-to-end ceiling gives 5.872 seconds per block and 293.6 seconds per complete
model evaluation. The 19-evaluation exact schedule cannot fall below roughly
93 minutes of transformer arithmetic, while the 417-block `maximum` schedule
cannot fall below roughly 40.8 minutes, before the approximately seven-minute
video decode estimate. Real execution must be slower than those floors.

This bound rules out a 30-minute ten-second result from exact kernel scheduling
alone. Reaching that class requires less model math—distillation, a validated
sparser attention structure, or fewer evaluated blocks—not more disk, unified
memory, or a debug-versus-release correction.
