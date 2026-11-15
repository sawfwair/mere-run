# SOV MLX head fusion and dispatch profiling — 2026-11-14

MLX GPU is the primary performance path. This pass tested fused BF16 head
normalization and RoPE, cached custom-kernel launch configuration, combined
BF16 projections, and MoE threadgroup sizes. **None established a meaningful
additional end-to-end speedup over the existing Q8 Metal/attention baseline.**
The head prototype remains an explicit diagnostic option; ordinary inference
defaults are unchanged. Projection fusion and MoE group changes were not
integrated.

Parity passed: **160 measured transcriptions and eight recorded first inferences**
matched frozen tokens. The complete captured verifier hidden state was bit
identical to the latest Q8 Metal/attention baseline with both dispatch wrappers.
The [JSON receipt](./sov-transcribe-mlx-heads-2026-11-14.json) retains both complete
end-to-end studies, operator samples, rejected experiments, native profiles,
source and binary hashes, and the bundled shader hash.

## Configuration

- Apple M5 Max, 128 GiB memory; macOS 27.2 (26B123), Xcode 27.2 (27B52).
- SOV-Transcribe-1 revision `505d8fae243924c0d8acdc622f7f0abab344eba3`.
- DFlash revision `8c0b3f3a98507d2e3dd35b1bee0f147fa3bded05`.
- MLX encoder/projector, Q8 group-64 custom experts, shared position masks and
  direct partial RoPE enabled. DFlash enabled; chunk size 1024; limit 448 tokens.
- Dynamic-int8 encoder disabled. Its earlier combination with Q8 experts remains
  unqualified for exact parity.
- Recordings: `Fixtures/D1/released/speech.wav`, 3.5183125 seconds, 13 frozen
  tokens; `Fixtures/Nemotron3/returning-speaker.wav`, 46.57 seconds, 138 tokens.
  Fixture paths are relative to `Tests/MereRunCoreTests/`; audio hashes are in
  every inference receipt.
- Release command: `swift build -c release --product mere.run -Xswiftc -disable-cmo`.

## Head kernel and host dispatch

`Gemma4HeadNormRoPE` combines Q/K/V head normalization, rotary positions and
output layout in one dispatch. Its reduction matches the native RMS tree. It
rounds normalization to BF16, applies and rounds the learned scale, then rotates
the normalized BF16 values. This preserves the intermediate rounding points.
Only BF16 256- and 512-wide heads with at most 16 query tokens use the kernel;
larger prefill and unsupported inputs retain their existing operations.

The isolated production-shader screen matched Q/K/V output exactly. Against
the already optimized direct-RoPE graph, median paired operator reduction was
**10.76% for the full-attention capture** and **-1.37% for sliding attention**.
The full-attention operator represents only part of the decoder; this did not
predict a comparable transcription gain.

The initial Swift wrapper rebuilt template, grid, dtype and output-shape
configuration on every invocation. The final wrapper owns Cmlx kernel handles
and caches immutable launch configuration by batch and query length. Current
projections, normalization parameters and integer positions remain inputs to
every call. No parameters or position values are captured in the configuration.

A matched debug diagnostic measured host graph creation at **31.47 microseconds
with generic binding versus 23.12 microseconds with prepared binding**, a 26.5%
reduction. It timed 30 alternating blocks of 100 calls, deliberately excluding
GPU evaluation. This is a host diagnostic, not release inference latency.

## End-to-end result

Each study used four fresh processes: two recordings in each of two sessions.
One resident target served both modes, with two warmups and ten measurements per
mode. Adjacent pairs alternated order; the second session reversed recording
order and started with fused heads. Only head processing changed within a pair.
Timing includes synchronized request cleanup and excludes loading, file reading
and policy switches.

| Wrapper study | Recording | Median paired latency reduction | Fused wins |
| --- | --- | ---: | ---: |
| Generic binding | 3.52 s | 0.14% | 11/20 |
| Generic binding | 46.57 s | 1.17% | 13/20 |
| Prepared binding | 3.52 s | -0.21% | 9/20 |
| Prepared binding | 46.57 s | 0.36% | 11/20 |

Positive values mean lower fused latency. These statistics use adjacent pair
ratios, not ratios of pooled mode medians. Each study compared one binary
internally; the wrapper studies used different binaries with the same shader.
Their absolute times must not be compared as a measure of the binding change.

Background CPU workers were active and untouched stages drifted substantially.
The final win counts and small reductions do not establish a useful latency
benefit. A quiet-machine confirmation would be needed before making a stronger
claim. The existing Q8 Metal/attention path remains the performance baseline.
Earlier gains are recorded in the [MoE](./sov-transcribe-metal-moe-2026-11-13.md)
and [attention](./sov-transcribe-attention-2026-11-14.md) qualifications.

## Other screens

Combining BF16 Q/K/V or gate/up matrices produced exact projection outputs on
the tested verification and prefill inputs, but no consistent timing benefit.
That approach was not integrated.

The MoE screen retained two reused routes and four outputs per SIMD group,
testing one, two, four and eight SIMD groups per threadgroup. All outputs were
bit identical. The existing two-group implementation had the lowest median in
that screen; other sizes did not demonstrate a robust gain. Existing production
geometry remains unchanged. Background load also affected these operator times.

The target verifier remains the dominant coarse generation component in the
synchronized profiles. These results support moving further optimization work
to larger costs, such as expert computation and verification efficiency.
Diagnostic stage barriers alter overlap and cannot be summed into release time.

## Validation and limits

- All 168 recorded outputs matched frozen BF16+DFlash token IDs. Other warmups
  are excluded. Verification rounds, accepted drafts, rejected rounds and reused
  verification rounds were unchanged between modes for each recording.
- Candidate IDs `[1, 16]` and captured BF16 verifier hidden state
  `[1, 16, 2816]` were bit identical to the Q8 baseline for both wrappers.
- Forty-two focused regression tests passed, covering strided inputs, batches,
  offsets through 65537, head layouts, staged rounding, parameter replacement,
  prefill fallback, existing expert kernels, SOV runtime and weights, and CLI
  validation and documentation contracts.
- The matched host-dispatch profile and real-checkpoint prepared-head profile
  passed. Independent NumPy FP32 fixtures used `MLX_ENABLE_TF32=0`; their initial
  default-TF32 run exceeded tolerance in unchanged encoder/draft fixtures. No
  tolerance was relaxed. Checkpoint parity and resident timings used default TF32.
- Release builds, scoped strict SwiftLint and `git diff --check` passed. The full
  repository gate remains deferred at the user's request.

Two English fixtures do not qualify multilingual recognition quality or
performance across Apple Silicon devices. The prototype remains off by default.

## Reproduce

```bash
.build/release/mere.run model benchmark sov \
  Tests/MereRunCoreTests/Fixtures/Nemotron3/returning-speaker.wav \
  --model-root .build/qualification/sov/target \
  --draft-model .build/qualification/sov/draft \
  --experimental-decoder q8-metal --experimental-attention \
  --interleave-head-norm --warmups 2 --repetitions 10 --json
```

Add `--head-norm-first` to reverse the starting order. For standalone prototype
runs, replace interleaving with `--experimental-head-norm`. Warmups/repetitions
apply per mode. JSON labels fused samples with `+heads` and aligns kernel labels,
resident times and measurements by index.

`profile_sov_head_norm.py` in `scripts/model-conversion/` reproduces the shared
head-shader screen from native projection captures. The optional
`SOVHeadDispatchProfileTests` writes creation-only timings when
`MERERUN_TEST_MLX_DEVICE=gpu` and `MERERUN_PROFILE_SOV_HEAD_DISPATCH` specifies an
output path. The MoE profiler accepts `--reuse-rows 2 --simdgroups 1 2 4 8` to
reproduce the group-size screen without changing the runtime kernel.
