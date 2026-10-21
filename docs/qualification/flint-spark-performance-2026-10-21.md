# Flint Spark performance investigation

The initial 4.30-token/s measurement used dense compatibility kernels and an
unoptimized MLX bridge. It was an execution diagnostic, not a representative
performance result for the native runtime.

## Publisher comparison

Recursive's [launch chart](https://recursive.com/updates/introducing-recursive-flint)
reports these decode rates on RTX Spark N1X with Flint NVFP4:

| Publisher measurement | Tokens/second |
| --- | --- |
| Short context, 1K input tokens | 155.82 |
| Normalized average through 128K context | 101.95 |

This investigation uses the published affine INT4 checkpoint on DGX Spark GB10.
The hardware, weight representation, kernels, and prompt protocol differ. None
of the measurements below establishes parity with the publisher's NVFP4 engine.

## Setup defects

The first source-build environment omitted
`MERERUN_MLX_CUDA_JIT_INCLUDE_PATH`. NVRTC could not find
`cute/numeric/numeric_types.hpp` when compiling the quantized tensor-core kernels.
Automatic dispatch caught that failure and selected dense compatibility for both
ordinary and routed projections. Supplying the matching CUTLASS/CuTe headers
enabled both operations. All 16 capability probes completed with native kernels
across sequence lengths 1, 5, 16, and 32, on default and leased streams.

The reused CMake bridge also had an empty `CMAKE_BUILD_TYPE` and no optimization
flags for its Swift libraries. Building only the CLI in release mode left that
dependency unoptimized. A separate bridge was rebuilt from the same pinned
sources in CMake `Release` mode, targeting CUDA architecture 121.

`scripts/prepare-linux-native.sh` now:

- Builds the CUDA bridge in `Release` mode.
- Checks that the matching CUTLASS/CuTe headers exist.
- Prints the JIT and CCCL header exports alongside the SwiftPM bridge exports.
- Rejects an unoptimized bridge during `--check`.

The real-host regression check rejected the old bridge with exit 69 and accepted
the optimized bridge. The original bridge and deployed owner services were
preserved.

## Measurements

All runs use native Swift model computation. Checkpoint identity remains
`RecursiveSI/flint-int4@3b93f9053b530488405ae9fca51d4900d6ce21f0`.
The model source remains `9b03cb2a37975a4efea01f20ff69b1177f3681d1`.
No model math or checkpoint conversion changed during this investigation.

| Configuration | Decode tokens/second | Scope |
| --- | --- | --- |
| Debug CLI, dense fallback, unoptimized bridge | 4.30 | Original arithmetic diagnostic, 47 generated tokens |
| Debug CLI, native kernels, unoptimized bridge | 15.00 | Arithmetic diagnostic, 46 generated tokens |
| Release CLI, native kernels, unoptimized bridge | 15.72, 14.88 | Two capped 128-token list requests |
| Release CLI, native kernels, release-optimized bridge | 27.71, 28.73 | Two capped 128-token list requests after warmup |

The capped requests use this prompt with temperature 0:

```text
Write a numbered list of the integers from 1 to 1000, one per line.
```

Token counts include reasoning. Decode timing excludes checkpoint loading and
prefill. Builds were active during the intermediate unoptimized-bridge runs.
The final optimized cohort runs after both builds finish, under normal shared
scheduler admission with a 48 GiB reservation.

The optimized CLI links the release-built package objects to the newly optimized
bridge, using the captured SwiftPM linker command with the bridge library paths
replaced. The bridge uses the same pinned Swift and C++ sources and compatibility
patches. This is a source-performance diagnostic, not a packaged distribution
qualification.

The optimized executable's SHA-256 is
`3291d02743de0cf07259cc8c3f2d00bd7396adde23d1c99b86bb797fadeea909`.
It also passed the same five loopback API checks as the original receipt:
arithmetic, parsed Python addition, a weather tool call, use of the synthetic
tool result, and HTTP 400 for unsupported JSON mode. The successful API calls
took approximately 2.40, 3.16, 2.26, and 3.61 seconds, respectively, with the
model loaded. These request timings are not decode-only throughput figures.

The corrected build reaches roughly 28 tokens/second in these short runs. A
substantial gap remains to the publisher's specialized NVFP4 implementation;
the source runtime has not implemented that fused execution path. This
investigation does not establish sustained or long-context throughput.

## Numerical boundary

The optimized GPU scalar fixture passed all 221 logit comparisons with maximum
absolute error `3.874302e-7`, with TF32 disabled. This checks the synthetic model's
math and cache behavior.

Released-checkpoint agreement is weaker. With native kernels forced, the
optimized five-token probe selected the reference next token at 4/5 positions
in a full forward, 4/5 with chunk size 1, and 5/5 with chunk size 3.
Full-forward logit cosine similarity was 0.998699, with maximum absolute
difference 7.945216 and RMS difference 0.863061. The earlier fallback-path 5/5
result does not qualify this native path.

The capped benchmark's visible outputs also differed between fresh runs at
temperature 0. Those requests measure throughput; they are not evidence of
output equivalence or deterministic generation.

These observations remain diagnostic. Broad quality, stable released-weight
numerical agreement, long-context behavior, and publisher-level performance
remain unqualified. The [original receipt](./flint-spark-2026-10-21.md) retains
the earlier measurements and correctness limits.

After testing, both authenticated owner services remained ready, the temporary
API port was closed, and the scheduler had no live tickets. Detailed outputs,
the Nsight trace, compiler logs, and failed attempts remain in the private
evidence bundle. The [machine-readable performance receipt](./flint-spark-performance-2026-10-21.json)
records the measured cohorts and qualification limits.
