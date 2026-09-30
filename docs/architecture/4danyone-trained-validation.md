# 4DAnyone trained checkpoint validation

The default native path uses **FP32 arithmetic with BF16 parameter storage**
and passes the trained FP32 comparison on Apple Silicon. Pure BF16 arithmetic
still fails its separate precision check. This report records local validation
on September 15, 2026, on an
Apple M4 Max with 128 GiB of unified memory.

The implementation remains uncommitted on `codex/4danyone`, based on
`99becddba5f73ddfbde8dfb740d8ff75c60bcbee`. The
[contributor API](../../Sources/MereRunCore/FourDAnyone/README.md) has no public
CLI command or managed model entry.

## Trained tensor comparisons

All comparisons use the complete released weights. FP32 checks upcast the
released BF16 values; they do not substitute a different checkpoint. The independent reference
executes the unmodified upstream graph with PyTorch 2.8.0 on CPU. Native runs
use Swift and MLX on the Metal GPU. Transformer tests retain all 30 blocks,
3,072 hidden channels, and 24 heads, with four targets and small spatial and
temporal grids. They cover one source and five sources with reference packing.

Normalized error means root mean square error divided by the reference's root
mean square value. Thresholds were set before the trained tests ran.

| Check | Maximum normalized error | Threshold | Result |
|---|---:|---:|---|
| FP32 transformer, direct and packed | 0.00176% | 0.01% | Passed |
| FP32 transformer computation with BF16 storage, direct and packed | 0.00176% | 0.01% | Passed |
| 24-step Base trajectory, FP32 computation with BF16 storage | 0.000125% | 0.01% | Passed |
| FP32 pose encoder, target and learned null | 0.000079% | 0.01% | Passed |
| FP32 VAE, one- and five-frame encode/decode | 0.000295% | 0.01% | Passed |
| BF16 pose encoder against FP32 | 0.670% | 3% | Passed |
| BF16 transformer arithmetic, direct | 4.507% | 3% | Failed |
| BF16 transformer arithmetic, packed | 4.957% | 3% | Failed |

The VAE inputs are five frames from the released Pexels example, resized and
cropped to 64 by 96 pixels. Transformer inputs use its VAE source encoding,
diagnostic RGB pose inputs, explicit noise, and the released frozen prompt.
These inputs test numerical behavior; they are not recovered human motion.
The table records the current precision policy and the separate BF16 arithmetic
diagnostic. The trajectory uses four targets, one source, and the same small
grid as the forward checks.

The earlier nine synthetic reference tests also passed on the GPU. They cover
the production checkpoint schema, padding, routing, schedule, cancellation,
and four-step grouped denoising. Those results remain separate from trained
checkpoint and visual qualification.

Three packed-reference forwards in one resident native session produced
byte-identical tensors in each tested precision. This establishes repeatability
for these inputs; it is not a generation-speed benchmark.

## BF16 investigation

The initial comparison used timestep 625 in the FP32 reference. The released
denoiser casts its timestep to BF16, which rounds 625 to 624. The native port
preserves this behavior in `computePrecision: .model` mode. A second FP32
control receives the same BF16-rounded
inputs as the native model, including the rounded timestep. Native drift
initially measured 4.6–5.0% with that control. The 3% threshold was not relaxed.

An additional run of the unmodified upstream graph under PyTorch CPU BF16
autocast differs from the aligned FP32 control by 5.56% for direct sources and
4.07% for packed sources. Native BF16 differs from that CPU BF16 run by about
5.1%. Intermediate native-to-CPU differences grow from about 0.43% after the
first block to 5.7–6.6% after the final block.

That initial evidence was consistent with accumulated precision differences.
It did not isolate individual operations or establish parity with CUDA's
autocast and attention kernels. The failed native precision check remains a
qualification blocker for BF16 arithmetic. FP32 computation is the verified
reference path and is now the default, independently of parameter storage.

### Default precision remedy

Promoting only timestep conditioning reduced error to 3.90% direct and 3.81%
packed. Adding FP32 text conditioning and the head yielded 3.32% and 4.63%.
Neither selective change passed the original 3% limit, so neither was adopted.

The adopted path stores all parameters in BF16 and runs transformer arithmetic
in FP32. MLX casts the active projection weights as needed; model parameters
remain BF16 before and after inference. The generation loop also keeps its
latent state in the selected computation dtype across every Euler step.

All four trained small-grid outputs are byte-identical to the native model
with FP32 parameter storage. Maximum normalized error against the independent
PyTorch FP32 reference is `1.7523056e-5`, below the stricter `1e-4` FP32 limit.
Stored parameters occupy 12,289,763,712 bytes and peak MLX allocation in the
latest focused run was 13,403,419,000 bytes. These observations apply to the small grid.

This validates an FP32 execution path with compact weight storage. It does
not establish parity with upstream CUDA/BF16 arithmetic. Full-resolution
execution and complete visual generation are separate checks.

The independent CPU exporter also ran all 24 Base steps with the complete
trained transformer, upstream camera routing, denoising function, and
scheduler. The native generator's final latent error was `1.2517155e-6`
normalized RMSE, below the `1e-4` limit. It completed in 27.69 seconds with
13.73 GB peak MLX allocation on the small diagnostic grid. Parameters stayed
BF16 and the denoising state stayed FP32. A separate routed four-step regression
passes on CPU and Metal and checks preservation of input and state precision.

### Activation correction

Frozen primitive inputs exposed an activation difference: MLX's BF16 GELU
and SiLU expressions rounded intermediate operations differently from the
PyTorch CPU primitives. Evaluating each activation in FP32 and then casting
back reduced the GELU normalized error from `2.40e-4` to `8.15e-8`; SiLU
became exact for these inputs. The correction is local to FourDAnyone.
Its independent 16,384-value regression fixture passes on CPU and Metal.

The trained model rerun preserves FP32 parity. BF16 direct error decreases
from 4.603% to 4.507%, and packed error decreases from 4.996% to 4.957%.
This is a verified primitive correction, but it does not resolve accumulated
model drift. Both transformer comparisons still fail the original 3% guard.

The roughly 20 MB operation cases can be replayed on CUDA without moving the
full checkpoint. The replay records hashes, device, and PyTorch/CUDA versions
and refuses a CPU fallback. CUDA execution and a matched full-model CUDA
trace remain outstanding. See the [operation replay commands](../../Sources/MereRunCore/FourDAnyone/README.md#replay-frozen-bf16-operations).

A portable full-model exporter also replays the frozen small-grid inputs with
BF16 weights and autocast, recording assembly, time/context embeddings,
selected blocks, and predictions. Its explicit CPU diagnostic completed both
30-block forwards and reproduced the earlier 5.56%/4.07% upstream BF16-to-FP32
drift. It verifies source and checkpoint identity and fails if its default
CUDA device is unavailable. The CUDA branch has not been executed.

Against this CPU BF16 trace, the corrected native prediction differs by
5.85% for direct sources and 4.88% for packed sources. Assembly differs by
less than `6.5e-6` normalized error, while first-block error is 0.24–0.30%
and final-block error is 5.89–7.27%. The growing differences still require
matched CUDA traces; the primitive correction does not establish complete
CPU-BF16 equivalence either.

## Production-size execution with default precision

The default FP32 computation with BF16 storage passed one full trained
forward at the released 121-frame, 1,280 by 704 resolution, with four targets
and one source. All 30 blocks completed. The output was finite FP32 with shape
`[4, 48, 31, 80, 44]`, while all stored model parameters remained BF16.

Peak MLX allocated memory was 40,191,302,056 bytes (40.19 GB). The learned
null-pose encoding took 0.96 seconds, transformer loading took 1.65 seconds,
and the forward took 817.38 seconds. Output standard deviation was 1.0656.

This run used tiled diagnostic source latents and null target poses. It
validates full-size execution of the precision remedy, not canonical motion
conditioning or complete novel-view videos. Other workloads were active;
durations are observations and must not be used as isolated speed comparisons.

### Historical execution before the activation correction

The separately enabled smoke check uses the released 121-frame, 1,280 by 704
resolution and four target views. It encodes the full learned null condition
and runs one trained BF16 transformer forward. Source latents are tiled
diagnostic data and target poses use the null condition.

The check passed with finite output of shape `[4, 48, 31, 80, 44]`. All 30
transformer blocks completed. Peak MLX allocated memory was 34,325,064,108
bytes (34.33 GB, or 31.97 GiB). The full null-pose encoding took 2.63 seconds,
transformer loading took 2.72 seconds, and the forward took 605.31 seconds.
Output standard deviation was 1.063.

This establishes shape, finite-output, and memory-execution evidence only.
It does not resolve the BF16 precision failure or establish the complete
24-step generation pipeline. Six-target groups and the larger packed-source
production workload remain unmeasured.

Other workloads were active on the machine. Recorded durations are execution
observations, not isolated performance benchmarks.

The full 30-block result above predates the activation correction. It is
retained as historical execution evidence; it does not validate the updated
activation path at full size across all blocks.

### Historical BF16 first-block profile

The updated runtime completed four resident first-block runs at the released
grid, with four targets and one source. It reused the frozen null-pose tensor
from the earlier smoke. Each stage was evaluated separately to measure its
cost. Query chunk sizes 512 and 1,024 produced exactly equal block outputs.

| Query chunk | First run | Second run | Peak MLX allocation |
|---|---:|---:|---:|
| 512 | 10.82 s | 23.12 s | 45.21–46.05 GB |
| 1,024 | 26.01 s | 19.42 s | 46.05 GB |

Video attention took 5.82–13.83 seconds, multiview attention 1.99–7.40
seconds, text attention 0.65–1.58 seconds, and feed-forward work 2.35–5.64
seconds. Concurrent inference and the large variation prevent a reliable
speed comparison. The default query chunk remains 512. Retaining the baseline
output and evaluating intermediate stages also makes this profile's memory
scope different from the earlier end-to-end forward.

## Canonical source package

The released Pexels source has been frozen into a package with the upstream
121-frame, 30-fps timeline, verified lossless canonical and GVHMR videos,
the released prompt, a four-view plan, and seed-4196 initial noise from the
upstream CPU-FP32-to-BF16 sampling sequence. Its manifest records every hash
and the actual source dimensions, 1,088 by 1,920 pixels.

The manifest remains `awaiting_motion_conditioning`. The package still needs
recovered motion, the foreground crop, four target skeleton videos, solved
cameras, and encoded source/pose tensors. No complete novel-view video has
been generated. See the [source preparation commands](../../Sources/MereRunCore/FourDAnyone/README.md#freeze-a-source-for-motion-preparation).

## Repository gate

`MERERUN_SWIFT_DISABLE_INDEX_STORE=1 ./scripts/check.sh` passed after the
precision policy change with exit code zero. SwiftLint checked 2,089 files with
no violations. XCTest reported 4,365 tests, 327 skips, and zero failures;
Swift Testing passed another 51 tests. The build, dependency and package
policies, documentation checks, CLI help sweep, and hygiene checks passed.

The six trained-weight checks, trained trajectory, production-size check,
block profile, conditioning precision experiment, and primitive replay are
opt-in and skip in the standard gate. Their separately observed results above
govern model qualification. The focused GPU suite ran 22 tests with three
skips; its only failing test was the pure BF16 transformer diagnostic, with
four failed assertions. The default FP32 computation, trajectory, VAE, pose,
fixture, and storage-precision checks passed. The standard gate's success
does not clear the separate BF16 arithmetic failure.

## Provenance and reproduction

- Upstream source: `8cd60c40d90882de07645cc435dcf24bc9b4fbd1`.
- Hugging Face asset revision: `4c80e87b805a5f8461cf339cdbe2fb4249e585aa`.
- Complete transformer checkpoint SHA-256:
  `aff60b0db2d333bd9e960a9cf3333cc8dd40fe76614a22f75a0da72be4e8289f`.
- Complete Wan2.2 VAE checkpoint SHA-256:
  `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`.

The exporter verifies those full-file hashes before using the weights. See
[trained validation commands](../../Sources/MereRunCore/FourDAnyone/README.md#validate-trained-weights)
and [machine-readable verification](./4danyone-native-verification.json).
Set `MERERUN_4DANYONE_DIAGNOSTIC_STAGES=1` with the trained tests to save native
assembly, time, context, and selected block tensors for further investigation.

## Remaining qualification

- Qualify BF16 arithmetic separately if it is needed; the default uses verified
  FP32 computation.
- Compare production-size VAE encoding and decoding at the intended precision.
- Run all 24 Base steps at the released resolution with canonical source,
  skeletons, and cameras; inspect synchronized target videos.
- Qualify proposal JPEG preparation and generation with packed references.
- Qualify motion recovery and dynamic reconstruction independently.
- Complete hosted CI and release review before advertising support.
