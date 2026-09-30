# 4DAnyone native generation prototype

This directory owns the Swift and MLX implementation of the Base 4DAnyone
denoiser and RGB pose encoder. It accepts prepared conditioning and explicit
initial latents. It is a contributor API and has no managed model ID or CLI
command yet.

The port follows the [pinned 4DAnyone source](https://github.com/ant-research/4DAnyone/tree/8cd60c40d90882de07645cc435dcf24bc9b4fbd1).
See the [architecture assessment](../../../docs/architecture/4danyone-native-assessment.md)
for the release contract, source evidence, license boundaries, and remaining
automatic motion recovery and dynamic reconstruction work.

## Ownership

| File | Responsibility |
|---|---|
| `FourDAnyoneTransformer.swift` | Patch projection, reference packing, pose injection, source timesteps, and output layout |
| `FourDAnyoneLayers.swift` | Video, multiview, and text attention; normalization, rotary embedding, and feed-forward layers |
| `FourDAnyonePoseEncoder.swift` | Symmetric 3D convolutions and the learned null condition |
| `FourDAnyoneModelLoader.swift` | Strict checkpoint partitions, layout conversion, and frozen prompt metadata |
| `FourDAnyonePlan.swift` | Stable camera IDs, cyclic group routing, and the Base flow-matching schedule |
| `FourDAnyoneGenerator.swift` | Sequential groups, progress, cancellation, and one-view-at-a-time VAE decoding |

The runtime reuses the Wan patch layout, layer normalization, and VAE, and the
SCAIL-2 attention chunk helper. FourDAnyone owns its transformer block: each
block performs video attention, attention across views within each frame,
text attention, and a feed-forward projection.

## Prepared inputs

All generation tensors use channel-first video layouts:

| Input | Released shape |
|---|---|
| Initial latents | `[views, 48, 31, 80, 44]` |
| Source latents | `[1, 48, 31, 80, 44]`, or five sources with references |
| Target pose features | `[views, 3072, 31, 40, 22]` |
| Null pose features | `[1, 3072, 31, 40, 22]`, or two entries with reference packing |
| Frozen prompt context | `[1, 512, 4096]`, BF16 |

Encode skeleton RGB videos with `FourDAnyonePoseEncoder`. The input shape is
`[views, 3, 121, 1280, 704]`. Compute null features by encoding an all-minus-one
video; trained biases make zero features an invalid replacement. Prepare these
features before loading the transformer to limit simultaneous weights.

For more than six targets, the default plan requires five sources: the input
video followed by four proposal references in upstream order. Prepare those
references using the released decode, JPEG quality-85, and VAE encode boundary.
Automatic proposal preparation, video sampling, skeleton rendering, and camera
geometry are outside this API. A view plan orders supplied features; it does
not generate their camera poses.

## Use the contributor API

```swift
import Foundation
import MereRunCore
import MLX

// prepared.safetensors contains the frozen tensors listed above.
let prepared = try MLX.loadArrays(url: preparedURL)
let model = try FourDAnyoneModelLoader.loadTransformer(from: checkpointURL)
let context = try FourDAnyoneModelLoader.loadPromptContext(from: promptURL)
let generator = FourDAnyoneGenerator(transformer: model)
let plan = try FourDAnyoneViewPlan(viewsPerLayer: 6)
let output = try generator.generate(
    initialLatents: prepared["initial_latents"]!,
    conditioning: FourDAnyonePreparedConditioning(
        sources: prepared["source_latents"]!,
        poseFeatures: prepared["pose_features"]!,
        nullPoseFeatures: prepared["null_pose_features"]!,
        promptContext: context
    ),
    plan: plan,
    progress: { progress in
        // Send progress to your diagnostic or UI channel.
    }
)
try MLX.save(arrays: ["latents": output], url: outputURL)
```

Keep canonical camera IDs as the first tensor dimension. Capture initial
latents from the reference runtime when comparing outputs: identical integer
seeds do not make PyTorch and MLX produce identical noise. The loader accepts
the complete released checkpoint or the exact transformer/pose partition.
It rejects missing, extra, partial companion, and incorrectly shaped tensors.
The trained `proj_4x` partition is retained even though inference uses `proj_2x`.

The default keeps checkpoint weights in BF16 storage and evaluates the
transformer and denoising state in FP32. MLX promotes weights for the current
operation; the model retains its BF16 parameters. This matches the verified
FP32 reference without retaining a second full FP32 checkpoint.

`dtype` controls stored parameter precision. `computePrecision: .float32`
controls arithmetic and is the default. `computePrecision: .model` uses the
parameter dtype for arithmetic; BF16 arithmetic remains experimental and
fails the current numerical limit. These are distinct execution modes.

`decodeViews(_:using:consume:)` accepts a loaded Wan2.2 48-channel VAE and passes
each canonical view to a consumer as `[frames, height, width, RGB]` values in
`[0, 1]`. The consumer owns encoding, frame rate, paths, and receipts. Verify
VAE precision and normalization against the source before publishing outputs.

Use one serial owner for a model. Cancellation is checked between groups,
transformer blocks, pose layers, and decoded views. A running MLX kernel must
finish before cancellation can take effect. The default query chunk is 512
tokens. See the [trained validation report](../../../docs/architecture/4danyone-trained-validation.md)
for measured execution scope and qualification failures.

## Numerical evidence

Run the checked-in fixtures with:

```bash
swift test --filter FourDAnyone
```

The fixture exporter executes unmodified upstream forward, routing, and
scheduler functions at the pinned commit. Transformer dimensions are reduced
to two blocks and 24 hidden channels; the pose encoder retains all production
channels. Synthetic weights and inputs exercise reference padding and crop,
source timesteps, nonzero null conditioning, and four-step grouped denoising.
The complete 1,182-key production schema comes from the released checkpoint
header. The fixture manifest records source and artifact hashes.

The fixture comparisons establish graph behavior with synthetic weights.
Additional trained FP32 checks pass on Metal at a small grid size, including
all 30 transformer blocks and VAE encoding and decoding. The default FP32
compute path with BF16 storage produces byte-identical small-grid predictions
to native FP32 weight storage, and passes the same `1e-4` normalized-error
limit against the independent reference. Pure BF16 arithmetic still fails its
separate precision check.

These checks do not establish CUDA/BF16 parity, visual quality, or speed.
The native rotary
path uses host FP64 trigonometry and FP32 tensor rotation; upstream rotates in
FP64. Full checkpoint qualification must measure this difference and compare
VAE source encoding, decoded frames, and synchronized views at the released
121-frame resolution. Turbo, motion recovery, and dynamic Gaussian
reconstruction are separate work.

## Validate trained weights

`scripts/reference-parity/export_4danyone_trace.py` runs the pinned upstream
graph with the released weights on CPU. It verifies the checkpoint hashes,
exports Wan VAE weights in native layout, and writes independent reference
tensors. Its short VAE input uses five frames from the bundled Pexels example
at 64 by 96 pixels. Transformer inputs use that source encoding, diagnostic
RGB pose inputs, four targets, and the released frozen prompt.

Create a validation directory with an `assets` subdirectory containing the
pinned `model.safetensors`, `Wan2.2_VAE.pth`, `prompt_context.safetensors`, and
`7017803-hd_1080_1920_30fps.mp4`. Use the dependencies listed in the exporter.
Run its `vae` stage before its `transformer` stage:

```bash
python scripts/reference-parity/export_4danyone_trace.py \
  --upstream /path/to/4DAnyone --assets /path/to/validation/assets \
  --output /path/to/validation/reference --stage vae
python scripts/reference-parity/export_4danyone_trace.py \
  --upstream /path/to/4DAnyone --assets /path/to/validation/assets \
  --output /path/to/validation/reference --stage transformer
MERERUN_TEST_MLX_DEVICE=gpu \
MERERUN_4DANYONE_VALIDATION_ROOT=/path/to/validation \
swift test --filter FourDAnyoneRealModelTests
```

The trained tests record maximum error, root mean square error, normalized
error, cosine similarity, elapsed time, and MLX peak allocated memory.
The FP32 comparison threshold is `1e-4` normalized root mean square error,
including FP32 computation with BF16 parameter storage.
The BF16 comparison permits `0.03` relative to an FP32 control receiving the
same BF16-rounded inputs, including the timestep. This measures
precision drift and does not establish parity with CUDA's mixed-precision
execution. Failed comparisons remain failures and require investigation.

For an independent comparison across all 24 Base steps, export the trained
small-grid trajectory with the same dependencies:

```bash
python scripts/reference-parity/export_4danyone_trajectory.py \
  --upstream /path/to/4DAnyone --checkpoint /path/to/validation/assets/model.safetensors \
  --reference /path/to/validation/reference/transformer-reference.safetensors \
  --output /path/to/validation/trajectory
MERERUN_TEST_MLX_DEVICE=gpu MERERUN_4DANYONE_TRAJECTORY=1 \
MERERUN_4DANYONE_VALIDATION_ROOT=/path/to/validation \
swift test --filter FourDAnyoneTrajectoryTests
```

The exporter runs the upstream model, camera routing, denoising function, and
scheduler in FP32 on CPU. The native check uses the default BF16 storage and
FP32 computation and compares the final latent state with a `1e-4` normalized
error limit. This tests the complete Base schedule at a small diagnostic grid;
it does not establish full-resolution visual quality.

The trained 24-step check passed on the M4 Max with normalized final-state
error `1.2517155e-6`. A routed four-step regression also checks that BF16
parameter storage preserves FP32 input and latent-state precision.

`FourDAnyoneProductionSmokeTests` is separately enabled by
`MERERUN_4DANYONE_PRODUCTION_SMOKE=1`. It encodes the full 121-frame null pose
and runs one four-target forward with FP32 computation and BF16 parameter
storage at the released 1,280 by 704 resolution.
It uses tiled diagnostic source latents and null target poses. Passing it
establishes shape, finite-output, and memory-execution evidence; it does not
establish video quality or the complete 24-step pipeline. The test stops
between blocks if the forward exceeds 15 minutes.

The default precision check passed on the M4 Max with 128 GiB of memory:
all 30 blocks returned finite FP32 outputs while stored parameters remained
BF16, with 40.19 GB of peak MLX allocation. The forward took 817 seconds with
other workloads active. These are execution observations, not an isolated
benchmark. Historical BF16 arithmetic results remain in the validation report.

## Replay frozen BF16 operations

The operation exporter isolates activation, normalization, projection, and
attention arithmetic from accumulated model drift. It verifies the released
checkpoint hash and saves about 20 MB of frozen inputs and trained weights.
Use the dependencies listed in the script:

```bash
python scripts/reference-parity/export_4danyone_operations.py \
  --checkpoint /path/to/model.safetensors --output /path/to/operations
```

Copy `operations-cases.safetensors` to a CUDA host and replay the identical
inputs there. The exporter fails if CUDA is unavailable:

```bash
python scripts/reference-parity/export_4danyone_operations.py \
  --cases /path/to/operations-cases.safetensors \
  --device cuda --output /path/to/cuda-reference
```

Copy the generated CUDA reference back beside the original cases, then run:

```bash
MERERUN_TEST_MLX_DEVICE=gpu \
MERERUN_4DANYONE_OPERATIONS_ROOT=/path/to/operations \
MERERUN_4DANYONE_REFERENCE_DEVICE=cuda \
swift test --filter FourDAnyoneOperationTests
```

The diagnostic writes errors for each primitive. Its pass status only checks
shapes and finite outputs; it does not qualify the complete BF16 graph.

For a complete small-grid BF16 reference, replay the frozen transformer inputs
with the full checkpoint on CUDA:

```bash
python scripts/reference-parity/replay_4danyone_bf16.py \
  --upstream /path/to/4DAnyone --checkpoint /path/to/model.safetensors \
  --reference /path/to/transformer-reference.safetensors \
  --output /path/to/new-cuda-trace
```

This exporter runs the unmodified upstream graph with BF16 weights, BF16
autocast, and SDPA. It saves assembly, time, context, blocks 0/14/29, and final
predictions for direct and packed sources. Native stages can be saved by
setting `MERERUN_4DANYONE_DIAGNOSTIC_STAGES=1` on the trained checks. Hashes
bind each trace to the frozen inputs, checkpoint, and pinned source revision.
The exporter defaults to CUDA and rejects an unavailable device. Its explicit
`--device cpu` path has been exercised locally; CUDA remains unverified.

## Profile a production-size block

After the production smoke has saved `production-null-pose.npy`, run:

```bash
MERERUN_TEST_MLX_DEVICE=gpu MERERUN_4DANYONE_PROFILE=1 \
MERERUN_4DANYONE_VALIDATION_ROOT=/path/to/validation \
swift test --filter FourDAnyoneProfilingTests
```

This profile repeats the first trained block twice for each query chunk size
(512 and 1,024). It evaluates video attention, multiview attention, text
attention, and feed-forward output separately. The receipt records timing,
memory, and output agreement. It uses diagnostic inputs and an already saved
null pose; it is not a complete generation benchmark.

## Freeze a source for motion preparation

The source preparation script uses the clean pinned upstream checkout to
decode the canonical 121-frame clip and verify lossless video round trips.
It copies the source and prompt, records the frame clock and four-view plan,
and freezes the upstream CPU noise before its BF16 cast. Use the dependencies
listed in the script and a new output directory:

```bash
python scripts/reference-parity/prepare_4danyone_source.py \
  --upstream /path/to/4DAnyone --source /path/to/source.mp4 \
  --prompt /path/to/prompt_context.safetensors \
  --output /path/to/source-package --seed 4196
```

The manifest explicitly remains `awaiting_motion_conditioning`. Recovered
motion, the foreground crop, four rendered skeleton videos, solved cameras,
and encoded source/pose tensors are required before this becomes a generation
bundle. Preserve the recorded canonical frame identity when preparing them.
