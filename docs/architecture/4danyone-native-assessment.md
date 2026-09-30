# 4DAnyone native Swift and MLX assessment

This assessment is for mere.run contributors deciding how to implement
4DAnyone on Apple Silicon. It records a source and checkpoint-header review
completed on September 15, 2026, and links the subsequent implementation and
trained-checkpoint validation. The initial assessment alone does not establish
inference compatibility, output quality, memory requirements, or speed on a Mac.

## Decision

**Proceed with a native multiview generation proof, starting from frozen
conditioning. Treat automatic motion recovery and dynamic Gaussian
reconstruction as separate implementation milestones.**

The generation model is a plausible extension of the Wan computation already
in mere.run. It needs a dedicated transformer block and orchestration path.
The automatic video-to-human pipeline requires additional models, geometry,
and license decisions. A complete, generally usable 4D reconstruction feature
is a substantially larger project.

The first useful proof consumes a prepared source clip, target skeleton
conditioning, camera metadata, and the released prompt tensor. It produces
synchronized target videos entirely through Swift and MLX. Reference
preparation remains outside that proof and must be identified in its receipt.
This demonstrates native generation, not automatic native video-to-4D.

Use the Base model for the initial implementation. Evaluate Turbo separately
under its noncommercial license. The Base choice alone does not resolve the
licensing of motion recovery or the skeleton schema.

## Evidence and scope

### Implementation follow-up

The first [native contributor API](../../Sources/MereRunCore/FourDAnyone/README.md)
now implements the Base transformer, RGB pose encoder, strict checkpoint
loading, frozen prompt loading, camera-group routing, and denoising from
prepared tensors. A decode helper connects the output to the existing
Wan2.2 VAE. There is no managed model entry or public CLI command yet.

Small CPU/FP32 fixtures execute the pinned upstream Python graph with synthetic
weights. They cover direct and packed targets, block outputs, the full-width
pose encoder, nonzero null features, and four-step routed denoising. The
production checkpoint schema is checked against all 1,182 released keys.
Subsequent [trained-checkpoint validation](./4danyone-trained-validation.md)
passes FP32 transformer, pose, and short-clip VAE comparisons on the Metal GPU.
The default now uses FP32 arithmetic with BF16 parameter storage and matches
the FP32 reference at the tested grid. Pure BF16 arithmetic fails the declared
precision threshold and remains unqualified. Complete 24-step visual generation
and CUDA/BF16 parity are still
unverified. The initial source review and its baseline inventory below remain
distinct from those implementation checks.

### Initial assessment

| Source | Reviewed revision | Purpose |
|---|---|---|
| mere.run | `99becddba5f73ddfbde8dfb740d8ff75c60bcbee` | Source baseline from `origin/main` |
| [4DAnyone source][upstream] | `8cd60c40d90882de07645cc435dcf24bc9b4fbd1` | Released inference graph and preprocessing |
| [4DAnyone assets][assets] | `4c80e87b805a5f8461cf339cdbe2fb4249e585aa` | Model header and asset-specific license declaration |
| [GVHMR source][gvhmr] | `6ec3ca39336c50492c0fae65fba2fb831fc7d866` | Dependency revision recorded by 4DAnyone |
| BiRefNet | `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` | Foreground model pinned in upstream `assets.py` |

The [paper][paper] describes multiview synthesis followed by dynamic Gaussian
reconstruction. The released [Nerfstudio guide][nerfstudio] exports one
synchronized frame and trains a static Splatfacto scene. The source README
lists open-source 4DGS reconstruction integration as unfinished. Static
reconstruction of individual frames does not establish a coherent dynamic
Gaussian model.

The initial review inspected source files, read the model's safetensors header through
bounded HTTP range requests, and compared architecture constants. It did not
download checkpoint payloads, install the Python runtime, or run inference.
The numerical inventory is recorded in
[the assessment evidence](./4danyone-native-evidence.json).

## Released generation contract

The [configuration][config] fixes generation at 121 frames, 704 pixels wide,
and 1,280 pixels high. The latent shape per view is `[48, 31, 80, 44]`.
Patches have shape `[1, 2, 2]`, producing 27,280 tokens per view.

The pipeline performs these stages:

1. Select a canonical clip and preserve its sampling clock.
2. Recover human motion using the GVHMR static-camera path.
3. Estimate foreground masks with BiRefNet. Apply SMPL-X, regress the required
   3D landmarks, solve framing, and render target skeleton videos.
4. Encode the source with the Wan2.2 VAE and the skeletons with PoseEncoder.
5. Generate proposal views when reference context packing is active.
6. Decode proposals, round-trip four references through JPEG, and encode them
   again for target generation.
7. Denoise camera groups with target context routing, then publish target
   videos, camera data, motion artifacts, and run metadata.

The [view planner][views] accepts groups of four or six. It activates reference
context packing only when the total requested views exceeds six. The standard
24-view layout uses six proposal views followed by four groups of six targets.
Public camera IDs remain stable while denoising groups change.

The fixed prompt context has BF16 shape `[1, 512, 4096]`. Loading that tensor
avoids shipping or running UMT5 for the released generation contract. Freeform
prompts would require separate validation.

## Reuse and missing computation

The following table records infrastructure and missing model work at the
assessed mere.run baseline, before the implementation follow-up above.

| Component | mere.run source | Assessment |
|---|---|---|
| Wan transformer primitives | [`Wan2Transformer.swift`](../../Sources/MereRunCore/Wan2/Wan2Transformer.swift) | Matching dimensions: 30 blocks, 3,072 hidden channels, 24 heads, and 14,336 feed-forward channels. Reuse primitives after parity checks. |
| Wan2.2 VAE | [`Wan2VAE.swift`](../../Sources/MereRunCore/Wan2/Wan2VAE.swift) | Strong reuse candidate. All 48 mean and standard-deviation constants match the upstream VAE. Full video and precision parity remain untested. |
| Checkpoint loading | [`Wan2ModelLoader.swift`](../../Sources/MereRunCore/Wan2/Wan2ModelLoader.swift) and `MereRunTensor` | Reuse streaming safetensors infrastructure. Add strict key coverage and explicit layout conversion for this checkpoint. |
| Multiview transformer | No 4DAnyone implementation found | Add learned multiview attention and modulation in every block, source packing, source timesteps, and pose feature injection. |
| RGB PoseEncoder | No equivalent found | Port the small 3D convolution network, including temporal prefix, padding, stride, and output scale. |
| Motion recovery | [`NativePoseDetector.swift`](../../Sources/MereRunCore/Pose/NativePoseDetector.swift) | Returns normalized 2D Vision landmarks. It does not produce GVHMR motion or SMPL-X meshes and cannot replace this stage without new evidence. |
| Foreground and body geometry | No BiRefNet, GVHMR, or SMPL-X implementation found | Separate model ports and geometry work. Preserve the conditioning schema and rendering behavior. |
| Camera and point export | [`MultiViewGeometryExporter.swift`](../../Sources/MereRunCore/Geometry/MultiViewGeometryExporter.swift) | Reuse serialization patterns. Its DA3 manifest explicitly sets `containsGaussianParameters` to `false`. |
| Dynamic Gaussian training and rendering | No corresponding runtime found | A separate optimizer, differentiable renderer, temporal representation, and artifact/player contract are required. |

SCAIL-2 offers another example of native conditioned-video orchestration, but
its 14B model and 16-channel Wan2.1 VAE do not match 4DAnyone.

### Checkpoint inventory

The released file contains 1,182 BF16 tensors. Counts come from header shapes,
not the model's name or an estimated parameter multiplier.

| Partition | Tensors | Parameters |
|---|---:|---:|
| Wan-shaped base partition | 825 | 4,999,787,712 |
| Multiview attention and modulation | 330 | 1,133,291,520 |
| PoseEncoder | 23 | 3,469,271 |
| ViewPack projections | 4 | 11,802,624 |
| Total | 1,182 | 6,148,351,127 |

The tensor payload is 12,296,702,254 bytes, approximately 12.30 GB or 11.45 GiB.
The base partition's familiar shapes do not establish that its trained values
equal the base Wan checkpoint. Load the released 4DAnyone weights.

## Compatibility details that need explicit tests

### Attention order and source packing

The [released transformer][dit] runs video self-attention, multiview attention,
text cross-attention, and the feed-forward network in that order. Multiview
attention changes the layout from views containing video tokens to frames
containing tokens from several views. DreamX projective camera attention has
different semantics and cannot substitute for this branch.

Reference packing appends the original source and, for dense generation, one
packed grid assembled from four projected references. Source timesteps are
zero. Their hidden states still interact with target views in each block, so
zero timesteps do not justify caching complete source hidden states across
denoising steps.

Preserve the learned `proj_2x` packing, replicate padding, cropping, and patch
channel order. Account explicitly for the checkpoint's unused `proj_4x`
partition. Do not use a permissive loader to discard unexplained keys.

### Scheduler and Turbo

The upstream scheduler samples `steps + 1` sigmas from one to zero, drops the
last entry, applies shift five, and steps to terminal zero. Its four-step sigma
sequence is `[1, 0.9375, 0.8333333, 0.625, 0]`.

The default constructor in
[`Wan2Generation.swift`](../../Sources/MereRunCore/Wan2/Wan2Generation.swift)
produces approximately `[1, 0.9097072, 0.7173175, 0.0244141, 0]` for four steps.
Use a dedicated schedule or prove an explicit-timestep configuration against
the reference. Matching step count and shift is insufficient.

Base uses 24 steps. Turbo uses four steps and a different routing profile.
For a six-view group, Turbo routing offsets are `0, 2, 4, 0`. Port the
[routing implementation][routing], including its ordering across pitch layers.

The [Turbo fusion code][turbo] applies both low-rank matrix deltas and direct
weight or bias differences. It merges in FP32, then copies into the model's
BF16 storage. Preserve its exact adapter identity and reject double fusion.
This is more specific than generic LoRA loading.

### Pose, pixels, and numeric precision

- [PoseEncoder][pose-encoder] prepends three copies of the first frame before
  its convolution stack. Its ordinary convolutions use symmetric padding.
  `Wan2VAECausalConv3D` uses causal temporal padding and is not a direct
  replacement.
- Null pose features come from encoding an all-minus-one input. Learned
  biases mean that substituting zero feature tensors changes conditioning.
- The reference preserves a JPEG quality-85 boundary between proposal decoding
  and reference re-encoding. Preserve it for parity before evaluating a
  lossless alternative.
- Skeleton color, depth ordering, thickness, crop, camera transforms, and
  frame sampling affect the learned condition. Visually similar skeletons
  are insufficient evidence of matching inputs.
- Upstream uses BF16 VAE weights and CUDA autocast. The native VAE loader
  defaults to FP32. Compare source latents and decoded video before choosing
  the native precision policy.
- Upstream computes rotary operations in FP64 before casting back. The native
  Wan path uses FP32 rotary computation. Validate the cast boundaries and
  attention outputs with reference tensors; do not assume matching formulas
  imply matching numerical behavior.
- Capture initial noise tensors for cross-runtime comparisons. Equal integer
  seeds do not establish equal PyTorch and MLX random samples.

## Memory and performance

The following values are arithmetic from released shapes. They are not measured
Mac peaks and must not be added blindly: several allocations occur in different
stages or share storage.

| Allocation | BF16 bytes | Approximate size |
|---|---:|---:|
| One pose feature, `[3072, 31, 40, 22]` | 167,608,320 | 160 MiB |
| Pose features for 24 target views | 4,022,599,680 | 3.75 GiB |
| Full six-proposal and 24-target pose banks, including three null entries | 5,531,074,560 | 5.15 GiB |
| One dense token tensor, six targets plus two packed sources | 1,340,866,560 | 1.25 GiB |
| One feed-forward hidden tensor for those eight views | 6,257,377,280 | 5.83 GiB |
| Canonical latents for 24 targets | 251,412,480 | 240 MiB |

Materializing the eight-view video self-attention score tensor in BF16 would
require about 266 GiB. Use a memory-efficient attention kernel. Then measure
the actual MLX kernel and temporary allocations at the production sequence
length; a small fixture cannot prove that this workload fits.

Upstream CPU offload still consumes the same physical memory pool on Apple
Silicon. Bound pose caches, stage lifetimes, and decoded media. Start with
sequential view groups and evaluate feed-forward chunking after parity.
Quantizing weights alone does not remove the large activation tensors.

For engineering planning, use a 64 GB or larger Mac for the first full-size
proof, with 128 GB providing more diagnostic headroom. These are proposed test
targets, not minimum supported configurations. Treat 32 GB feasibility as an
unanswered measurement question.

The [upstream performance report][performance] lists RTX 4090 SDPA denoising
times of 2.71 minutes for six Turbo views and 16.20 minutes for 24 Turbo views.
It reports about 22.18 GB of peak CUDA allocated memory. Preprocessing is
reported separately. Those measurements do not include proof of Mac speed or
complete dynamic reconstruction, and they do not support a cross-device
speedup estimate.

## License boundaries

The [asset license declaration][asset-license] and [third-party notices][notices]
assign terms per component. An Apache license on the main checkpoint does not
cover the entire automatic workflow.

| Component | Declared terms | Integration consequence |
|---|---|---|
| 4DAnyone checkpoint, Wan VAE, frozen prompt | Apache-2.0 | Retain provenance and applicable notices in converted packs. |
| Turbo adapter | CC BY-NC-SA 4.0 | Keep separate from the Base pack and from any general commercial-use claim. |
| GVHMR source and checkpoint | Research, educational, and nonprofit use; commercial use prohibited | An unrestricted end-to-end product needs appropriate permission or a replacement, followed by conditioning validation. |
| HMR2 and ViTPose assets | MIT and Apache-2.0, respectively | Preserve individual asset notices. |
| YOLOv8 detector | AGPL-3.0 | Resolve distribution and integration terms or qualify an alternative. |
| SMPL-X body assets | Separate licensed acquisition | Availability in a download workflow does not establish redistribution rights. |
| MHR70 regression tensors and schema | Apache-2.0 tensors; Sapiens2 terms for names, ordering, and structural material | Track these separately. The notices identify use restrictions in the schema agreement. |
| BiRefNet | MIT | Still requires a native model implementation or separately qualified substitute. |

The native rewrite does not by itself remove source, model, or asset
obligations. Start with technical reference work within applicable terms.
Resolve the complete asset set before presenting a commercial-ready pipeline.

## Proposed implementation boundaries

Keep the experimental runtime in `Sources/MereRunCore/FourDAnyone/`, with a
subsystem README. Reuse Wan primitives inside the owning library where their
behavior matches. A dedicated block makes its operation order explicit and
keeps DreamX's different attention contract understandable. No new package
target is needed to begin the proof.

Use separate typed records for prepared conditioning, view plans, generation
results, and optional reconstruction results. The conditioning record needs
camera IDs, coordinate conventions, sampling times, preprocessing provenance,
and hashes of the exact source and pose inputs.

Put request validation and execution in a Core operation. CLI, API, graph, and
Studio callers can consume that operation after it is qualified. Target videos
and cameras are a multiview generation result; only attach a dynamic Gaussian
artifact after a reconstruction runtime produces and verifies it.

Package converted assets with exact source revisions, hashes, tensor mappings,
precision, and license metadata. Keep fixed prompt conditioning as a small
asset. A converted VAE can be shared only after its values and conversion
contract are verified. Do not require an unused T5 installation for convenience.

## Ordered milestones and acceptance

| Milestone | Deliverable | Evidence required to proceed |
|---|---|---|
| 1. Reference fixtures | Frozen source/pose inputs, camera order, initial noise, and intermediate tensors from the pinned implementation | Fixture provenance and redistribution terms; exact shapes and input hashes |
| 2. Native computation | Strict loader, PoseEncoder, reference packing, one block, full transformer, and schedule | Compare independent upstream outputs at each boundary; record tolerances, errors, and precision |
| 3. Direct multiview proof | Six Base target videos from prepared conditioning | Full 121-frame output, camera alignment, output inspection, repeated runs, and measured Mac memory and latency |
| 4. Dense generation | 24 views with reference packing and routing | JPEG boundary, route ordering, cross-group consistency, cancellation, stage recovery, and bounded memory |
| 5. Automatic preparation | Native foreground, motion, body geometry, and skeleton rendering | Licensing resolved; compare tracked boxes, 2D/3D motion, framing, and conditioning pixels |
| 6. Product integration | Shared operation, catalog, model guide, CLI/API/graph/Studio contracts | Closest contract tests and `./scripts/check.sh`; explicit artifact and capability claims |
| 7. Dynamic reconstruction | Defined 4D representation, optimizer, renderer, export, and playback | Camera/time-aligned reconstructions and temporal quality evaluation against the selected reference method |

After Milestone 2, decide whether full-size memory and attention throughput
justify continuing on the target Mac. After Milestone 4, evaluate whether the
multiview result is useful independently of automatic preparation. License
clearance and reference-fixture acquisition can proceed alongside the native
computation work without claiming either is already complete.

## Validation status

Completed: source review, checkpoint-header inventory, shape arithmetic,
comparison of all 96 VAE normalization constants, scheduler comparison, and
checks of this report's local links and whitespace. The native Base prototype
and nine new tests now pass the repository gate:

- `MERERUN_SWIFT_DISABLE_INDEX_STORE=1 ./scripts/check.sh` exited successfully.
- SwiftLint reported zero violations in 2,083 files.
- XCTest ran 4,352 tests with 316 skips and zero failures. Swift Testing ran
  another 51 tests with zero failures.
- All nine 4DAnyone tests passed on MLX CPU. Transformer stages match the
  upstream FP32 fixture within `3e-5` maximum absolute error, the four-step
  routed result within `5e-5`, and the full-width pose encoder within `2e-4`.
- The fixture and upstream source hashes match the checked-in manifest.

See the [native verification record](./4danyone-native-verification.json) for
the scope and remaining checks.

Not performed: loading trained checkpoint payloads, GPU inference, BF16
qualification, production-resolution generation, VAE parity, video output
review, hosted CI, or PR publication. The changes are local and uncommitted.
Automatic motion recovery and dynamic Gaussian reconstruction remain separate
milestones. These local checks establish a native computation prototype, not
a qualified video-to-4D release.

[upstream]: https://github.com/ant-research/4DAnyone/tree/8cd60c40d90882de07645cc435dcf24bc9b4fbd1
[assets]: https://huggingface.co/AntResearch/4DAnyone/tree/4c80e87b805a5f8461cf339cdbe2fb4249e585aa
[gvhmr]: https://github.com/zju3dv/GVHMR/tree/6ec3ca39336c50492c0fae65fba2fb831fc7d866
[paper]: https://arxiv.org/abs/2608.20335
[nerfstudio]: https://github.com/ant-research/4DAnyone/blob/8cd60c40d90882de07645cc435dcf24bc9b4fbd1/docs/nerfstudio.md
[config]: https://github.com/ant-research/4DAnyone/blob/8cd60c40d90882de07645cc435dcf24bc9b4fbd1/fdanyone/config.py
[views]: https://github.com/ant-research/4DAnyone/blob/8cd60c40d90882de07645cc435dcf24bc9b4fbd1/fdanyone/views.py
[dit]: https://github.com/ant-research/4DAnyone/blob/8cd60c40d90882de07645cc435dcf24bc9b4fbd1/fdanyone/vendor/diffsynth/models/wan_video_dit.py
[routing]: https://github.com/ant-research/4DAnyone/blob/8cd60c40d90882de07645cc435dcf24bc9b4fbd1/fdanyone/model/routing.py
[turbo]: https://github.com/ant-research/4DAnyone/blob/8cd60c40d90882de07645cc435dcf24bc9b4fbd1/fdanyone/model/turbo_lora.py
[pose-encoder]: https://github.com/ant-research/4DAnyone/blob/8cd60c40d90882de07645cc435dcf24bc9b4fbd1/fdanyone/vendor/diffsynth/models/wan_video_pose_encoder.py
[performance]: https://github.com/ant-research/4DAnyone/blob/8cd60c40d90882de07645cc435dcf24bc9b4fbd1/docs/inference_performance.md
[asset-license]: https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/LICENSE.md
[notices]: https://github.com/ant-research/4DAnyone/blob/8cd60c40d90882de07645cc435dcf24bc9b4fbd1/docs/THIRD_PARTY_NOTICES.md
