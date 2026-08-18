# LTX 2.5 upstream parity

mere.run implements the official LTX 2.5 inference family directly in Swift
and MLX. It does not launch Python, PyTorch, a sidecar, or an upstream script.

The compatibility target is immutable:

- code: `Lightricks/LTX-2` release `v1.2.0`, commit
  `d151147788a9284cca791edc6ce898007e727fe6`
- weights: `Lightricks/LTX-2`, revision
  `dd53cc2cd45bbeaa3563dfb575cba3f49cf44761`
- DFR detailer: `Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler`,
  revision `74c4e68ee7dd99f3997d5a1bb1a3784941822222`

When `--width` and `--height` are omitted, recipe-native geometry is preserved:
1536x1024 for standard two-stage, keyframe interpolation, and DFR; 1920x1088
for HQ; and 768x512 for dev one-stage. Explicit dimensions always win.

## Install

The standalone distilled bundle is about 71.1 GB. The complete bundle is
about 123.8 GB and adds the dev transformer, distilled LoRA, DiffVAE, temporal
upsampler, and duration head.

```bash
mere.run model pull video-ltx25-distilled-bf16 --accept-model-license
mere.run model pull video-ltx25-full-bf16 --accept-model-license
```

The model and the DFR detailer are separately gated on Hugging Face. Accepting
the main model gate does not accept the detailer gate.

```bash
mere.run adapter pull ltx25-pixel-spatial-upscaler-x2 --accept-license
```

An existing checkpoint can instead be bound from any registered model-catalog
location. Pulls verify immutable file sizes and hashes; a catalog entry alone
does not prove that every checkpoint file is installed.

## Pipeline matrix

| Official pipeline | Native mere.run surface | Coverage |
| --- | --- | --- |
| Distilled two-stage | `video generate --model video-ltx25-distilled-bf16` | Ancestral distilled schedule, x2 latent upsampling, generated AV or video-only |
| Dev plus distilled-LoRA two-stage | `video generate --model video-ltx25-full-bf16` | Guided dev stage, runtime LoRA fusion, distilled refinement |
| HQ two-stage | `video generate --model video-ltx25-full-bf16 --ltx-preset hq` | Res2s, official seeds, schedules, guidance, and LoRA strengths |
| Dev one-stage | `video generate --model video-ltx25-full-bf16 --ltx-pipeline dev-one-stage` | Target-resolution guided generation without distilled LoRA |
| IC-LoRA | repeat `--lora` and `--video-conditioning` | Stacked LoRAs, reference spatial/temporal metadata, masks, strengths, optional stage-one output |
| Keyframe interpolation | `--ltx-pipeline keyframe-interpolation` plus repeat `--image-conditioning` | Exact append-only guiding-token semantics at every pixel frame, including frame zero; strengths and checkpoint CRF |
| Generated keyframes | `--num-generated-keyframes` or repeat `--generated-keyframe` | Absolute-position keyframe slots, including DFR-owned seam slots |
| A2Vid | `video generate --audio` | Frozen source-audio latents, audio windowing, original soundtrack mux, arbitrary image guides |
| DFR | `video generate --dfr` | Generated seam layout, spatial detailing, zero to two temporal x2 refinement rounds |
| Retake | `video retake` | Timed video/audio masks, SDR containers and scene-linear EXR sequences, modality preservation controls |
| HDR IC-LoRA | `video generate --hdr ... --video-conditioning ... --lora ...` | ACEScct/LogC3 transforms, precomputed contexts, HQ frame doubling, DiffVAE, EXR plus HLG master |
| Dub-It | `video dub-it` | Reference video identity plus clean negative-time reference audio, original fractional FPS |
| Text-to-audio | `audio generate` | Audio-only denoising and decode with CFG/STG, LoRA, custom sigma, and DurationHead controls |

The upstream repository has twelve entry-point files but thirteen rows here
because generated keyframes are an independently testable checkpoint feature
shared by multiple official pipelines.

## Shared controls

The native surface includes the model-affecting upstream controls:

- arbitrary finite LoRA strengths, including zero and negative weights;
- explicit descending sigma schedules, Euler ancestral sampling, HQ Res2s,
  Bong math controls, and independent random streams;
- video/audio CFG, STG, rescale, modality guidance, block selection, and skip
  intervals;
- native Gemma 4 prompt enhancement and precomputed text contexts;
- DurationHead prediction. Omitting `--num-frames` on supported LTX 2.5
  generation uses the official 1–20 second range; an explicit frame count wins;
- fractional frame rates. The same value drives temporal RoPE, audio length,
  retake masks, and the encoded MP4 clock;
- convolutional or diffusion video VAE decode, explicit spatial tiling, HDR
  transfer, half-float EXR output, and BT.2020/HLG Main10 masters.

`--image` and `--end-image` remain convenient first/last-frame aliases. Use
repeatable `--image-conditioning PIXEL_FRAME:PATH[:STRENGTH[:CRF]]` for the
complete upstream conditioning surface.

## Native acceleration controls

The acceleration surface remains native Swift, MLX, and Metal:

- `video session` keeps the LTX 2.5 transformer, decoders, and LoRA adapters
  resident. Its exact-key bounded prompt cache retains materialized Gemma 4
  connector outputs; repeat prompts avoid text-encoder reload and execution.
- `--ltx-transformer-execution compiled` reuses one compiled MLX graph across
  all 48 V2 transformer blocks by rebinding each block's parameters. Eager is
  the compatibility default because fusion order can change floating-point
  results slightly.
- `--ltx-guidance-projection-cache automatic|enabled` materializes the positive
  prompt's per-block text K/V projections once per denoising evaluation and
  reuses them across STG and modality-guidance passes. It has an explicit
  unified-memory reserve and falls back rather than overcommitting memory.
- `--ltx-teacache` enables calibrated block-residual reuse for the full
  two-stage Euler and HQ Res2s paths. First and last steps always compute;
  conditioned, unconditional, perturbed, and isolated guidance branches reuse
  their own residuals under one synchronized guidance-group decision. Res2s
  primary and midpoint evaluations keep separate state. TeaCache is opt-in and
  requires eager execution. `--ltx-teacache-threshold` makes the speed/quality
  tradeoff explicit, while
  `--ltx-teacache-calibration-output` records full-compute drift data without
  skipping any block. Five corrected-checkpoint trajectories fit the maximum
  drift across each synchronized guidance group; defaults are `0.235` for
  Euler and `0.39` for Res2s. This is a deterministic approximate mode, not a
  pixel-identical acceleration, and remains disabled unless requested.
- DiffVAE neighborhood attention routes to a fused three-dimensional Metal
  online-softmax kernel on supported Apple GPUs. The bounded MLX SDPA tiling
  path remains the automatic compatibility fallback.
- `mere.run model optimize video-ltx25-full-bf16` creates source-bound native
  transformer and compact text-connector packs under
  `.mere-run/ltx25-native-v1`. It streams tensor bytes in physical file order,
  rewrites transformer keys to the native module namespace, and does not
  quantize or rewrite payload bytes. Loading applies the same requested BF16
  normalization as the official path, including the checkpoint's FP32 tensors.
  Connector loading no longer has to touch the 42 GB official transformer
  checkpoint.

TeaCache and guidance-projection caching are not stacked: enabling TeaCache
uses the exact projection path against which its drift calibration was
measured. Timing JSON reports cache builds/reuses/fallbacks, TeaCache decisions,
and computed versus reused transformer stacks.

The residual-reuse policy follows the
[TeaCache paper](https://arxiv.org/abs/2411.19108) and
[official implementation](https://github.com/ali-vilab/TeaCache), with new
coefficients measured against the pinned LTX 2.5 checkpoint and this native
transformer rather than copied from an older LTX release.

## Hardware-specific upstream options

The following upstream switches are CUDA/PyTorch execution strategies, not
model capabilities, and therefore have no one-for-one Apple-Silicon switch:

| Upstream option | Native disposition |
| --- | --- |
| CPU/disk offload | Swift loaders stream large safetensors and release phases under unified-memory admission |
| FP8 and NVFP4 policies | The pinned native catalog uses the BF16 checkpoints; no silent precision substitution |
| `torch.compile` | `--ltx-transformer-execution compiled` provides shared compiled MLX block execution; eager remains the exact-default path |
| DiffVAE CUDA optimization presets | Apple GPUs use the native fused 3D Metal neighborhood-attention kernel with an MLX fallback; CUTLASS/CuTe remain CUDA-only |
| `max_batch_size` | `--ltx-guidance-projection-cache` removes repeated positive-context projections without CUDA transfer batching and preserves a unified-memory reserve |
| prompt-enhancer static KV cache | Resident `video session` prompt embeddings provide exact reuse after the first materialization |
| multi-GPU runners | Apple Silicon uses one unified-memory device |

The upstream HDR command also accepts a directory for batch convenience.
mere.run exposes the full single-item model workflow through CLI, Studio, and
the local API; shell or graph workflows provide batching without a second
inference implementation.

## API and Studio

`POST /v1/videos/generations` accepts the same flags through its `options`
object, so new model controls do not require a parallel Python schema. The
macOS Studio exposes primary LTX 2.5 controls directly and retains an extra
arguments field for the complete CLI surface. Capabilities advertise native
LTX 2.5, source/reference media, audio-video output, HDR, Retake, Dub-It, and
text-to-audio support.

## Validation boundary

`./scripts/check.sh` verifies formatting, build, unit/integration tests, every
command help surface, and repository hygiene. Loader and recipe tests use
typed synthetic fixtures. A real-checkpoint smoke is a separate hardware
qualification step and should record the exact model root, output, media
tracks, frame geometry, duration, and command revision.

Two opt-in tests validate the installed official tensor layouts without making
the 123.8 GB checkpoint a normal CI dependency:

```bash
MERERUN_TEST_LTX25_ROOT=/path/to/LTX-2.5 \
  swift test --filter \
  LTXAudioToVideoSupportTests/testInstalledLTX25AudioEncoderCheckpointWhenAvailable

MERERUN_LTX25_DIFFVAE_WEIGHTS=/path/to/LTX-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
  swift test --filter \
  LTXDiffusionVideoDecoderTests/testInstalledOfficialCheckpointMetadataCoversEveryNativeParameter
```

The release qualification set covers video-only two-stage generation,
generated synchronized audio-video, source-audio A2Vid with soundtrack
readback, arbitrary timed keyframe interpolation, Retake with preserved audio,
and text-to-audio. `model info --components` also verifies the complete packed
full-model layout. An explicit external binding gets catalog-derived model info
without writing a manifest into its read-only checkpoint directory.

The separately gated DFR detailer cannot be checkpoint-smoked until the
account behind `HF_TOKEN` has accepted that adapter repository's gate.
