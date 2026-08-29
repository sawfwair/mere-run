# LTX 2.5 upstream parity

`mere.run` implements the official LTX 2.5 inference family directly in Swift
and MLX. It does not launch Python, PyTorch, a sidecar, or an upstream script.

The compatibility target is immutable:

- **Code:** `Lightricks/LTX-2` release `v1.2.0`, commit
  `d151147788a9284cca791edc6ce898007e727fe6`
- **Weights:** `Lightricks/LTX-2.5`, revision
  `dd53cc2cd45bbeaa3563dfb575cba3f49cf44761`
- **Managed distilled distribution:**
  `Sawfwair/LTX-2.5-Distilled-BF16-MLX-Q4-Text`, revision
  `cf8a174746cd14796c81ca2b54e035dc32e69bd8`
- **Managed full distribution:** `Sawfwair/LTX-2.5-Full-BF16-MLX`, revision
  `ac74d124f7211fc3cb8b32f418a08d8e71655c8d`
- **DFR detailer:** `Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler`,
  revision `74c4e68ee7dd99f3997d5a1bb1a3784941822222`

When `--width` and `--height` are omitted, recipe-native geometry is preserved:
1536 x 1024 for standard two-stage, keyframe interpolation, and DFR;
1920 x 1088 for HQ; and 768 x 512 for dev one-stage. Explicit dimensions take
precedence.

## Install

The public self-contained managed distilled bundle is about 53.9 GB. The
managed Full BF16 bundle is about 119.7 GB and adds the dev transformer,
distilled LoRA, DiffVAE, temporal upsampler, and duration head. Both packages
ship transformers directly in mere.run's native module-key layout, without a
second source checkpoint or post-install re-keying step.

```bash
mere.run model pull video-ltx25-distilled-bf16 --accept-model-license
mere.run model pull video-ltx25-full-bf16 --accept-model-license
```

The managed distilled model is ungated. The managed full/dev model and DFR
detailer are separately gated on Hugging Face; accepting either one does not
accept the other.

```bash
mere.run adapter pull ltx25-pixel-spatial-upscaler-x2 --accept-license
```

An existing checkpoint can instead be bound from any registered model-catalog
location. Pulls verify immutable file sizes and hashes; a catalog entry alone
does not prove that every checkpoint file is installed.

## Pipeline matrix

| Official pipeline | Native `mere.run` surface | Coverage |
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

The upstream repository has twelve entry-point files but thirteen table rows
because generated keyframes are an independently testable checkpoint feature
shared by multiple official pipelines.

## Shared controls

The native surface includes the model-affecting upstream controls:

- Arbitrary finite LoRA strengths, including zero and negative weights.
- Explicit descending sigma schedules, Euler ancestral sampling, HQ Res2s,
  Bong math controls, and independent random streams.
- Video and audio CFG, STG, rescale, modality guidance, block selection, and skip
  intervals.
- Native Gemma 4 prompt enhancement and precomputed text contexts.
- DurationHead prediction. Omitting `--num-frames` on supported LTX 2.5
  generation uses the official 1–20 second range; an explicit frame count takes
  precedence.
- Fractional frame rates. The same value drives temporal RoPE, audio length,
  retake masks, and the encoded MP4 clock.
- Convolutional or diffusion video VAE decode, explicit spatial tiling, HDR
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
- The public, immutable managed Distilled and Full distributions store their
  transformers directly in mere.run's native module-key namespace. Tensor
  payload bytes and BF16 precision are unchanged; no source transformer is
  retained beside a generated cache. Distilled uses MLX affine Q4/group-64 for
  eligible Gemma language weights while retaining the LTX projection and
  tokenizer assets in BF16/raw form. Full retains the official BF16 text tower.
  `mere.run model optimize` remains available for compatible official or
  offline roots and treats a managed pre-keyed package as already optimized.
- Prompt tensors are evaluated and cached before the Gemma tower is unloaded.
  Image/reference latents are likewise evaluated before the video VAE encoder
  is released. Neither encoder remains resident during denoising; a later
  session request reloads it only when its prompt or conditioning is not cached.
  Set `MERERUN_LTX_MEMORY_TRACE=1` to print MLX active, cache, and peak memory at
  the duration, text, image-conditioning, and denoising phase boundaries.

TeaCache and guidance-projection caching are not stacked: enabling TeaCache
uses the exact projection path against which its drift calibration was
measured. Timing JSON reports cache builds/reuses/fallbacks, TeaCache decisions,
and computed versus reused transformer stacks.

The residual-reuse policy follows the
[TeaCache paper](https://arxiv.org/abs/2411.19108) and
[official implementation](https://github.com/ali-vilab/TeaCache), with calibrated
coefficients measured against the pinned LTX 2.5 checkpoint and this native
transformer rather than copied from an older LTX release.

## Hardware-specific upstream options

The following upstream switches are CUDA/PyTorch execution strategies, not
model capabilities, and therefore have no one-for-one Apple-Silicon switch:

| Upstream option | Native disposition |
| --- | --- |
| CPU/disk offload | Swift loaders stream large safetensors and release phases under unified-memory admission |
| FP8 and NVFP4 policies | Transformer inference stays on the pinned BF16 checkpoint; the optional source-bound text-tower cache is explicit MLX Q4 |
| `torch.compile` | `--ltx-transformer-execution compiled` provides shared compiled MLX block execution; eager remains the exact-default path |
| DiffVAE CUDA optimization presets | Apple GPUs use the native fused 3D Metal neighborhood-attention kernel with an MLX fallback; CUTLASS/CuTe remain CUDA-only |
| `max_batch_size` | `--ltx-guidance-projection-cache` removes repeated positive-context projections without CUDA transfer batching and preserves a unified-memory reserve |
| prompt-enhancer static KV cache | Resident `video session` prompt embeddings provide exact reuse after the first materialization |
| multi-GPU runners | Apple Silicon uses one unified-memory device |

The upstream HDR command also accepts a directory for batch convenience.
`mere.run` exposes the full single-item model workflow through CLI, Studio, and
the local API; shell or graph workflows provide batching without a second
inference implementation.

## API and Studio integration

`POST /v1/videos/generations` accepts the same flags through its `options`
object, so additional model controls do not require a parallel Python schema. The
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
