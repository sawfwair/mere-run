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

## Hardware-specific upstream options

The following upstream switches are CUDA/PyTorch execution strategies, not
model capabilities, and therefore have no one-for-one Apple-Silicon switch:

| Upstream option | Native disposition |
| --- | --- |
| CPU/disk offload | Swift loaders stream large safetensors and release phases under unified-memory admission |
| FP8 and NVFP4 policies | The pinned native catalog uses the BF16 checkpoints; no silent precision substitution |
| `torch.compile` | Not applicable to MLX execution |
| DiffVAE CUDA optimization presets | The architecture and tiled decode are native; CUTLASS/CuTe presets are CUDA-only |
| `max_batch_size` | Native guidance passes use the MLX execution plan rather than CUDA transfer batching |
| prompt-enhancer static KV cache | Performance-only; prompt enhancement output is unchanged |
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
