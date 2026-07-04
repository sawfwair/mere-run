# Changelog

All notable changes to this public repository will be documented in this file.

The format is based on Keep a Changelog.

## Unreleased

### Added

- added CoreMIDI steering for `music realtime` on macOS, including
  `--list-midi-inputs`, `--midi-input`, channel/note-offset filters, and
  repeatable `--midi-cc` mappings for Magenta RT2 note and control changes.

### Changed

- Ornith lanes (`text-agent-ornith-9b`, `text-agent-ornith-35b-mlx`) now
  generate with thinking enabled by default in `text chat` and `api serve`,
  and default to the model's published sampling (temperature 1.0, top-p 0.95,
  top-k 20) when none is set explicitly. These R1-style tunes degenerate into
  repetition loops or signature echo without reasoning. `text chat` gains
  `--no-thinking` to disable reasoning generation and `--top-k` for explicit
  cutoff control; reasoning output stays hidden unless `--thinking` is passed.
  Other Qwen-family lanes keep the existing no-think default.

### Fixed

- Qwen-family (Q35 runtime) MoE routing now renormalizes top-k router scores
  by default, matching the Qwen3.5 architecture default in HF transformers
  and mlx_lm. Checkpoints that omit `norm_topk_prob` (Ornith 35B, Qwen3.6
  nano) were previously routed with un-renormalized scores, dampening every
  MoE block's output by the top-k softmax mass and compounding into
  systematically repetition-biased logits — reasoning models looped instead
  of terminating. Per-layer hidden states now match the mlx_lm reference to
  within quantized-kernel noise, and greedy decode openings match verbatim.
- Added an env-gated reference-parity harness: the runtime dumps per-layer
  hidden-state stats (`MERERUN_Q35_DEBUG_LAYER_DUMP`) and prompt token ids
  (`MERERUN_Q35_DEBUG_PROMPT_TOKENS`), with matching mlx_lm dump/compare
  scripts under `scripts/reference-parity/`. Self-referential byte-parity
  gates cannot catch bugs that are wrong the same way on both sides.
- `api serve` chat completions now report real `prompt_tokens` in the `usage`
  object for both streaming (`stream_options.include_usage`) and non-streaming
  responses, and `total_tokens` includes the prompt side. Previously
  `prompt_tokens` was always `0` and `total_tokens` only counted completion
  tokens.

## 0.18.0 - 2026-07-01

### Added

- added `image train-lora --visualize` and `image visualize-run` for a
  loopback LoRA training dashboard that reads run manifests, loss CSVs,
  typed JSONL progress events, samples, checkpoints, and adapter artifacts.
- macOS Studio app: inline audio/video playback (AVKit / AVAudioPlayer) for
  generated speech, music, sound effects, and video; determinate download and
  generation progress with bytes, speed, and percentage; drag-and-drop file
  attachment; a searchable run library with rename and delete; and a first-run
  welcome flow.
- macOS Studio app: a `Sound FX` mode plus Advanced templates wrapping
  `sfx generate` and `sfx video`; voice-cloning controls for `speech synthesize`
  (mode / profile / reference audio / save-profile); a Hugging Face token field
  backed by `config set hf-token`; run-completion notifications; a real Run/Help
  menu; and an app↔CLI version display.
- macOS Studio app: hardened-runtime entitlements, Info.plist camera/microphone
  usage strings with a runtime camera-permission gate for `vision track-live`,
  a notarization-safe bundle layout (executable code moved out of
  `Contents/Resources`), and `scripts/package-macos.sh` plus a `macos-release`
  workflow that build, Developer ID-sign, notarize, staple, and publish the DMG.
- added `mere.run plugin { list, info, install, doctor }` for live official
  companion-plugin catalog discovery, `pipx` install previews, opt-in
  execution, and post-install manifest verification.
- added hosted Linux x86_64 CUDA release artifacts alongside the default
  x86_64/amd64 CPU tarball and Debian package lanes.
- added dedicated benchmarking docs plus local `model benchmark api-workload`
  and `model benchmark code` lanes for serving workloads and generated-code
  eval slices.
- added `vision caption --prompt-file`, `--focus`, and `--trigger-token` for
  domain-aware dataset captions and deterministic LoRA trigger prefixes.
- added FLUX.2 Klein support to `image train-lora` when `--model` resolves to a
  Klein base model, reusing the native Swift Klein LoRA trainer.
- added `image train-lora --checkpoint-interval` for FLUX.2 Klein so
  intermediate adapters can be compared instead of relying only on the final
  LoRA checkpoint.
- added the `image-klein-base-9b` managed model id for the undistilled BF16
  FLUX.2 Klein Base 9B LoRA-training target.
- added native `vision ocr --backend infinity` support for Infinity-Parser2,
  with managed `vision-ocr-infinity-flash` and explicit-pull
  `vision-ocr-infinity-pro` model IDs, plus `--infinity-runtime external` for
  upstream parser/vLLM parity checks.
- added `vision-ocr-infinity-pro-int8` as an explicit-pull native Infinity
  Pro OCR option for quality-focused evals while keeping LightOnOCR as the
  default `vision ocr` backend.
- added `text-code-north-mini` as a managed GGUF coding model target for Cohere
  Labs North Mini Code through the local `text-code` runtime.
- added `text-agent-ornith-9b` as an experimental native Qwen-family MLX/OptiQ
  coding-agent target backed by the pinned public Ornith 1.0 9B snapshot.
- added `text-agent-ornith-35b-mlx` as a local native Qwen-family MLX Q4
  coding-agent target for testing converted Ornith 1.0 35B snapshots.
- added `text-agent-ornith-35b` as a managed native GGUF coding-agent target
  for larger Ornith evals through the existing `text-code` runtime.

### Changed

- improved native Qwen-family and Gemma decode paths, including Q35 switch
  routing alignment with MLX-LM, dense-path MLXFast attention learnings, and
  faster Gemma4 greedy decode.
- locked fast Krea and Klein LoRA recipes to the proven local training step
  counts used for the current native image-training workflow.

### Fixed

- macOS Studio app: Read Image inspect/caption are no longer blocked (the CLI
  auto-downloads the vision-language model); child CLI processes (including a
  long-lived `api serve`) are terminated on app quit instead of orphaned;
  partial UTF-8 output split across pipe reads is no longer dropped; the invalid
  `finder` SF Symbol is replaced; a window minimum size prevents prompt-bar
  clipping; and library persistence failures are surfaced instead of swallowed.
- fixed `image generate --input` for FLUX.2 Klein by routing the input through
  Klein reference-image conditioning, and clarified `--ref-image` support in
  CLI help and docs.
- fixed mflux-format FLUX.2 Klein transformer loading by mapping
  `time_guidance_embed.linear_*` weights into the native Swift transformer's
  nested timestep embedder path.
- split `<think>...</think>` model output into preserved reasoning metadata and
  visible response text across native chat responses, API non-streaming output,
  and code benchmark reports, including reopened-reasoning diagnostics for
  loop-like model output.
- fixed code benchmark stop handling for generated-code eval slices.
- fixed Qwen3 Code GGUF pull validation so preferred nested hub files are
  accepted through symlinked managed install roots.
- fixed MLX CUDA BF16 sigmoid JIT smoke failures by patching the upstream CUDA
  unary kernel path during `scripts/prepare-linux-native.sh`.

## 0.17.0 - 2026-06-23

### Added

- added `mere.run image train-lora` for native Krea 2 Raw LoRA training,
  including managed `image-krea2-raw` pulls, Krea transformer LoRA injection,
  training manifests/metrics, and Krea 2 Turbo LoRA inference via
  `image generate --lora`.
- added native Krea 2 Turbo text-to-image support through the managed
  `image-krea2-turbo` model, including component-only Hugging Face pulls,
  split-transformer validation, Swift MLX MMDiT sampling, docs, and tests.
- added `mere.run video generate --end-image` and `--end-image-strength` for
  LTX start-to-end keyframe conditioning.

## 0.16.0 - 2026-06-16

### Added

- added native LTX 2.3 unified audio/video generation through the managed
  `video-ltx23-av-mlx` model, including the split MLX checkpoint layout,
  Gemma 3 prompt-conditioning companion model, separate video/audio VAE files,
  BWE vocoder, and LTX 2.3 upscaler components.
- added LTX 2.3 guidance across the CLI, runtime, model-source, and testing
  docs, with examples that pull `video-ltx23-av-mlx` and render synchronized
  audio/video at the trained 24 fps timing.
- added `scripts/compare-ltx-av-audio.py` to compare native unified-AV output
  against upstream LTX-2 media and inspect codec, sample-rate, loudness, and
  model-root hints when validating audio quality.

### Changed

- changed `video generate --duration` handling for LTX to resolve to valid
  frame counts and warn when unified-AV renders use timing that can stretch
  motion relative to audio.
- changed `model info --components` to render LTX 2.3 split-model components
  directly, including resolved symlink targets and companion text-encoder
  readiness, instead of forcing the layout through generic file enumeration.

### Fixed

- fixed LTX unified-AV MP4 assembly so native audio is written at the expected
  24 kHz path and muxed through the shared media I/O layer on both Apple and
  FFmpeg-backed platforms.
- fixed LTX model validation and managed-model metadata for the split LTX 2.3
  layout so pulls, model info, and video generation agree on the required files.

## 0.15.0 - 2026-06-13

### Added

- added native OpenAI-compatible `/v1/embeddings` to `mere.run api serve`,
  backed by `text-embed-qwen3-0.6b`, and documented Open WebUI as an optional
  companion UI with Docker and pip connection recipes.
- added OpenAI-ish `/v1/images/generations`, `/v1/images/edits`,
  `/v1/audio/speech`, and `/v1/audio/transcriptions` endpoints backed by
  native image, Qwen3-TTS, and ASR runtime paths, with installed-only sidecar
  model listing plus MP3/Opus/AAC/FLAC speech transcoding and SRT/VTT
  transcription output.
- added an Open WebUI smoke harness, native function-calling settings guidance,
  conservative model capability metadata, Open WebUI chat model filtering,
  per-model metadata wrapper import, and Open WebUI-style image edit multipart
  compatibility for `image[]` uploads plus optional masks.
- added `mere.run open-webui quickstart` to start a local mere.run API server,
  launch the official Open WebUI Docker image, and apply the same native
  chat/RAG/image/TTS/STT configuration used by the smoke harness.
- documented Docker Compose and `uvx` Open WebUI companion paths for longer-lived
  Linux/DGX Spark and same-host Python installs.

### Fixed

- fixed `scripts/install-local.sh` on macOS so local installs stage every built
  framework and SwiftPM bundle, including Magenta RT2, plus the vendored MLX
  shader bundle before invoking the packaged installer.

## 0.14.0 - 2026-06-08

### Added

- added `image generate --structured-prompt` / `--json-prompt`, an opt-in local
  text-chat adapter that expands short image prompts into typed structured JSON
  captions before generation, defaulting the adapter to
  `text-chat-gemma4-12b-4bit` with reviewable `--structured-prompt-output`
  artifacts and Ideogram-friendly prompt token budgeting.
- added Gemma structured-image JSON recovery so malformed local adapter output
  can be repaired and retried before image generation falls back.
- added runtime-pool TTL eviction so loaded API models with `ttlSeconds` unload
  after they sit idle, while pinned models stay protected from automatic
  TTL/LRU eviction.
- added runtime-pool memory-guard tiers (`off`, `safe`, `balanced`,
  `aggressive`, `custom`) so pressure LRU uses tiered soft/hard ceilings rather
  than fixed resident-memory ratios. Elevated pressure pauses extra concurrent
  admissions and evicts the least-recently-used idle unpinned model; critical
  pressure evicts every idle unpinned model.
- added API/runtime status details for active requests, admission pressure,
  cache summaries, loaded runtime entries, and per-model benchmark state so
  server health can be inspected without private tooling.
- added `music generate --source-audio` and `--reference-audio` so ACE-Step can
  run source-conditioned cover generation from regular audio files.
- added `music generate --analyze-source-audio` so ACE-Step covers can use
  5 Hz LM audio understanding to fill missing BPM, key/scale, language, and
  time-signature metadata from the source song before direct DiT generation.
- added `music analyze` so ACE-Step can inspect regular audio files and emit
  JSON metadata from the 5 Hz LM audio-understanding path before cover/remix
  workflows.
- added ACE-Step cover controls for source-latent noise, cover strength,
  reference audio, task routing, metadata hints, and lyrics-file workflows so
  covers can be tuned between faithful and style-transferred outputs.
- added ACE-Step Haar DCW sampler correction with upstream defaults for cleaner
  native diffusion output.
- added managed ACE-Step 1.5 XL Turbo pulls with `music-acestep-xl-turbo`, plus
  optional 4B 5 Hz LM installation through `music-acestep-xl-turbo-lm4b`.

### Changed

- changed ACE-Step cover prompting to preserve bracketed lyric sections and to
  let source-audio understanding fill only missing metadata, keeping user style
  prompts in control when direct values are supplied.
- changed structured image generation to keep generated JSON artifacts
  reviewable while retrying adapter output with a narrower repair prompt.

### Fixed

- fixed model-store size reporting for real directories that contain symlinked
  payload directories, and added `model info` storage layout details so wrapper
  directories no longer look like tiny installs.
- fixed native runtime backend diagnostics on stderr for text and image
  generation so MLX/Metal vs GGUF backend selection is visible during smoke
  tests.
- fixed runtime status after pressure eviction so models touched by settings or
  memory-guard state still appear even when they are no longer loaded locally.
- fixed SwiftPM target membership so the MagentaRT2 README stays out of CLI
  compilation.

## 0.13.1 - 2026-06-05

### Fixed

- fixed GGUF chat statistics so `text chat --stats` reports llama.cpp prefill
  and decode throughput when available, instead of only the CLI wrapper's
  end-to-end rate.
- fixed Linux arm64 CUDA Debian package portability by declaring the missing
  `libcufft-13-0` runtime dependency and covering it in the Linux package
  fixture.
- fixed the GB10/Spark smoke sweep so `text-chat-q36-nano-gguf` is exercised as
  its own text-chat row and no-pull speech-ASR checks do not download a TTS
  fixture by accident.

## 0.13.0 - 2026-06-05

### Added

- added native Gemma4 and Qwen-family MTP benchmarking support, including
  `--variant mtp` / `--variant no-mtp` controls, speedup reporting, JSON output,
  status surfacing, and guide coverage so decode acceleration can be compared
  from the public CLI instead of private notebooks.
- added Gemma4 MTP runtime wiring with typed draft-head loading, cache-aware
  generation, short-prompt gating, and tests around model decoding, manifest
  handling, and benchmark output.
- added Q36 MTP runtime support on the existing Qwen-family path, including
  draft-head resource discovery, benchmark coverage, and CLI guidance for when
  MTP is enabled or skipped.
- added `image-generate` support for Ideogram4 SDNQ checkpoints, including
  SDNQ quantized loaders, text-feature handling, scheduler support, VAE weight
  loading, model manifests, catalog metadata, validation, docs, and runtime
  tests.
- added native Magenta RT2 realtime music support through the new
  `music realtime` command, with prompt-conditioned streaming generation,
  playback controls, managed-model metadata, docs, and a vendored
  `magentart.xcframework` runtime.
- added `music generate` support for prompt audio controls that are shared with
  the realtime path, plus parser tests for duration, seed, temperature, guidance,
  and scheduler options.
- added richer model capability/status reporting for installed runtime support,
  recommended IDs, MTP availability, realtime music support, and SDNQ-backed
  image generation.

### Changed

- expanded the public docs, guides, and README model surface to include the new
  MTP benchmark path, Ideogram4 image runtime, and realtime music workflow.
- updated runtime model pooling so MTP-capable text runtimes and new media
  runtimes can expose their support cleanly through the CLI and API-facing
  status surfaces.
- extended managed model manifests and validators with SDNQ, Magenta RT2, and
  MTP metadata while keeping model installs typed and schema-checked.
- refreshed third-party notices for the newly bundled Magenta RT2 runtime
  artifact.

### Fixed

- fixed MLX CUDA JIT discovery so packaged Linux installs export the resolved
  CUDA CCCL include path and can find `cuda/std/*` during NVRTC compilation.
- fixed model pull and guide output around newly recommended runtime families
  so capability output, install guidance, and user-facing docs agree on the
  current public model IDs.

## 0.12.0 - 2026-06-04

### Added

- added `text-chat-gemma4-12b` and `vision-chat-gemma4-12b` managed model
  support for Google's dense Gemma 4 12B-it checkpoint, including native
  unified image preprocessing, catalog metadata, API serving, validation, and
  user-facing docs.
- added `model benchmark vlm`, a local VLM benchmark command with a tiny
  synthetic smoke suite plus an `lmms-eval` bridge for existing datasets such as
  MathVista testmini, MMMU validation, ChartQA, DocVQA validation, and MME.

### Fixed

- cleaned Gemma4 chat responses so completed or dangling hidden-thinking blocks
  do not leak into CLI or API output.

## 0.11.0 - 2026-05-31

### Changed

- the default `text chat` model is now Qwen3.6-35B-A3B, chosen hardware-aware
  via the machine's unified memory: `text-chat-q36-nano` (MLX) on Apple
  Silicon and `text-chat-q36-nano-gguf` (llama.cpp) on Linux CUDA, stepping
  down to `text-chat-gemma4-turbo` then the 4B `text-chat-gemma4-nano` on
  lower-memory machines. It replaces the gemma4-31B default, which was both the
  slowest (~1 tok/s on GB10, ~6 on M4 Max) and the largest download (62 GB);
  q36-nano is ~10x faster at comparable quality and a third of the size.
- `api serve` now defaults to a chat engine (serving `text-chat-q36-nano`)
  instead of `text-code`, so `mere.run api serve` is an OpenAI-compatible chat
  server out of the box.
- Q35 MTP speculative decode is now gated by prompt length instead of a blanket
  off-on-CUDA: it is used only above `MERERUN_Q35_MTP_MIN_PROMPT_TOKENS`
  (default 6144), where it is a ~1.5-2.5x decode win, and skipped at short
  prompts where it regresses. The env still forces it on/off.

### Added

- `text-chat-q36-nano` (Qwen3.6 35B-A3B OptiQ 4-bit MLX, with the MTP draft
  head), plus `text-chat-q36-nano-gguf` and `text-chat-q35-nano-gguf` GGUF
  variants for the llama.cpp engine.
- `text chat` now routes GGUF (`.codegenGGUF`) chat models through the
  llama.cpp engine — on Linux CUDA this uses the GB10-tuned quantized-MoE
  kernels MLX lacks (~68 tok/s on a GB10 vs ~13 for the MLX path).
- bundled `llama-server` in the Linux package for persistent GGUF serving.
- `scripts/e2e_gb10.sh`, a real per-category CUDA inference sweep that flags
  missing-kernel crashes smoke tests miss.
- `vision inspect --prompt`; `guide --list --markdown` (table with a
  Description column); model-card/voice-clone reference passage in the speech
  synthesize guide.

### Fixed

- `text-chat-q35`/`-nano` failed at chat time with "tokenizer does not have a
  chat template"; their catalog download patterns omitted `chat_template.jinja`.
- `vision caption`/`inspect` no longer claim to be "downloading" a model that
  is already cached.

### Removed

- the `text-chat-q35`, `text-chat-q35-nano`, and `text-chat-q35-nano-gguf`
  models. Qwen3.6-35B-A3B (q36-nano) supersedes the A3B tier at the same speed,
  and DeepSeek V4 Flash covers the 96-128 GB max-quality tier, so the
  Qwen3.5-122B flagship had no remaining niche. The Q35 *runtime* and the
  `text-chat-q35` serving-engine alias remain (they now serve q36-nano).

## 0.10.0 - 2026-05-29

### Fixed

- fixed managed pulls for canonical model IDs that share a Hugging Face repo by
  linking snapshot contents into per-model install roots and writing manifests
  there instead of mutating the shared Hub snapshot manifest.
- fixed Linux CUDA execution for several MLX quantized model paths by using
  dequantized matmul fallbacks when MLX CUDA lacks `quantized_matmul`,
  `GatherQMM`, or Metal-only packed binary kernels.
- fixed Linux CUDA `text code` runs by bundling the matching `llama-cli` and
  using it as an isolated subprocess on packaged Linux installs, avoiding the
  in-process llama.cpp/MLX CUDA loader collision seen on GB10 hosts.
- fixed Q35 hidden-thinking chat prompts so the runtime pre-fills an empty
  `<think>` block when `--thinking` is omitted, preventing short responses from
  spending their token budget on hidden reasoning.

## 0.9.1 - 2026-05-28

### Added

- added typed Gemma4 runtime KV cache modes (`default`, `polar2`, and `auto`)
  so runtime settings, API serving, status output, and benchmarking can promote
  decode-deferred PolarKV for longer Gemma4 prompts while keeping short prompts
  on the default cache path.

### Fixed

- fixed the Linux arm64 CUDA packaging source so CUDA `.deb` artifacts declare
  `cuda-cccl-13-0` and the MLX CUDA JIT include patch is applied after CMake
  materializes its `_deps/mlx-src` checkout.

## 0.9.0 - 2026-05-28

### Added

- added a Linux arm64 CUDA package lane to the GitHub Actions `linux-release`
  workflow for self-hosted arm64 CUDA runners, plus architecture-aware CUDA
  linker paths, cuDNN/NCCL provisioning, CUDA 13 CCCL runtime JIT include
  handling, CUDA CCCL/runtime `.deb` dependencies, and a combined checksum
  manifest for available Linux artifacts.

### Fixed

- fixed Linux patterned Hugging Face model pulls so nested files like
  `tokenizer/added_tokens.json` resolve through redirect URLs without double
  encoding and image nano models can install on Linux CUDA hosts.

## 0.8.0 - 2026-05-23

### Added

- added headless Linux release packaging for the `mere.run` CLI, including an
  x86_64 tarball, amd64 Debian package, GitHub Actions packaging workflow, and
  package manifest verification.

### Fixed

- fixed Linux tarball installs by wrapping `mere.run` with a launcher that
  resolves symlinks, sets `LD_LIBRARY_PATH` to colocated runtime libraries, and
  executes the real `mere.run-bin` payload.
- fixed Linux package runtime bundling so resolved OpenBLAS and Swift runtime
  libraries are copied as portable files instead of broken host symlinks.

## 0.7.0 - 2026-05-22

### Changed

- aligned ACE-Step music generation with the upstream `Ace-Step1.5` checkpoint
  layout by defaulting to `acestep-v15-turbo`, `acestep-5Hz-lm-1.7B`, and
  scheduler `shift=1.0` while preserving legacy turbo layout compatibility.
- changed custom ACE-Step turbo timesteps to map against the full reference
  timestep set instead of the current `fixNFE` slice.

### Fixed

- fixed managed ACE-Step pulls so upstream checkpoint directories are no longer
  renamed into the legacy `music-*` layout.
- fixed `text-chat-mebot` resolution so missing standalone MeBot installs fail
  clearly instead of falling back to Klein image models.
- fixed `demo-all-models.sh --only` so unsupported or missing model filters fail
  before the demo loop starts.

## 0.6.0 - 2026-05-20

### Added

- added `text-chat-gemma4-turbo`, a managed MLX NVFP4 Gemma 4 26B-A4B-it
  MoE tier for 32 GB Apple Silicon Macs, including catalog, manifest,
  validation, docs, and native Swift Gemma runtime wiring.
- added Gemma4 Turbo KV-cache defaults so `text-chat-gemma4-turbo` runs with
  4-bit TurboQuant KV cache from token 0 while preserving explicit KV override
  flags.
- added `demo-all-models.sh` for smoke-testing installed managed models.

### Changed

- bumped `github.com/huggingface/swift-transformers` from 1.3.2 to 1.3.3.

## 0.5.3 - 2026-05-16

### Added

- added `mere.run status` for a quick local snapshot of API health, the served
  model, the active model store, and installed managed models.

### Fixed

- fixed DeepSeek V4 Flash model resolution so existing imatrix GGUF symlinks
  are reused instead of triggering another 81 GB download.
- routed DS4 chat completions through a shared client adapter that repairs
  non-stream JSON responses containing literal control characters before they
  reach OpenAI-compatible clients.
- fixed the root CLI so `mere.run --version` reports the public release version.

## 0.5.2 - 2026-05-15

### Fixed

- fixed `mere.run model pull` so a zero-exit pull is rejected when the model is
  still not discoverable by `mere.run model list`.
- added disk-space preflight checks for managed Hugging Face pulls, including
  actionable cache/model-store relocation guidance for low-space volumes.

## 0.5.1 - 2026-05-15

### Fixed

- fixed the DS4 premier agent tier to pull the page-aligned DeepSeek V4 Flash
  imatrix GGUF and ship the matching refreshed `ds4` runtime binaries.

## 0.5.0 - 2026-05-13

### Added

- added the DS4 premier agent tier for 96 GB+ Apple Silicon Macs, including
  vendored `ds4` runtime packaging, setup readiness checks, and OpenAI-compatible
  local serving through `mere.run agent start`.
- added Pi agent installation and launch support so high-memory Macs can choose
  between the DS4 premier path and the Pi local agent runtime.
- expanded `mere.run api serve` Chat Completions compatibility with typed
  OpenAI request fields, per-engine capability validation, native function-tool
  mapping, and streaming usage chunks.

### Changed

- changed the macOS Studio first-launch behavior so dragging the app into
  Applications no longer auto-installs the terminal CLI; Settings now provides
  explicit buttons to install the bundled CLI and optional `use-mere-run` Codex
  skill.

## 0.4.13 - 2026-05-12

### Fixed

- fixed Image Nano (`image-zimage-nano`) generation from the public mflux Z-Image checkpoint by mapping mflux VAE wrapper keys onto the Swift VAE loader and aligning sigma shift values with mflux.

## 0.4.12 - 2026-05-12

### Added

- added native HiDream O1 image generation support with managed model IDs, validation, CLI reference-image options, tests, docs, and optional e2e smoke coverage.

### Changed

- bumped `github.com/huggingface/swift-transformers` from 1.3.0 to 1.3.2.

## 0.4.11 - 2026-05-11

### Fixed

- fixed public app release bundles so the packaged Studio app embeds a freshly built `mere.run` CLI payload and its runtime support assets from the same checkout.

## 0.4.10 - 2026-05-10

### Added

- added a Studio Models sheet for inspecting managed models, revealing their folders in Finder, opening the model store, and purging downloaded installs.

### Fixed

- fixed Image Nano (`image-zimage-nano`) to pull and load the public 4-bit mflux Z-Image Turbo quant layout.
- tightened Studio readiness handling so unsupported or unknown model states cannot be bypassed, model changes refresh capabilities, and preflight failures no longer leave stale running items.

## 0.4.9 - 2026-05-10

### Fixed

- fixed Studio model pulls so the missing-model overlay is replaced by download progress and readiness is rechecked after the pull completes.

## 0.4.8 - 2026-05-10

### Changed

- made Image Nano (`image-zimage-nano`) the default image model for CLI and Studio first-run workflows.

## 0.4.7 - 2026-05-10

### Changed

- updated Studio readiness so modes respect managed model capability support before offering pulls or runs

## 0.4.6 - 2026-05-09

### Fixed

- fixed app launch from `/Applications` when no custom CLI path is set by making package-root discovery stop reliably at the filesystem root

## 0.4.5 - 2026-05-09

### Fixed

- fixed Studio first-open previews so large media outputs are not read into memory as text and image previews are downsampled before display
- fixed duplicate Studio model-readiness checks from stacking during launch, mode changes, and model changes

## 0.4.4 - 2026-05-09

### Added

- added a site-matched MereRun app icon for the macOS Studio bundle

### Changed

- polished the public DMG into a drag-to-Applications layout with release support files hidden under `.mere-run`

## 0.4.3 - 2026-05-09

### Added

- added `mere.run guide`, an offline CLI cookbook reader with per-command guides across image, text, speech, vision, music, video, model management, API serving, setup, and agent workflows
- added public docs and Codex skills that point agents and users at the same guide material

## 0.4.2 - 2026-05-09

### Added

- added Hugging Face pull sources for `image-klein-nano`, `image-zimage-nano`, and `image-zimage-base`
- added Studio first-launch Terminal CLI bootstrap from the app-bundled payload

### Changed

- removed private archive/R2 model-source downloads; managed pulls now use cataloged Hugging Face snapshots only
- changed the DMG to a Studio-first layout with a separate `CLI/` payload folder and Applications shortcut
- taught `install.sh` to install from the DMG `CLI/` payload folder when present

## 0.4.0 - 2026-05-07

### Added

- third-party notices for vendored runtime artifacts
- GitHub issue templates, pull request template, Dependabot, and a lightweight security workflow
- MereRun Studio, a user-facing SwiftUI macOS app shell with a unified output canvas, mode-aware prompt bar, local Library, guided model readiness, and the technical command surface preserved behind Advanced
- Studio mode mapping for image, text chat/code, speech, vision, music, and video workflows backed by the public `mere.run` CLI
- persistent Studio library metadata under Application Support with local output previews for images, media, text, JSON, and generic files
- repeatable `MereRun.app` bundle creation through `scripts/build_mere_run_app.sh`
- streaming output support for `mere.run text chat --stream`, including live Studio canvas rendering for chat and code runs

### Changed

- clarified public contributor guidance and removed maintainer-specific workflow files from the repo surface
- documented local API serving safety defaults, advanced operator flags, and HTTPS-only remote artifact expectations
- hardened API request validation for generation parameters and operator-controlled LoRA selection
- kept `shell_exec` out of non-interactive tool auto-approval even when `--auto-approve-tools` is passed
- made signed model-source endpoint failures fail closed unless fallback is explicitly allowed
- hardened recoverable runtime construction and conditioning failures to throw typed errors instead of terminating the process
- default GUI launch instructions now use the app bundle path while keeping `swift run mere.run.app` as a contributor smoke path
- Studio CLI resolution now prefers local build products before installed binaries during development

### Fixed

- fixed DMG installs when the optional packaged model-source URL sidecar is absent
- fixed nested ASR managed-model root normalization so qwen3/parakeet alternates do not recurse indefinitely
