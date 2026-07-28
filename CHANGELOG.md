# Changelog

All notable changes to this public repository will be documented in this file.

The format is based on Keep a Changelog.

## 0.28.0 - 2026-07-28

This release turns the optional macOS Studio into a complete, first-class
creative and operations client over the public `mere.run` CLI while preserving
the headless runtime and Relay/node ownership boundaries.

### Added

- added dedicated Studio workspaces for image editing, adapters, Magenta RT2
  realtime music, SCAIL-2 subject continuity, 3D creation, advanced vision,
  voice and voice cloning, SFX/Foley, model and adapter training, and structured
  text/image utility workflows. Each surface uses typed CLI contracts, durable
  Library jobs, artifact previews, and copyable commands instead of reimplementing
  inference in the app.
- added a complete SCAIL-2 subject workflow with multi-subject references and
  selectors, SAM preview/full-video tracking, immutable correction keyframes,
  before/after playback, continuity controls, and durable output inspection.
- added native 3D creation experiences for TripoSR, TRELLIS.2 PBR
  reconstruction, and ordered InstantMesh multiview input, including
  engine-specific controls, immutable run directories, manifest statistics,
  and embedded Quick Look model previews.
- added an Advanced Vision Lab for face and pose overlays, optical-flow vectors,
  live tracking and depth-video review, geometry point clouds, and preservation
  of JSON, EXR, mask, camera, and 3D sidecars.
- added production Voice, SFX, Training, Music Tools, and Utility labs with
  recording and reusable voice profiles, streaming feedback, transcription,
  Foley conditioning, CLAP scoring, waveform inspection, ACE-Step analysis,
  MuScriptor piano-roll review, LoRA/LoKr dataset previews and live metrics,
  vector comparison, and protected-PII review.
- added a top-level Serving & Agents console with API preflight and lifecycle,
  external-server reconnection, LAN/auth safety, model residency and runtime
  policy, memory/CPU/Metal/thermal telemetry, request/cache/batching activity,
  and typed Pi agent readiness, installation, configuration, and sessions.
- added a top-level Runs & Operations workspace for local durable runs and Relay
  jobs, including polling, inspection, artifact fetch, reveal, cancellation,
  immutable retry, Relay profile setup, and device sign-in.
- added a first-class Plugin Manager with catalog search, channel selection,
  installed version/path and manifest verification, copyable pinned commands,
  confirmed install/update, reveal, doctor, and custom catalog support.
- added Model Health & Repair with structured manifest audits, confirmed repair
  of missing known manifests, correctness/performance quality suites, and
  durable JSON reports.
- added structured CLI data needed by these thin clients: runtime process and
  model-pool telemetry, agent status, enriched `plugin list --json`
  installation/verification state, and `model repair-manifests --json`
  preview/apply reports.

### Changed

- expanded Studio command coverage and inverse contract tests so every public
  CLI capability has an app-owned typed or utility path and every Studio option
  remains accepted by ArgumentParser.
- expanded the VitePress workflow, plugin, model-management, serving, image,
  video, music, speech, SFX, text, and vision documentation to cover the new
  first-class paths and their CLI/runtime contracts.
- kept Studio creator-facing and local: Relay and its node app continue to own
  node identity, fleet placement and scheduling, worker lifecycle, model
  distribution, inventory, and hardware telemetry. World/Cosmos authoring
  remains in the separate Diorama product rather than being duplicated.
- expanded the packaged app's camera and microphone permission/entitlement
  coverage for first-class live vision, recording, transcription, and voice
  reference workflows.

## 0.27.1 - 2026-07-27

### Added

- added Sparkle 2.9.2 to the macOS Studio with a standard Check for Updates
  command and daily background discovery through the stable HTTPS appcast.
  Updates are Developer ID signed, notarized, verified with a pinned Ed25519
  key before extraction, and delivered through a signed feed.

### Fixed

- made SCAIL-2 fast-profile preflight return a structured blocked report when
  its managed four-step adapter is missing or fails byte-count/SHA-256
  verification. Generation still resolves the adapter strictly before loading.
- bundled and signed Sparkle's versioned framework and nested helpers
  inside-out while preserving the Downloader service entitlements and framework
  symlinks.

## 0.27.0 - 2026-07-27

This release brings the macOS Studio to full public CLI capability coverage,
adds the production ACE-Step music workflow, promotes Laguna S 2.1 to managed
chat and serving with validated DFlash acceleration, and refreshes the public
VitePress documentation.

### Added

- added a machine-readable, shared 89-command CLI capability contract emitted by
  `mere.run catalog --json`, plus executable CLI-help, App-to-contract, and
  inverse contract-to-App drift tests.
- expanded the macOS Studio into typed Text, Image, Video, Music, Speech, SFX,
  Vision/VFX, adapter, durable-run, world, setup, model, plugin, Open WebUI, and
  API workspaces. Graph Studio and Node remain explicit deep-linked product
  boundaries rather than duplicated orchestration surfaces.
- added first-class LTX text/image/audio-to-video and start/end-keyframe
  conditioning, Wan TI2V, SCAIL-2, Cosmos3, mask preparation, latent export,
  and resident video sessions to the macOS app.
- added production ACE-Step cover/repaint/flow-edit/retake, candidate ranking,
  stems, recipes, LRC/DAW export, adapter training, and resident serving controls
  to the macOS app, alongside full image LoRA/edit/3D, text JSON/LoRA/KV, and
  Vision/VFX workflows.
- added typed ACE-Step task and checkpoint capability routing for text-to-music,
  repaint, cover, cover-nofsq, extract, lego, and complete. Invalid tasks now
  fail during CLI parsing, and Base-only operations fail before checkpoint
  weights load on Turbo/SFT models.
- added native ACE-Step repaint ranges with upstream-compatible chunk masks,
  source-silence conditioning, per-step clean-source injection, latent boundary
  blending, and post-VAE original-waveform splicing. New controls expose edit
  start/end, chunk-mask mode, preservation mode, and balanced-mode strength.
- added immutable XL-SFT and XL-Base managed checkpoints, native continuous
  non-Turbo inference with CFG/APG/ADG and Euler/Heun integration, adaptive
  draft/song/final/edit presets, 4B LM planning, and automatic duration.
- added warm resident music sessions and API batching, deterministic best-of-N
  seed fanout, technical candidate scoring/ranking, retake interpolation, and
  upstream-style semantic flow editing.
- added full resident API control parity for inference, guidance, cover,
  repaint, flow edit, reference audio, LM planning, VAE tiling, and task
  instructions. Batch items keep independent best-of-N counts and responses
  include their final effective conditioning metadata.
- added native stacked PEFT LoRA and LyCORIS LoKr inference plus ACE-Step
  flow-matching training through `music train-adapter`. Both formats were
  train-save-reload tested against the installed XL-Turbo checkpoint.
- added synchronized LRC, exact checkpoint/adapter recipe provenance, PCM16,
  PCM24 and Float32 export, normalization/fade/dither controls, extracted
  stems, and portable DAW/REAPER bundles.
- added generation recipe schema 2 with the final effective BPM, duration,
  key/scale, vocal language, and time signature after LM planning and explicit
  overrides.
- added exact min-p sampling across native MLX and llama.cpp text generation,
  OpenAI-compatible `min_p` requests, per-model runtime defaults, and the chat,
  tool-call, and code benchmark lanes. Filtered tokens retain exactly zero
  probability, greedy output is unchanged, and Laguna DFlash applies the same
  normalized distribution to draft sampling and target rejection correction.
  Laguna uses the measured `0.02` default while retaining explicit
  `--min-p 0` control; other managed-model defaults are unchanged.
- added managed, opt-in Laguna S 2.1 chat and OpenAI-compatible serving with
  immutable official target and DFlash companion revisions, explicit
  OpenMDW-1.1 acceptance, checkpoint validation, and hardware support
  guidance. Pulling `text-chat-laguna-s-2-1` installs both checkpoints;
  `text chat` and `api serve --engine text-chat-laguna` enable the validated
  sampling recipe and automatic DFlash routing without making the 74 GB pair
  a setup or machine default.
- added complete macOS Studio access to Laguna: model-aware API and Open WebUI
  routing, first-class min-p controls in Chat and Code, persistent min-p runtime
  policy in both model editors, and a typed target/DFlash benchmark workspace
  with checkpoint, decode, fixture, concurrency, adaptive-routing, response-log,
  and JSON controls.
- added lossless Laguna target verification, ragged continuous batching, and
  machine-readable acceptance, recovery, batching, prefill, and decode
  metrics. A length-aware output-budget router bypasses DFlash work for short
  requests, uses the measured 12-token proposal default, and falls back
  losslessly after one clearly low-acceptance round or two sub-threshold
  rounds. Greedy verification keeps the anchor, proposals, and target
  verification GPU-lazy until one round-level readback. A resident-process
  target/DFlash crossover benchmark records exact decode lengths, mixed
  concurrency, output fingerprints, and MLX memory.

### Changed

- sorted Laguna routed-expert prefill and multi-token verification work by
  expert before the NVFP4 gather matmuls, with the reference routing order
  retained for small decode forwards and as a
  `MERERUN_LAGUNA_SORTED_MOE=0` rollback path.
- fused small-route gate/up gather-GEMV plus SwiGLU for the measured Laguna
  NVFP4 and LFM2 affine-8 decode layouts. Both paths preserve the native down
  projection, fall back on incompatible shapes, and retain explicit
  environment rollback controls. The pinned MLX fork exposes the required
  quantized Metal helper headers only to custom kernels that request them.
- fused Laguna's measured M4 Max sorted NVFP4 prefill gate/up projections and
  SwiGLU with expert-aligned scheduling and linear permutation inversion.
  Native down projection, weighting, and reduction remain unchanged, with
  explicit rollback controls and portable fallback on every other layout.
- established a task-local MLX default stream for LFM2 chat and preparation,
  matching the other native MLX engines and fixing first use from a new Swift
  task.
- refreshed the pinned MLX and mlx-swift forks to MLX 0.32.1 while preserving
  native affine 1-bit Metal/CUDA execution, the generation-17 NAX correctness
  gate, and the Linux/CUDA bridge. The runtime and local gate now reject Metal
  libraries whose exact mlx-swift revision or generated-kernel hash does not
  match the compiled binary.
- changed `--non-cover` into a compatibility alias for the explicit
  `cover-nofsq` task instead of a boolean that could silently erase task
  semantics.
- source-conditioned ACE-Step tasks now inherit source duration and require
  source audio explicitly. Non-Turbo checkpoints now use their native
  guidance and continuous schedule rather than the distilled Turbo sampler.
- ACE-Step LM vocal-language metadata now accepts only the exact supported
  upstream language codes, normalizes valid codes to lowercase, and discards
  malformed multi-value or non-vocal planner output.
- resident music API requests now reject a model other than the loaded model,
  reject unsupported response formats with actionable JSON errors, and honor
  per-item candidate counts in heterogeneous batches.

### Fixed

- fixed macOS packaging so the inner app is notarized, stapled, and
  Gatekeeper-validated before the DMG is assembled; a notarized outer image can
  no longer conceal an unstapled app.
- fixed the open Advanced Studio surface so switching creation modes also
  switches to the matching typed command and carries over its quick-composer
  values.
- fixed ACE-Step prompt, unconditional, and automatic repaint chunk masks to
  match upstream boolean-mask semantics. The runtime previously supplied
  `2.0` where upstream converts that value to boolean `true` (`1.0`), causing
  stationary broadband-noise output from otherwise parity-matched components.
- fixed text-to-music LM conditioning so generated 5 Hz audio codes reach the
  diffusion model, and made `--seed` cover both LM planning/code sampling and
  diffusion for repeatable recipes.
- strengthened best-of-N ranking and the listening regression with spectral
  structure and tail-continuity metrics, so non-silent broadband noise and
  prematurely dead endings no longer pass as high-quality candidates.

## 0.26.0 - 2026-07-22

This release contains every change merged since `v0.25.0`: the LTX 2.3
draft/final product contract, the refreshed DwarfStar runtime, and a broad
documentation and CLI-help accuracy pass.

### Changed

- separated LTX video checkpoint quality from output modality. `video generate`
  now uses `--quality draft|final` and `--output-mode
  video-only|audio-video`, defaults to fast draft video-only, and supports the
  full dev + distilled-LoRA final pipeline without forcing an audio stream.
  The former `--variant distilled|unified-av` selector remains compatible
  (#206).
- clarified the LTX product workflow around fast draft iteration and
  higher-quality final renders: audio suppression controls the deliverable,
  while checkpoint selection controls the meaningful speed/quality tradeoff
  (#206).
- refreshed the bundled DwarfStar runtime for the DeepSeek V4 Flash premier
  agent tier from upstream `be434773` to `efdadd41`, including the current
  Metal kernels, server request hardening, KV-cache fixes, and session/runtime
  improvements while preserving the existing Mere launch and OpenAI-compatible
  API contract (#204).
- corrected drift and expanded coverage across the public CLI and docs,
  including configuration, model runtime and benchmarking, API vision routes,
  image, speech, text, video, vision, agent workflows, and site navigation
  (#203).

### Fixed

- fixed canonical `video-ltx23-full-mlx` inspection and validation when an
  existing compatible install still carries the legacy
  `video-ltx23-a2vid-mlx` manifest ID. Component reporting now uses the
  requested full layout while retaining an explicit compatibility warning
  (#206).
- fixed bare and explicit `video generate --variant distilled` requests on
  LTX 2.3 split MLX roots. The CLI now routes those roots through the native V2
  standalone-distilled transformer, preserves the video-only MP4 contract,
  skips unnecessary audio VAE/vocoder loading and decoding, and exposes phase
  timings without changing legacy merged distilled or unified-AV output
  (#206).

## 0.25.0 - 2026-07-21

This release contains every change merged since `v0.24.0` across native
Cosmos3 Edge inference, persistent action-conditioned worlds, and the SCAIL-2
recast workflow.

### Added

- added a native Swift/MLX implementation of NVIDIA Cosmos3 Edge through
  `video cosmos3`, covering text-to-image, image editing, text/image/video
  generation, policy prediction, action-conditioned forward dynamics, inverse
  dynamics, and image/video reasoning (#200).
- added the managed `video-cosmos3-edge-mlx` model with an immutable official
  snapshot pin, complete generation and understanding checkpoint validation,
  published per-mode sampling defaults, and explicit OpenMDW-1.1 model-license
  acceptance (#200).
- added `world serve --backend cosmos3` with normalized `camera_pose`
  trajectories, semantic camera controls, direct `model_space_actions`, warm
  model reuse, autoregressive seed progression, pixel-stable public-frame
  handoffs, and exact inverse traversal of cached navigation edges without
  stochastic regeneration (#200).
- added native Cosmos3 packed SigLIP2 vision, multimodal reasoner, tokenizer,
  action packing for all 15 published domains, shifted-flow UniPC schedules,
  Wan VAE loading, and generation/understanding mixture-of-transformers
  execution (#200).
- added independently generated Cosmos3 checkpoint inventories and numerical
  parity fixtures against pinned NVIDIA sources. Runtime inference neither
  vendors nor invokes upstream Python, PyTorch, or Diffusers code (#200).
- added a checksum-pinned, remote-only Apache-2.0 LightX2V four-step adapter
  to the managed adapter catalog for native SCAIL-2 rendering (#199).
- added review-first SCAIL-2 recast preparation with dense painted-mask
  corrections, replacement reference matting, stable multi-subject tracking,
  palette validation, immutable manifests, previews, and quality reports
  (#199).

### Changed

- made the proven four-step SCAIL-2 adapter recipe the default for
  `video animate`, including its 832x480 geometry, no-CFG Euler schedule, and
  shift 5. The configurable 40-step UniPC recipe remains available through
  explicit `--profile quality` selection (#199).
- changed native SCAIL-2 attention execution to split the query axis across
  independently evaluated Metal command buffers while preserving global
  key/value attention, keeping full 81-frame windows below the macOS GPU
  watchdog without introducing local or sliding-window attention (#199).
- extended persistent-world session responses and transition receipts with an
  explicit backend identity, action space/domain, and normalized model-space
  trajectory. `raw_actions` remains a compatibility alias (#200).
- documented the Cosmos3 model-license boundary, immutable NVIDIA source pins,
  native implementation provenance, public CLI modes, and world-session
  behavior in the model registry, guides, and third-party notices (#200).

### Fixed

- fixed SCAIL-2 reference and driving mask-role semantics for animation versus
  replacement, including legal palette handling, dense correction encoding,
  aspect-fit replacement references, and canonical white review backgrounds
  (#199).
- fixed SCAIL-2 tracking across collapsed masks and visibility gaps so objects
  can re-anchor from their seeds without swapping stable multi-subject IDs
  (#199).
- fixed `video animate --preflight` on clean installations so the default
  managed adapter is reported as a missing pinned input instead of throwing
  before the preflight report; real renders still require checksum-verified
  adapter installation (#200).

## 0.24.0 - 2026-07-19

This release contains the complete `v0.23.0..v0.24.0` delta: 22 commits
changing 115 files across model storage, live speech, documentation, Graph
Studio contracts, native SCAIL-2 video, and workflow-graph authoring.

### Added

- added byte-exact `model storage --json` accounting and dry-run-first
  `model gc`, including shared-payload ownership, legacy-link preservation,
  stale partial/snapshot/blob collection, macOS Studio cleanup previews,
  cross-process locking, and an ownership recheck before deletion (#190).
- added revision-addressed Hub snapshots, ETag-addressed hard-linked blobs,
  paginated tree discovery, and zero-copy adoption of matching legacy
  payloads (#190).
- added resident Parakeet live transcription for raw PCM and streamed files,
  preserving the V1 ready, partial, commit, stats, final, cancellation,
  silence, and bounded-backpressure contract already used by Qwen live ASR
  (#191).
- added live ASR backend capabilities to `mere.run status` so local and remote
  schedulers can deliberately select Parakeet or Qwen (#191).
- added generated command inventories, CLI/documentation ownership and
  navigation contract tests, and
  `scripts/update-docs-command-reference.sh` so command or runtime-page drift
  fails the normal repository gate (#192).
- added dedicated public guides for plugins, persistent worlds, Graph Studio,
  and the Graph v2 runtime generation while preserving Workflow Graph
  `schema_version: 1` as the stable portable ABI (#192, #193).
- added native Swift/MLX SCAIL-2 14B subject animation and replacement through
  `video animate`, including exact seven-color mask packing, mixed-stream RoPE,
  OpenCLIP and UMT5 conditioning, Wan 2.1 VAE support, 81/5 segmented clean
  history, additional references, Flow-UniPC CFG sampling, typed preflight,
  MP4 output, and a managed model ID for the separately packaged MIT-licensed
  MLX checkpoint (#194).
- added `video prepare-masks`, a typed native SAM 3.1 preparation workflow for
  one to six SCAIL subjects with reference-seeded preview masks, bounded
  corrections, independent video tracking, seven-color palette encoding,
  quality reports, immutable manifests, and Apple ProRes or FFmpeg output
  (#196).
- added first-class creative material nodes for reusable text, numeric, boolean,
  JSON, seed, choice, join, template, enhancement, and image-description
  workflows, with native deterministic intrinsics, strict template validation,
  explicit installed-model execution, recursive typed catalog schemas, frozen
  seeds, presentation metadata, synchronized public schemas and fixtures, and
  source-correlated run provenance (#197).

### Changed

- changed `model remove` to report referenced and reclaimable bytes and delete
  unshared backing payloads by default while preserving shared, external, and
  legacy-linked data. `--keep-cache` retains link-only removal, and model
  listings now label per-model values as non-additive `Referenced` bytes
  (#190).
- changed live `--backend auto` policy to match batch ASR: transcription uses
  fast Parakeet while translation uses quality-first Qwen (#191).
- refreshed the documentation homepage, getting-started flow, CLI reference,
  navigation, model sources, speech, workflow, storage, testing, and
  development guidance around all 22 top-level commands and the current
  runtime surface (#192).
- documented Graph Studio's local-first security boundary: native authoring,
  persistence, catalogs, validation, preflight, and execution use Tauri
  IPC/Rust without Python, a loopback HTTP API, authentication, or network
  access. Direct SSH, Relay, hosted Authorization Code + PKCE, and Mere Node
  device authorization remain explicit opt-in boundaries (#195).
- pinned the public SCAIL-2 MLX snapshot and kept checkpoint conversion,
  upstream PyTorch exporters, and publication tooling outside the runtime.
  Multi-reference ordering, tail pad/trim behavior, and driving-audio
  preservation now flow through native animation and replacement (#194,
  #196).
- extended graph catalogs and public schemas with recursive value shapes and
  presentation metadata, and propagated source graph/input fingerprints
  through materialized jobs and `run.json`. Graphs using creative material
  nodes require a 0.24.0-or-newer worker (#197).
- removed the untrusted Homebrew tap step from the hosted macOS CI lane and
  kept platform-specific ProRes round-trip coverage off headless runners
  (#196).

### Fixed

- fixed model-size reporting that double-counted shared symlink targets and
  allowed repo-flat caches, obsolete revisions, and incomplete downloads to
  accumulate indefinitely (#190).
- fixed SCAIL mask correction propagation so edits stay bounded, aligned
  native mask semantics with the pinned upstream fixtures, accepted upstream
  H.264 mask backgrounds, and matched upstream codec boundary margins (#196).

## 0.23.0 - 2026-07-18

### Added

- added checksum-gated canonical Gemma 4 chat templates, typed assistant
  tool-call and tool-response history, and a deterministic
  `model benchmark tool-continuations` lane for validating grounded final
  answers after one- and two-tool chains against real checkpoints.
- added a pinned `mlx-vlm` conversion script for reproducing the current Gemma
  4 12B affine 4-bit MLX artifact directly from Google's verified dense
  checkpoint without modifying an installed managed model, published the
  audited conversion as `Sawfwair/gemma-4-12B-it-MLX-4bit@v1.0.0`, and pinned
  managed compact Gemma pulls to the exact hosted release commit.
- added real-time Qwen speech transcription through `speech listen` and raw
  PCM stdin streaming, with bounded ingestion, adaptive speech detection,
  incremental mel/FFT processing, partial and silence-commit events, session
  rollover, exactly-once terminal results, and discoverable streaming protocol
  support in `mere.run status`.
- expanded portable workflow graphs with deterministic stable-topological
  parallel scheduling, graph-level concurrency limits, node-level bounded
  retries and timeouts, and fail-fast cancellation of in-flight child work.
- added verified cross-run node caching with dependency, argument, model,
  provider, and input-artifact fingerprints plus explicit `auto`, `never`, and
  `refresh` cache policies.
- added typed resource requirements and named secret references, with
  preflight enforcement for accelerator and system memory, disk, CPU, network,
  configured secrets, provider versions, and worker compatibility. Newly
  materialized graph jobs require a `mere.run` 0.23.0 worker or newer.
- added reusable plugin-side workflow composition, provider output-port
  references, structured catalog metadata, and shared compatibility fixtures
  for consistent graph decoding across Swift, Python, TypeScript, and Rust.
- added repeatable `run fetch --artifact <name>` selection with verified reuse
  of already-downloaded artifacts, allowing interrupted and partial remote
  result retrieval without transferring the complete artifact set again.
- added `graph run-job` and `graph submit-job` so one verified immutable export
  can execute unchanged through local, SSH, and Relay while mutable run state
  remains in a separate directory.

### Fixed

- fixed graph cancellation, retry, timeout, and resume cleanup so every owned
  child process is terminated and reusable outputs remain digest-verified.
- suppressed harmless SSH archive metadata warnings without hiding transfer or
  integrity failures.
- fixed canonical `image-zimage-nano` resolution for valid managed installs
  created from the current mflux mirror's legacy `@main` manifest.

## 0.22.0 - 2026-07-17

### Added

- added native LTX phase timing reports and a typed JSONL `video session`
  worker for resident standalone-distilled and full dev + distilled-LoRA
  inference. Full sessions keep the transformer resident and install all 1,660
  official mixed-rank LoRA targets as reversible runtime layers, preserving
  the base weights for dev stage one and enabling the adapter only for
  distilled stage two. On an M4 Max, matched resident requests reduced
  end-to-end time from 103.447 to 29.947 seconds (3.45x) for the standalone
  lane and from 347.040 to 235.862 seconds (1.47x) for the full lane.

- added fully native LTX 2.3 source-audio-to-video generation through
  `video generate --audio`, including exact 16 kHz stereo mel conditioning,
  the audio VAE encoder, frozen-audio multimodal guidance, full/dev stage one,
  streaming distilled-LoRA stage two, the shared managed
  `video-ltx23-full-mlx` bundle for both unified AV and A2Vid, compatibility
  resolution for the former `video-ltx23-a2vid-mlx` ID, structured preflight
  reporting, and original source-segment MP4 muxing without a soundtrack-only
  fallback.

- added managed `text-chat-bonsai-27b-1bit` and
  `text-chat-bonsai-27b-2bit` support for Prism ML's packed binary and ternary
  dense Qwen3.6 27B vision/reasoning checkpoints, including exact revision
  pinning, native packed linear and embedding execution, the published 262K
  context and sampling defaults, OpenAI-compatible image inputs, and opt-in
  affine 4-bit or 8-bit KV caches for long-context memory control.

- added managed Buffalo-L face analysis through `vision face detect`, `embed`,
  `compare`, and warm-session `batch`, including RetinaFace-style boxes and five-point landmarks,
  normalized 512-dimensional ArcFace embeddings, CPU/CoreML execution-provider
  controls, machine-readable JSON, and exact checkpoint validation.

- added constrained `json_object` generation for native MLX Gemma and
  Qwen-family chat models through OpenAI `response_format` and the new
  `text chat --response-format json_object` option. The runtime now enforces a
  complete root-object JSON grammar token by token, disables thinking and
  speculative/batched/pipelined Qwen decoding for JSON requests, advertises
  structured JSON without strict JSON Schema support, and rejects the still
  unsupported llama.cpp/GGUF Q36 lane explicitly.
- added Workflow Graph V1 with typed LoRA, image, and video nodes; immutable
  content-addressed job bundles; local and resumable graph execution; the
  public graph worker protocol; SSH and direct relay executors; and unified
  inspect, watch, fetch, cancel, retry, and remote-list commands.
- added provider-qualified graph nodes backed by a versioned plugin process
  contract, exact provider catalog pinning, typed scalar/JSON/directory/
  collection ports, structured NDJSON node events, and provider capability
  reporting through local, SSH, and relay workers.
- added dependency-scoped node fingerprints, exact model/adapter/provider
  provenance, verified resume admission, typed artifact manifests, and stable
  canonical hashes for portable execution and reproducible retries.
- added durable mere.world relay sign-in with automatic refresh-token rotation,
  relay fleet inspection and node inventory refresh, typed no-eligible-node
  diagnostics, resumable artifact transfers, and per-job execution telemetry.
- added a checksum-pinned public adapter catalog and `mere.run adapter list`
  / `mere.run adapter pull` commands, beginning with the promoted Mere Platform
  Assistant v22 release for Gemma 4 12B 4-bit. Text chat and API serving now
  accept the catalog id `mere-platform-assistant` anywhere `--lora` accepts a
  local adapter path.

### Changed

- upgraded the shared Prism packed binary and ternary kernel stack used by the
  new Bonsai 27B text models and the existing `image-bonsai-binary` and
  `image-bonsai-ternary` generators. Matched M4 Max release runs measured 67.80
  and 45.04 decode tokens/s for the 1-bit and 2-bit text models, plus 9.91 and
  14.04 seconds for four-step 512x512 binary and ternary image generation.
- enabled Krea 2 LoRA-training and adapter-backed image graph stages on
  constrained CUDA workers through 4-bit bases, shape-scoped quantized
  fallbacks, sequential model phases, and Linux-package CUDA discovery.
- added CUDA 12.8-aware Debian metadata for Linux release artifacts, including
  Lambda Stack-compatible dependency alternatives alongside the CUDA 13.0
  dependency family. Packaging still derives the toolkit major from the linked
  `libcudart` SONAME and fails closed for unknown majors unless a maintainer
  supplies the complete dependency override.
- standardized restricted-model usage terms and explicit acceptance across
  model listing, information, pull, preflight, setup, Studio, and related
  command guidance while retaining upstream license and notice artifacts.
- refreshed the README capability map and CLI documentation around the current
  modality-first command surface, Relay/Nodes, portable workflows, adapters,
  and the official plugin ecosystem.

### Fixed

- added native affine 1-bit CUDA quantize, dequantize, and matrix-vector
  execution to the pinned MLX fork. Linux CUDA now keeps Prism Bonsai binary
  text and image projections packed during inference instead of materializing
  dense weights; 2-bit and wider models retain their existing dispatch paths.
  On an NVIDIA GB10, matched 32-token Bonsai 27B runs measured 16.74 decode
  tokens/s for native 1-bit versus 4.15 for the dense fallback and 14.71 for
  native 2-bit. One-step 512x512 binary and ternary image runs also completed
  through their respective native 1-bit and 2-bit transformer paths.
- updated the official plugin catalog default and bundled plugin guide to use
  the live `sawfwair/mere-run-plugins` repository after the old catalog path
  was retired.
- kept image-generation graph preflight declarative so it no longer initializes
  Metal merely to describe the selected backend, and included bounded stderr,
  stdout, and termination details when a child preflight fails.
- packaged the ONNX/CoreML face-analysis runtime correctly in the signed macOS
  app bundle.
- preserved Swift 6.0 Linux CLI compatibility for the pinned MLX Swift fork
  through a CPU-compatible base manifest while retaining the Swift 6.3
  CUDA/CGen package graph, and corrected literal quant-mode reporting in the
  DGX Spark e2e sweep.
- kept CUDA quant-kernel capability decisions isolated by bit width, group
  size, and quantization mode, and routed fused projections plus tied 1-bit/
  2-bit output embeddings through the dense fallback when the native CUDA
  matrix kernel cannot execute them.

## 0.21.0 - 2026-07-14

### Added

- added native Swift/MLX Microsoft TRELLIS.2-4B image-to-3D reconstruction at
  512 resolution, including direct official safetensors loading, pinned DINOv3
  ViT-L/16 conditioning, three 12-step flow stages, sparse ConvNeXt O-Voxel
  shape/PBR decoding, flexible-dual-grid mesh extraction, managed model pull,
  separate image/vision CLI commands, and hashed OBJ/PLY/GLB plus six-channel
  `.pbrvox` material artifacts. The DINOv3 dependency remains separately
  license-gated and requires user acceptance/authentication.
- added native Apple-platform VFX primitives through `vision pose` and
  `vision flow`, with body/hand/face landmark JSON and dense Middlebury `.flo`
  optical-flow output from the system Vision framework.
- added native Swift/MLX geometry and depth workflows: MoGe-2 single-image
  metric geometry, Video Depth Anything temporal depth, and Depth Anything 3
  multi-view camera solving and point geometry. The commands emit bounded,
  provenance-rich depth, confidence, camera, EXR, PLY, GLB, preview, and
  3DGS-initialization artifacts as appropriate to each model.
- added native TripoSR single-image and InstantMesh four/six-view object
  reconstruction, with managed converted checkpoints, deterministic input
  admission, native isosurface extraction, and hashed OBJ/PLY/GLB run
  artifacts. InstantMesh intentionally does not bundle or invoke Zero123++.
- added native Swift/MLX Wan 2.2 TI2V-5B image-to-video generation and reusable
  warm world sessions with first-frame conditioning, camera controls, terminal
  latent chaining, managed checkpoint verification, and MP4/PNG artifacts.
- added the native DreamX causal world runtime and `world` CLI, including the
  converted AR transformer, persistent block-causal and cross-attention caches,
  PRoPE camera conditioning, multi-transition session state, and a loopback
  `world serve` API with asynchronous jobs, progress, cancellation, reset, and
  unload. The converted causal checkpoint remains a local-only managed artifact
  because the public FP32 file is not the streamed BF16 MLX layout.
- added native MMAudio large 44.1 kHz text-to-audio and video-to-audio SFX
  generation, including managed pinned assets, DFN5B CLIP and Synchformer video
  conditioning, the joint/fused MMDiT flow model, magnitude-preserving VAE,
  BigVGAN-v2 decoding, negative prompts, preflight support, and explicit
  CC-BY-NC-4.0 checkpoint, Apple research-only CLIP, and MIT BigVGAN licensing
  metadata.
- added explicit affine 8-bit resident KV caches for Gemma4, Qwen-family, and
  LFM2 plus opt-in compressed-MLA and fused sparse-MoE execution for the native
  Psi/GLM runtime, with numerical, structural-memory, and policy tests. Affine
  8-bit is a memory control relative to full-precision KV; Gemma Turbo's default
  4-bit TurboQuant cache remains smaller.
- added `image generate --progress-json`, a machine-readable progress stream
  for wrappers: one JSON object per event on stderr
  (`{"event":"progress","stage":"denoising","step":2,"total_steps":4}`),
  replacing the human-readable progress text and taking precedence over
  `--quiet` for progress output.
- added native MuScriptor full-mix audio-to-MIDI transcription through
  `music transcribe`, with managed small/medium/large checkpoints, exact HTK
  mel preprocessing, MLX transformer inference, instrument conditioning,
  JSON/JSONL note events, and multi-track MIDI output.
- added native MuScriptor musical-context enrichment for MIDI output: tempo,
  beat phase, meter, and key detection with per-field confidence, reviewable
  beat-position JSON, and standard MIDI tempo/time/key meta events without
  quantizing source-relative note timing.
- added catalog-backed managed-model download size estimates so model listings,
  capabilities, Studio readiness, and pull preflights can show the expected
  local storage commitment before a checkpoint download begins.

### Changed

- upgraded TRELLIS.2 appearance and geometry export with glTF-spec linearized
  vertex colors, material-factor normalization, deterministic PBR texture-atlas
  baking into self-contained textured GLB, a watertight narrow-band remesh,
  closed-rim capping, morphological cavity sealing, and closest-surface color
  projection. `--texture-seed` can now re-roll appearance independently while
  preserving structure and shape from the base seed.
- polished the macOS Studio app: narrow windows keep an icon rail for mode
  navigation instead of hiding the sidebar; image generation shows a
  determinate per-step progress bar (parsing both the human `Generating (N/M)`
  stderr text and the `--progress-json` event stream); example prompt chips
  wrap instead of truncating; a blank chat no longer shows a duplicate
  "New chat" header; the composer's model pill shows a human-readable model
  name (exact id in the tooltip) and the paste button appears only when the
  clipboard holds an image; results cap to a readable width on wide windows;
  Listen uses a mic-badged waveform icon; sidebar footer controls use themed
  styling.
- overhauled the public docs theme with bundled typography, refreshed code
  presentation, and upgraded local search styling.
- reduced autoregressive prefill and decode work across shared Qwen, ACE-Step,
  ASR, TTS, OCR, VLM, MuScriptor, and Falcon paths with final-position output
  projection, pipelined GPU sampling, compact grouped-query caches, and true
  cached Falcon grounding.
- kept Qwen-VL caption prefill, deep-stack features, sampling, and decode inputs
  on device. A matched three-run M4 Max release gate measured a 2.54-second
  median versus 2.67 seconds (-4.9%) and 2.03 versus 2.13 GB RSS (-4.7%), with
  byte-identical captions.
- kept the shared autoregressive GPU queue saturated after first-token
  confirmation while preserving the final-token boundary fast path. A matched
  96-token LFM2 release A/B on an M4 Max 128 GB measured 63.25 versus 62.47
  decode tokens/s (+1.2%) and 9.61 versus 9.79 GB peak physical footprint
  (-1.8%).
- generalized batched MuScriptor decode across independent audio chunks and
  made `music transcribe --chunk-batch-size` a safety upper bound, adaptively
  clamped by current MLX unified-memory headroom, compute-type-scaled lane
  estimates, and a model-complexity live beam budget. Oversized requested beams
  retain their search width by microbatching forwards to the lane budget. Beam
  search preserves independent typed cache lanes throughout. In a
  matched warm M4 Max 128 GB large-model beam-4 run, the selected two-chunk
  group had a 3.69-second median and 50.38 GB peak footprint versus the
  pre-batching baseline's 22.38 seconds and 11.15 GB; all output hashes matched.
  One chunk took 4.70 seconds at 28.17 GB, while four chunks was dominated at
  6.16 seconds and 87.48 GB and is now capped for that model/beam combination.
- enabled compatible-row continuous decode batching automatically for Gemma4,
  Qwen-family, and LFM2 serving when `--max-active-requests` is above `1`.
  Engine-specific environment variables remain force-on/force-off overrides;
  LFM2 uses ragged row-offset-aware KV lanes, batched short-conv state, one
  sampling readback per step, immediate finished-row compaction, and exact
  serial fallback for incompatible caches. Two simultaneous matched 96-token
  LFM2 requests completed in 0.932 seconds versus a 1.148-second baseline tail,
  about 23% higher aggregate throughput.
- changed Linux CUDA quantized matmul selection to probe native
  `quantized_mm` and `GatherQMM` independently, retaining an automatic dense
  fallback for runtimes that do not provide either kernel; Linux native
  preparation also skips unused llama tools and server targets.
- generalized batched classifier-free guidance across Qwen Image Edit, Z-Image,
  FLUX.2 Klein, and HiDream O1, with a shared policy, model-specific overrides,
  an automatic estimated-MLX-allocation-headroom gate, and exact serial shape
  fallbacks. Forced batching bypasses the estimate and can increase peak memory
  or exhaust unified memory.
- moved LTX tiled VAE overlap blending from per-tile CPU readbacks and Swift
  pixel loops to device-side MLX accumulation and normalization.
- streamed LTX video frames directly to FFmpeg stdin or AVAssetWriter one frame
  at a time while overlapping the next device transfer, avoiding a monolithic
  host frame buffer or raw-video spool. LTX audio is transferred and written in
  aligned chunks, and float WAV output supports incremental writes without a
  second whole-file `Data` copy.
- enabled supported fused scaled-dot-product attention shapes by default for
  SAM 3.1, LightOn OCR, and selected vision encoders, with
  `MERERUN_FUSED_SDPA=0` as the portable fallback. Installed release checks
  measured about 3% lower SAM text-prompt latency with 13.2% lower peak
  footprint, and 1.46x LightOn OCR throughput with 55% lower peak footprint;
  RSS stayed effectively flat in both cases.
- removed Ideogram 4's redundant single-segment block mask by default. Custom
  Ideogram QKV-normalization, AdaLN, and residual kernels remain opt-in through
  `MERERUN_IDEOGRAM4_FUSED_KERNELS=1`: microbenchmarks improved, but the
  installed-checkpoint warm path was 1.65x slower with no MLX peak-memory win.
- kept the most recently used embedding, image, image-edit, TTS, and ASR
  sidecar runtimes resident in `api serve`; matching requests now reuse loaded
  model state, concurrent use of mutable generators is serialized, and
  switching models unloads the previous resident before loading its
  replacement. Cold sidecar work is exclusive across lanes, and catalog/path
  size estimates plus conservative working-set floors trigger proactive idle-resident eviction or reject a load
  projected to cross the configured hard memory guard. Sidecars use a
  bounded five-minute idle TTL with autonomous expiry, poll live managed-model
  `pinned` and `ttlSeconds` changes while idle, join memory-pressure eviction,
  and report residency/readiness/counters under `/runtime/status` and
  `mere.run status` without evicting active or queued work. The special
  `qwen-image-edit` repository lane is resident but currently uses default
  lifecycle settings because it is not configurable through `model runtime`.
- measured the resident API path with three matched release requests per lane:
  median latency fell from 2.10 to 1.23 seconds for TTS (-42%), 0.154 to 0.057
  seconds for ASR (-63%), 1.193 to 0.710 seconds for a one-step image request
  (-40%), and 0.021 to 0.013 seconds for embeddings (-38%). TTS and ASR outputs
  were byte-identical; retained models remain subject to the documented idle
  TTL and memory-pressure eviction policy.
- extended fair FIFO request admission to every local inference route, so the
  default `--max-active-requests 1` serializes chat and media activation peaks;
  explicit runtime model load/unload maintenance shares the same queue, and
  higher concurrency remains an explicit throughput and unified-memory choice.
- bounded API embedding requests to 256 texts and 2 MiB of UTF-8 content, then
  length-packed 8,192-token-capped rows into sequential batches with at most
  8,192 padded tokens while preserving response order.
- bounded OpenAI-compatible image generation and edit requests to dimensions
  divisible by 16 from 16 through 4,096 pixels and 4,194,304 total pixels, and
  return malformed image/TTS JSON as `400 invalid_request_error` responses
  rather than server errors.
- bounded explicit image inference to 100 steps, ASR decode to 4,096 tokens,
  and combined TTS input/voice instructions to 32 KiB of UTF-8 text so one API
  request cannot grow compute or prompt tensors without limit.
- switched Darwin memory-guard decisions to `ri_phys_footprint`, which accounts
  for unified-memory allocations that RSS can miss; status retains RSS for
  compatibility and other platforms continue to use it as the fallback.
- pinned the Linux CUDA CMake bridge to the exact `mlx-swift` checkout selected
  by SwiftPM under the active Linux Swift toolchain;
  `MLX_SWIFT_CUDA_COMMIT` remains an explicit diagnostic override.
- gated default CUDA `.deb` metadata on a linked CUDA 13 `libcudart` SONAME;
  other or unknown toolkit majors now fail closed unless a maintainer supplies
  the complete `MERERUN_PACKAGE_LINUX_DEPS` override. Tar packaging is unchanged.
- changed the default code-benchmark model set to the models supported and
  recommended for the current machine instead of a fixed cross-machine trio.

### Fixed

- fixed explicit `--models-root` and `MERERUN_MODELS_DIR` overrides so model
  pull preflight reports and follow-up actions keep the requested path even
  before that model-store directory exists.
- fixed DreamX camera compilation so the final aligned view reaches the
  requested translation or rotation magnitude instead of scaling motion by the
  full pixel-frame count.
- fixed Linux Swift 6 build compatibility across package manifests and the new
  MMAudio and TRELLIS.2 runtime paths.
- fixed repeat Linux packaging in an existing output directory so the
  regenerated checksum manifest does not include and invalidate its previous copy.

## 0.20.0 - 2026-07-07

### Added

- added `run list --json` and `run inspect --json`, typed readback reports for
  durable run directories, structured report JSON files, and saved run-plan JSON
  files. `run list` discovers existing artifacts under a workspace root and
  emits per-entry inspect actions; `run inspect` summarizes run status,
  manifests, events, actions, and artifacts without starting a workflow.
- added legacy/plugin run manifest readback to `run inspect --json`, so older
  runpod-backed run folders surface provider, GPU, dataset, recipe, command,
  status, and sample/artifact files as warning-level entries instead of opaque
  decode blockers.
- added compact run metrics to `run inspect --json`, including loss CSV summary,
  latest loss, step range, sample image count, checkpoint count, and adapter
  count for durable run directories.
- added `created_at` and `updated_at` fields to `run list --json` entries when
  known from native manifests, events, legacy/plugin manifests, reports, or
  saved plans, so wrappers can sort workspace run browsers without extra
  inspection passes.
- added optional per-candidate LoRA preflight commands to
  `image dataset discover --json` via `--training-output-root`,
  `--training-model`, and `--training-recipe`, so wrappers can turn discovered
  dataset leaves into concrete `image train-lora --preflight --json` actions.
- added `image generate --preflight --json`, a typed structured preflight report
  for local image generation. It checks model availability, input/reference
  files, LoRA files, structured-prompt paths, output overwrite risk, and emits
  declarative follow-up actions plus `result.run_plan` before loading the model
  or writing an image.
- added `image run-plan` support for saved `image.generate` plans, allowing
  generation plans emitted by preflight JSON to be preflighted, materialized into
  durable run directories, or executed later without reconstructing CLI
  arguments.
- added `model pull --preflight --json`, a typed structured preflight report for
  managed model downloads. It checks catalog source availability, machine
  support, installed state, model-store and hub-cache paths, disk headroom, and
  emits declarative pull/open actions before downloading.
- added `api serve --preflight --json`, a typed structured preflight report for
  the local OpenAI-compatible server. It checks host/port settings, non-loopback
  auth requirements, selected engine/model availability, LoRA paths, runtime
  limits, KV cache settings, companion models, and emits redacted start/status
  actions before starting the server or loading a model.
- added `video generate --preflight --json`, a typed structured preflight
  report for native LTX video generation. It checks prompt/options, model root
  availability, image and end-keyframe paths, output overwrite risk, resolved
  dimensions, resolved frame count/duration, unified AV fps warnings, and emits
  declarative start/pull/open/reveal actions before loading MLX or writing MP4s.
- added `vision track --preflight --json`, a typed structured preflight report
  for SAM 3.1 video tracking. It checks video path availability, prompt/box/point
  parsing, init/end frame options, threshold/resolution limits, model
  availability, output/json/mask destinations, and emits declarative start,
  pull, reveal, and open actions before loading SAM or extracting frames.
- added `sfx video generate --preflight --json`, a typed structured preflight
  report for Woosh video-to-audio generation. It checks raw-video versus `.npy`
  feature input mode, output WAV state, VFlow/DVFlow model availability, raw
  video Synchformer requirements, duration/steps/CFG/renoise options, and emits
  declarative start, pull, reveal, and open actions before loading MLX or
  generating audio.

### Changed

- clarified that NVIDIA GB10/DGX Spark remains a supported Linux arm64 CUDA
  package target when artifacts are built and smoke-tested on matching hardware.

### Removed

- removed GitHub Actions workflows that produced release artifacts from the
  public repository; local macOS and Linux packaging scripts remain available
  for reproducible builds.

## 0.19.0 - 2026-07-04

### Added

- added native `text train-lora` for reviewed chat-style SFT data, with
  Gemma4 LoRA injection, loss-masked training batches, JSONL dataset loading,
  training metrics, generated artifacts, and a loopback viewer through
  `--visualize`.
- added `image train-lora --preflight --json`, a typed structured preflight
  report for expensive LoRA runs. It checks dataset/caption shape, edit-pair
  mismatches, placeholder text, duplicate captions, model-family support,
  output overwrite risk, and emits follow-up actions instead of starting
  training blindly.
- added CoreMIDI steering for `music realtime` on macOS, including
  `--list-midi-inputs`, `--midi-input`, channel/note-offset filters, a robust
  MIDI 1.0 stream parser, note on/off handling, repeatable `--midi-cc`
  mappings, and `--midi-monitor`/MIDI logging flags for diagnosing live
  controller input before loading Magenta RT2.
- added `model benchmark code --thinking` for reasoning-enabled code evals.
  The benchmark keeps reasoning split away from scored code and disables
  HumanEval stop sequences that can fire inside a thinking block.
- added Ornith support to the `model benchmark q36-mtp` lane so MTP sidecar
  experiments can be measured against `text-agent-ornith-9b` and
  `text-agent-ornith-35b-mlx` with the existing counters.
- added an env-gated Q35 reference-parity harness:
  `MERERUN_Q35_DEBUG_LAYER_DUMP`, `MERERUN_Q35_DEBUG_PROMPT_TOKENS`, and
  matching mlx_lm dump/compare scripts under `scripts/reference-parity/`.
  Self-parity gates cannot catch defaults that are wrong the same way on both
  sides, so this compares against an independent implementation.
- added `scripts/build_mlx_metallib.sh`, a stamped MLX Metal kernel-library
  builder/verifier that records the mlx core version, mlx-swift pin, toolchain,
  and kernel-source hash next to `default.metallib`.
- added Linux CUDA package hardening for release artifacts: CUDA package
  suffixes, CUDA runtime/JIT `.deb` dependencies, CUDA CCCL discovery in the
  installed launcher, and an arm64 release guard that requires
  `MERERUN_LINUX_ACCEL=cuda` for meaningful arm64 packages.
- added docs for the mlx-swift fork policy, the measured compiled-call
  overhead cliff, and why the public repo stays on stock upstream mlx-swift
  until the fast-path lock removal is proven thread-safe.
- added image-generation docs for applying Klein LoRAs to reference images with
  `--ref-image`, including starting strength/LoRA-scale guidance.

### Changed

- shipped a large decode-performance train across Gemma4 and Q35. Measured
  highlights include Q35 MoE decode improving from 53.1 to 87.3 tok/s,
  Q35 prefill from 558-703 to 803-847 tok/s, Gemma4 sampled chat from
  26.0 to 30-35 tok/s, Gemma4 short-prompt TTFT from 5.4s to 2.7s, and
  quantized-KV long-context decode copy traffic dropping from about
  1.4 GB/token to zero.
- changed Q35 prefill chunking from 512 to 1024 tokens by default, with
  `MERERUN_Q35_PREFILL_CHUNK_TOKENS` for controlled overrides. Warm Ornith
  35B MLX prefill now reaches roughly 1350 tok/s in serve mode while keeping
  greedy output byte-identical across exact causal chunk sizes.
- optimized native text LoRA training with gathered lm_head loss,
  logSumExp-minus-gather cross entropy, async loss-readback cadence, a
  text-appropriate MLX buffer-cache cap, and mid-run partial checkpoints.
  Long 12B-4bit QLoRA steps improved from about 33.0 to 24.7 s/step while
  steady memory dropped from about 111 GB to 38 GB.
- optimized Q35 decode with fused q/k/v quantized projections, batched
  GPU-side sampling, stacked gate/up expert matmuls, and Q35 prefix/prefill
  plumbing that keeps sampling and prefill work on the GPU longer.
- optimized Gemma4 decode with pipelined sampled-token readback, top-p
  prefiltering, fused quantized projection loads, in-place quantized KV cache
  growth, single-query sliding-window cache reuse, final-position-only prefill
  logits, and MTP round batching for greedy verification.
- optimized LFM2 repeated-prompt serving with opt-in prefix-KV reuse and fork
  isolation for `KVCacheSimple`; local repeated-prompt TTFT moved from about
  11.7s to 0.3s in the measured lane.
- optimized Qwen3 TTS with GPU-side sampling and a depth-1 pipelined talker
  decode path, yielding roughly 20-25% faster local synthesis in the measured
  lane.
- optimized STT, OCR, embeddings, GLM-4.7 Flash, and ACEStep LM paths by
  reducing readbacks and batching decode/sampler operations; OCR throughput
  improved by roughly 17% in the measured lane.
- optimized FalconPerception decode with an O(T) KV cache and gated trace-only
  heads, replacing repeated full-prefix work in the vision-language path.
- tightened denoise-loop hygiene across HiDream, Klein, and Krea2 by
  precomputing sigma/control values, gating debug-only work, and hoisting
  repeated setup out of inner loops.
- made Magenta RT2 prompt swaps opt-in non-blocking, with a 1ms prompt-wait
  poll so realtime audio generation can keep moving while control state
  changes.
- Ornith lanes (`text-agent-ornith-9b`, `text-agent-ornith-35b-mlx`) now
  generate with thinking enabled by default in `text chat` and `api serve`,
  and default to the model's published sampling (temperature 1.0, top-p 0.95,
  top-k 20) when none is set explicitly. These R1-style tunes degenerate into
  repetition loops or signature echo without reasoning. `text chat` gains
  `--no-thinking` to disable reasoning generation and `--top-k` for explicit
  cutoff control; reasoning output stays hidden unless `--thinking` is passed.
  Other Qwen-family lanes keep the existing no-think default.
- improved the macOS Studio interface with a more polished command surface,
  clearer generated-media handling, and command-catalog coverage for the newer
  music, benchmark, plugin, and Open WebUI flows.
- refreshed release and Linux documentation around the supported public
  boundary: macOS app/DMG remain macOS-only, Linux artifacts stay headless
  CLI-only, x86_64 CPU/CUDA artifacts are the hosted release lanes, and arm64
  release packaging is CUDA/self-hosted only.
- bumped GitHub Actions dependencies, including checkout, upload-artifact, and
  action-gh-release, to the current major versions used by the release lanes.

### Fixed

- Qwen-family (Q35 runtime) MoE routing now renormalizes top-k router scores
  by default, matching the Qwen3.5 architecture default in HF transformers
  and mlx_lm. Checkpoints that omit `norm_topk_prob` (Ornith 35B, Qwen3.6
  nano) were previously routed with un-renormalized scores, dampening every
  MoE block's output by the top-k softmax mass and compounding into
  systematically repetition-biased logits, so reasoning models looped instead
  of terminating. Per-layer hidden states now match the mlx_lm reference to
  within quantized-kernel noise, and greedy decode openings match verbatim.
- `api serve` chat completions now report real `prompt_tokens` in the `usage`
  object for both streaming (`stream_options.include_usage`) and non-streaming
  responses, and `total_tokens` includes the prompt side. Previously
  `prompt_tokens` was always `0` and `total_tokens` only counted completion
  tokens.
- fixed stale MLX metallib handling. The runtime now detects unstamped or
  mismatched `default.metallib` resources, can replace stale libraries from a
  matching build product, fails loudly when no compatible metallib is
  available, and keeps the app bundle layout notarization-safe by flattening
  MLX resources under `Contents/Helpers`.
- fixed the Magenta RT2 engine failure caused by an unhydrated LFS metallib and
  removed the unsafe swap option that made the failure mode harder to reason
  about.
- fixed Linux CUDA Swift support linkage so hosted CUDA package builds can link
  against prepared native mlx-swift artifacts and explicit CUDA library paths.
- fixed macOS package bundle path capture in packaging scripts so release
  workflows inspect and sign the bundle that was actually built.
- fixed SAM 3.1 text-prompt tokenizer installation so segmentation prompts use
  the intended tokenizer assets.
- fixed strict-concurrency and stale test-support issues surfaced by Linux
  builds after the Qwen3 TTS pipelining work.
- fixed local generated artifacts by ignoring the root `live.wav` capture file
  without broadly ignoring audio fixtures.

### Removed

- removed public-repo maintainer release command wrappers; release orchestration
  now stays outside the public OSS distribution.

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
