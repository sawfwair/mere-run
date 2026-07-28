# MereRunApp

Optional SwiftUI studio wrapper around the public `mere.run` CLI.

- `StudioTypes.swift`: user-facing mode, draft, and request types.
- `CommandCatalog.swift`: mode-to-command templates.
- `MereRunController.swift`: child-process launching and log capture.
- `StudioLibraryStore.swift`: local library persistence.

Do not duplicate runtime logic here. The app should translate UI state into CLI
arguments and let the public executable remain the behavioral source of truth.

`MereRunContract` is the compile-time and machine-readable boundary between the
two products. `mere.run catalog --json` emits that same contract. App forms must
use its typed choices, and CLI/App tests must prove that every emitted option is
both cataloged and accepted by ArgumentParser.

The primary Video surface uses `--quality` and `--output-mode`; it must never
emit the legacy `--variant` compatibility selector. Its attachment workflow
supports a start image, end keyframe, and source audio. Advanced Video contains
guided SCAIL-2, Cosmos3, mask-preparation, latent-export, and resident-session
workflows. Video also opens a first-class SCAIL Subject Studio with multi-subject
reference/selector authoring, preview and full-video SAM tracking, immutable
keyframe corrections, before/after playback, complete continuity/profile controls,
and durable Library jobs.

Text uses the same contract for native/MLX chat, code, embeddings,
anonymization, and text-LoRA training. Chat exposes typed text/JSON response
format, reasoning policy, context and KV controls, LoRA application, tool
permissions, and preflight. The first-class Utility Lab adds vector norms and
cosine-similarity inspection plus original/protected PII span review. Training
opens the unified Training Studio rather than a generic form.

Image uses the shared contract for generation/editing, LoRA training,
validation, dataset discovery, durable plans and dashboards, TripoSR,
TRELLIS.2, and InstantMesh. The primary Studio surface includes multi-reference
editing, structured prompts, LoRA catalog IDs or local adapters, Krea tuning,
preflight, and machine-readable progress. Training Studio adds dataset previews,
preflight, launch/resume, loss metrics, samples, checkpoints, and run comparison
for Krea 2 and FLUX.2 Klein. Utility Lab renders validation artifacts, candidate
dataset diagnostics, and materialized plan paths.
Create Image opens a first-class 3D Creation workspace for TripoSR, native
TRELLIS.2 PBR reconstruction, and ordered 4/6-view InstantMesh. It includes
engine-specific controls, immutable output directories, embedded orbitable
Quick Look models, manifest statistics, and the shared progress/Library lifecycle.

Music is a production workspace, not a prompt-only wrapper. Studio exposes
quality planning, covers/repaint/flow edits, source and timbre-reference audio,
candidate ranking, LM planning, adapter stacks, stems, LRC, recipes, and DAW
delivery. Music Tools adds standalone ACE-Step understanding with structured
results, MuScriptor transcription with an embedded MIDI piano roll, and resident
server health and lifecycle. Realtime opens the dedicated Magenta RT2 workspace;
Training opens the shared LoRA/LoKr studio with dataset audio previews and live
loss events. Advanced retains the complete raw command surface.
The API key is injected through `MERERUN_API_KEY`, never placed in process
arguments.

Vision keeps the quick Read Image path in Studio while Advanced exposes the
complete VLM/VFX family: multi-image captioning, LightOn/GLM/Infinity OCR,
grounding, text/box/point segmentation and tracking, camera capture, Buffalo-L
face analysis, native pose and optical flow, video depth, MoGe geometry, and
DA3 ordered multiview reconstruction. Coordinates remain typed, ordered CLI
arguments; machine-readable results and mask directories use explicit output
pickers. Image-to-3D workflows share the Image workspace instead of being
duplicated.
Every vision-oriented Studio mode also opens the first-class Advanced Vision Lab.
It renders face/pose overlays and dense optical-flow vectors, plays live tracking
and depth review videos, embeds geometry point clouds, and preserves every JSON,
EXR, mask, camera, and 3D sidecar as a durable Library artifact.

Operations covers the verified adapter catalog, durable local/SSH/Relay run
listing and lifecycle, DreamX/Cosmos3 world sessions, server status, installed
model quality gates, physical model storage and safe garbage collection, and
typed per-model runtime policy. Server credentials are passed through
`MERERUN_API_KEY`, not process arguments. Visual Graph v2 authoring and fleet
policy stay in their canonical products: the app links directly to Graph
Studio and the Node/Relay console instead of reimplementing those surfaces.

Serving is also a top-level first-class destination. **Serving & Agents**
owns API preflight/start/stop/restart, external-server reconnection, LAN/auth
safety, text and sidecar residency, load/unload and runtime policy, unified
memory/process CPU/Metal/thermal telemetry, observed request and cache/batching
traffic, typed Pi readiness/install/configure/session actions, copyable client
setup, and sanitized lifecycle activity. It polls the authenticated
`/runtime/status` contract and tolerates older payloads with missing additive
fields. App-owned servers and agent sessions remain durable Library runs; the
CLI/runtime remains the behavioral source of truth.

Voice Studio is the first-class path for styled or cloned synthesis, reusable
profiles, reference recording, streaming feedback, A/B playback, file/live
transcription, and transcript editing. SFX Lab covers text generation, video
Foley, conditioning, AE encode/decode, CLAP scoring, waveform review, NPY
metadata, and durable artifacts. The packaged app and embedded CLI carry the
microphone usage description and audio-input entitlement required by those
capture paths.

Models, Setup, plugins, Qwen3.6 MTP and Laguna DFlash benchmarks,
API serving, and Open WebUI also use contract-backed typed forms. Laguna is
available as a managed chat/API engine, while Chat and Code expose min-p and
the runtime-policy editors can persist or clear it. The run console recognizes
adapter catalogs and structured JSON receipts, and can copy or save a receipt.
Hugging Face tokens, API keys, and the Open WebUI admin password cross the
process boundary through environment variables instead of appearing in argv.
The executable contract test requires every local Advanced template and every
app-owned guide/config helper to resolve to a CLI help-verified capability.
The inverse coverage test also requires every command in the shared contract to
have an App-owned template or utility surface, so a newly cataloged CLI command
cannot silently ship without a macOS path.

`scripts/package-macos.sh` signs, notarizes, staples, and Gatekeeper-validates
the app before placing that already-stapled app into the signed and notarized
DMG. `LinuxNativeBridgeTests.testMacOSPackageEmbedsTheStapledAppBeforeCreatingTheDMG`
guards that ordering. Release proof must validate the mounted/installed app and
its embedded CLI, not only the outer DMG.

The packaged app embeds Sparkle and exposes **Check for Updates…** in the app
menu. Release builds use the stable HTTPS appcast, automatic daily discovery,
an Ed25519-signed archive and feed, Developer ID verification, and
pre-extraction signature validation. Sparkle updates the complete app bundle,
including its embedded CLI, as one atomic unit.
