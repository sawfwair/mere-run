# mere.run Studio for macOS

Optional SwiftUI studio wrapper around the public `mere.run` CLI.

The packaged app registers two typed local-launcher routes. The strict
`mererun://preview?path=…` route accepts one readable absolute artifact path and
may show Quick Look but must not import or mutate artifacts. The strict
`mererun://library/import?receipt=…` route accepts one readable absolute receipt
path; `StudioLibraryStore` validates the versioned receipt and referenced media,
owns persistence and deduplication, and publishes navigation through
`StudioNavigationCoordinator`. External launchers must never edit
`library.json` directly.

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

The primary Video surface uses model-family-aware controls: LTX uses `--quality`
and `--output-mode`, while native MiniMax-H3 exposes its exact `17*n+5` frame
cadence, adaptive or explicit denoising schedule, weight-residency policy, and
exact, balanced, or maximum denoise acceleration, plus
ordered Ref2VA image/video/audio references without emitting incompatible LTX
flags. Its general attachment workflow supports a start image, end keyframe,
and source audio. Advanced Video contains
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
opens the unified Training Studio rather than a generic form. Laguna XS and
Inkling-Small are explicit model families; Inkling reasoning effort is available
in chat and training, while omitted target modules preserve the runtime's full
attention, MLP, expert, shared-outer, and unembedding training defaults.

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
Audio Lab is the first-class restoration workspace for native AP-BWE and
UniverSR enhancement plus ViperX two-stem, four-stem, dereverb, and denoise
RoFormer workflows. It exposes model-specific compute/chunk controls, previews
source and outputs, and preserves manifests and every generated stem in Library.
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

Runs & Operations is a top-level first-class destination over the public
`executor` and `run` contracts. It discovers local durable reports, lists Relay
jobs, polls typed inspection state, shows artifact inventories, reveals local
runs, and exposes verified fetch, cancellation, and immutable Relay retry.
Client-side Relay profile setup and device sign-in also go through the CLI;
Studio streams the approval URL but never handles the credential itself.
Studio owns the creator's run inbox; Relay remains the control plane for node
identity, placement, scheduling policy, model distribution, worker lifecycle,
and fleet telemetry. The app links to the Relay console instead of copying
those schemas or controls.

Diorama is the separate first-class Worlds app and owns durable world projects,
navigation, exploration, saved routes, review, and `.diorama` bundles. Studio
owns only the typed local `world serve` runtime endpoint, authentication, model
selection, status, and the handoff to `https://diorama.mere.run`; it must not
duplicate Diorama's product experience.

Models includes one-click managed downloads with live CLI output, explicit
third-party terms acceptance, cancellation with resumable partials, and a
dedicated Health & Repair workspace. Successful downloads refresh the inventory
without reopening the sheet. Manifest audit is a structured dry run, repair
requires confirmation and writes only missing known manifests, and
installed-model correctness/performance gates run as durable Library jobs with
JSON reports. Existing model browsing, storage cleanup, and runtime policy
remain in the Models and Serving destinations rather than being duplicated.
Installed MiniMax-H3 models expose a native optimize/rebuild action with streamed
receipts instead of requiring a terminal round trip.

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
transcription, transcript editing, and native Sortformer speaker diarization
with JSON/RTTM timelines and segment-tuning controls. SFX Lab covers text generation, video
Foley, conditioning, AE encode/decode, CLAP scoring, waveform review, NPY
metadata, and durable artifacts. The packaged app and embedded CLI carry the
microphone usage description and audio-input entitlement required by those
capture paths.

Plugins is also a top-level catalog workspace. It consumes the CLI's enriched
plugin snapshot, shows installed path/version and manifest verification, offers
channel selection and copyable pinned install commands, confirms install or
update, and runs the plugin's fixed doctor verb. Plugin implementations remain
out-of-process.

Setup, the complete `model benchmark` family — fused Lite/Comprehensive quality
suites, chat, code, VLM, tool-call and tool-continuation slices, Gemma4 KV and
MTP, Qwen3.6 MTP, Laguna DFlash, and API workload replay — API serving, and Open
WebUI also use contract-backed typed forms. Laguna is
available as a managed chat/API engine, while Chat and Code expose min-p and
the runtime-policy editors can persist or clear it. The run console recognizes
adapter catalogs and structured JSON receipts, and can copy or save a receipt.
Hugging Face tokens, API keys, and the Open WebUI admin password cross the
process boundary through environment variables instead of appearing in argv.
Geospatial is a first-class category covering TerraMind flood and fire tile
inference and TESSERA and OlmoEarth encoders. Models adds read-only store
locations — search roots and explicit model bindings — so an external volume is
registered without leaving the app. Plugins covers catalog details, out-of-PATH
runs, and rollback to a retained signed bundle. Voice Studio streams live
microphone transcription in addition to file transcription, and Serving hosts
the resident vision grounding endpoint.

The executable contract test requires every local Advanced template and every
app-owned guide/config helper to resolve to a CLI help-verified capability.
The inverse coverage test also requires every command in the shared contract to
have an App-owned template or utility surface. A third test walks the CLI
command tree itself and requires every public leaf command to be either
cataloged or listed in `contractExemptCommandIDs` with the reason it stays
CLI-only, so a new CLI command cannot silently ship without a macOS path or a
recorded decision that it should not have one.

The public `scripts/build_mere_run_app.sh` path produces a contributor/CI app
bundle and verifies its nested-code layout. Maintainer-only Developer ID
signing, notarization, stapling, DMG assembly, Sparkle feed generation, and
upload live in the separate private release-tools repository. Release proof
must validate the mounted/installed app and its embedded CLI, not only the
outer DMG.

The packaged app embeds Sparkle and exposes **Check for Updates…** in the app
menu. Release builds use the stable HTTPS appcast, automatic daily discovery,
an Ed25519-signed archive and feed, Developer ID verification, and
pre-extraction signature validation. Sparkle updates the complete app bundle,
including its embedded CLI, as one atomic unit.
