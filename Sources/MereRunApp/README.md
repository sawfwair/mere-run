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
workflows.

Text uses the same contract for native/MLX chat, code, embeddings,
anonymization, and text-LoRA training. Chat exposes typed text/JSON response
format, reasoning policy, context and KV controls, LoRA application, tool
permissions, and preflight in both Studio and Advanced. Advanced also owns the
full text-LoRA training form.

Image uses the shared contract for generation/editing, LoRA training,
validation, dataset discovery, durable plans and dashboards, TripoSR,
TRELLIS.2, and InstantMesh. The primary Studio surface includes multi-reference
editing, structured prompts, LoRA catalog IDs or local adapters, Krea tuning,
preflight, and machine-readable progress; Advanced exposes every specialist
workflow and training control.

Music is a production workspace, not a prompt-only wrapper. Studio exposes
quality planning, covers/repaint/flow edits, source and timbre-reference audio,
candidate ranking, LM planning, adapter stacks, stems, LRC, recipes, and DAW
delivery. Advanced adds the complete ACE-Step diffusion/layout controls,
standalone audio understanding, MuScriptor MIDI/event transcription, Magenta
RT2 playback and MIDI steering, LoRA/LoKr training, and the resident music API.
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

Operations covers the verified adapter catalog, durable local/SSH/Relay run
listing and lifecycle, DreamX/Cosmos3 world sessions, server status, installed
model quality gates, physical model storage and safe garbage collection, and
typed per-model runtime policy. Server credentials are passed through
`MERERUN_API_KEY`, not process arguments. Visual Graph v2 authoring and fleet
policy stay in their canonical products: the app links directly to Graph
Studio and the Node/Relay console instead of reimplementing those surfaces.

Models, Setup, Speech, SFX, plugins, Qwen3.6 MTP and Laguna DFlash benchmarks,
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
