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
