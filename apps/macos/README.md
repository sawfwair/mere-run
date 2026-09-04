# mere.run Studio for macOS

Optional SwiftUI studio wrapper around the public `mere.run` CLI.

The packaged app registers two typed local-launcher routes. The strict
`mererun://preview?path=…` route accepts one readable absolute artifact path and
may show Quick Look but must not import or mutate artifacts. The strict
`mererun://library/import?receipt=…` route accepts one readable absolute receipt
path; `StudioLibraryStore` validates the versioned receipt and referenced media,
owns persistence and deduplication, and the window's `NavigationModel` opens the
imported row. External launchers must never edit `library.json` directly.

- `StudioNavigation.swift`: `StudioDomain`, `StudioTask`, `StudioDestination`,
  and the per-window `NavigationModel`.
- `StudioRootView.swift`: the `NavigationSplitView` shell, the content header
  (`StudioTaskControl.swift`), and the prompt workspace; hosts every task in
  the detail area.
- `StudioTypes.swift`: user-facing mode, draft, and request types.
- `CommandCatalog.swift`: `CommandTemplateID`, `CommandDraft`, and the
  `CommandTemplate` record type.
- `Catalog/`: one file per command category holding that category's
  `CommandTemplate` records and the function that builds each template's argv.
  `CommandFlags.swift` is generated from `MereRunCapabilityCatalog` by
  `./scripts/update-studio-command-flags.sh`, so every flag the app emits is a
  constant the shared contract declares and a renamed flag is a compile error;
  `ArgumentBuilder` appends positionals, switches, `--flag value` pairs,
  repeated options, and `--x` / `--no-x` pairs, and `optionUnlessDefault` drops
  a value the contract already declares as the CLI's default. `CommandDefaults`
  holds each template's starting draft, reading the contract's `default_value`
  where it declares one. `Fixtures/command-argv.txt` and
  `Fixtures/command-default-drafts.txt` in the test target pin both;
  `./scripts/update-studio-argv-fixture.sh` re-records them.
- `Jobs/`: the job model. `JobStore` owns every child process the app launches
  behind the `MereRunProcessRunning` seam (`Process()` appears only in
  `Jobs/ProcessRunner.swift` and the synchronous `CLIBootstrapInstaller`
  version probe), with three lanes: `inference` for Studio runs (capped at two
  with a FIFO queue), `utility` for the hand-built CLI reads and writes behind
  `utilityCommandResult` (capped at four, FIFO), and `probe` for readiness and
  `status --json` probes (never queued, deduplicated by key so a repeated probe
  joins the one in flight and a probe with stale Settings is superseded). A
  `JobRequest` is either a catalog command (template plus draft, which drive
  validation and output detection) or raw arguments (`JobRequest.utility` /
  `.probe`, which skip preflight and capture complete stdout and stderr for the
  submitter). The store publishes one observable `Job` per job (state, status,
  progress, log, live output, artifacts, result), raw output chunks through
  `events`, and a lossless `completions` stream.
- Output detection: `ArtifactResolver` reads, in order, (1) the CLI's
  `--receipt` line — the final stdout NDJSON object
  `{"event":"result","exit":0,"outputs":[…]}`, whose first entry is the primary
  artifact and whose sidecars carry a `role` (`detections`, `masks`, `recipe`,
  …); (2) the output kind `MereRunCapabilityCatalog` declares for the
  capability together with the `--output` path the request asked for, once it
  exists; (3) the stdout path contract and `fileExists` probing, kept as the
  fallback for the commands that print no receipt and for an older CLI. Roles
  reach the UI on `Artifact.sidecarRole` / `roleLabel` and are persisted on
  `StudioLibraryItem.artifactRoles`, so a result surface labels a sidecar
  instead of guessing from its extension; sidecars found by probing are
  labelled from the same draft fields that located them
  (`StudioArtifactRole.inferred`). The app appends `--receipt` and
  `--progress-json` to the launched argv for the capabilities the contract
  lists in `receiptCapabilityIDs` / `progressJSONCapabilityIDs`, never on a
  `--preflight` run (the CLI rejects that pair) and never in the "Will run"
  preview or a library row's command, which stay the command a person would
  type. A run that prints a receipt skips the 350 ms output poll; every other
  command with an output file keeps it.
- `MereRunController.swift`: the facade views bind to. It snapshots Settings
  and the CLI launch into a `JobRequest` for every lane, awaits utility and
  probe jobs on behalf of their callers (readiness results are evaluated
  against the request current at completion), mirrors the foreground inference
  job into its published console fields, and still owns the Advanced draft and
  persisted settings.
- `StudioLibraryStore.swift`: local library persistence.

## Shell

The window is one `NavigationSplitView`. The sidebar lists fifteen **domains**
in four sections — Create (Image, Video, Music, Sound, Voice, 3D), Converse
(Chat), Understand (Vision, Audio, Text, Earth), System (Models, Server, Runs,
Plugins) — with the machine status cluster as its only footer. The sidebar
header is the wordmark (`mere` and a green period in Caveat Medium, bundled
under `Resources/Fonts` and registered at launch by `MereRunTheme.Brand`), and
the selected row is a solid accent pill drawn by the row itself (the native
`List` highlight is switched off; selection, arrow keys, and VoiceOver are
unchanged). The footer pill reads "Ready · N models" once the status probe
answers, "Serving · N models" while the local server is up, and "Server
unreachable" (in red) if the probe never answers, with "· N running" appended
while jobs are in flight. It opens the **Activity popover**
(`StudioActivity.swift`), a 340pt panel the shell draws over the window from the
bottom-left: one row per running or queued job in the inference and utility
lanes (never a probe) with its progress and a stop control, over the app↔CLI
version handshake and a link into the Server page. With nothing running the same
panel shows the local server, the models root, and the resolved CLI path. It
reads the `JobStore` directly — the lanes for which rows exist, each `Job` for
its own progress — so nothing about the work in flight is mirrored on the
controller.

Every domain has **tasks** in a 52pt header at the top of the content column,
beside the Library (the window toolbar stays empty so the Library column runs to
the top of the window): one segmented pill for up to six tasks, or five segments
plus a "More" menu segment for Vision. The header's leading item is the domain
glyph, title, and one-line subtitle; trailing are the Library, Inspector, and
Command toggles (the Library and Inspector toggles appear on prompt tasks only).
Twelve tasks are the composer-driven prompt modes (`StudioMode`) and
keep the feed, composer, and Library column; every other task hosts a former
specialist sheet inline, full height, with its own controls and no Done button.
Nothing that used to be a sheet is modal any more; the remaining sheets are
true tasks (third-party terms, the image mask editor, Relay sign-in, rename,
and the Guide). `StudioDestination` (domain + task) persists per window under
`studio.destination`; `studio.mode` still records the last prompt mode so its
draft and readiness survive a detour through a System task.

The Library column appears on the prompt tasks (`StudioTask.isPromptTask`:
Generate, Compose, Speak, Chat, Code, Read, Find, Segment, Track, Transcribe);
every other task — Subjects, Realtime, Models, Train, the labs — takes the full
content width even inside a Create domain. Chat and Code fill that column with
their thread list instead (threads never file into the media Library). It is
filtered to the current domain by default with an All
segment — a row is filed under its command's domain (`CommandTemplateID.studioDomain`),
so 3D meshes land under 3D and benchmark reports under Models — and picking a
row from another domain switches the destination to it.

Beside the search field are a kind filter (All / Images / Video / Audio / Text,
plus "Favorites only") and a list-or-grid toggle; grid is three thumbnails
across with the title on hover. Thumbnails are the real thing per kind — the
picture, an `AVAssetImageGenerator` poster frame for video, a peak silhouette
for audio, the first line for a text result — decoded off the main actor and
cached by path, size, and modification date (`StudioLibraryThumbnail.swift`).
Rows carry a hover star (`StudioLibraryItem.isFavorite`, an additive optional),
rename in place, and drag out to Finder or any app. ⌘ and ⇧ click build a batch
(`StudioLibrarySelection`) with a bar for Reveal, Save to…, and Delete; Delete
asks first and offers to move the run's files to the Trash. Filtering and
day-grouping live in `StudioLibraryPresenter`, so both are testable without a
view. The view mode, kind, and favorites filter persist per window under
`studio.libraryView`, `studio.libraryKind`, and `studio.libraryFavorites`.

Every prompt task holds its own `StudioDraft`: switching tasks or domains parks
the one being left and restores the one being entered, so a typed prompt and an
attachment survive the detour. Across relaunches the scene remembers the prompt,
the system text, and the attachment of each task under `studio.drafts`
(`StudioDraftMemory`); settings come back from the task's defaults, which is
also the baseline the inspector diffs against.

**Chat** is the Converse archetype (`StudioConversationView.swift`,
`StudioThreadList.swift`). A **thread list** replaces the Library column there —
every chat and code thread, searchable, grouped Today / Earlier, with a compose
button (⌘N) — and threads never appear in the media Library. The thread header
carries the title, the model chip (the same filtered picker as the composer,
`StudioModelPicker.swift`), and a "System" chip that edits the system prompt in a
popover; changing either applies from the next turn, and every assistant turn
records the model, system prompt, and decode speed it ran with
(`StudioMessage.model` / `.systemPrompt` / `.tokensPerSecond`, additive optionals
in `library.json`). Under a reply: Copy, Retry, Branch, and "model · tok/s ·
time". Editing a user turn truncates the thread as before, or **Branch** starts
a new thread from that point (before a user turn, after an assistant turn). The
task control's **Code** is a preset inside the same thread list — the
`text code` command, its default model and system prompt, monospaced code
blocks with proportional prose — and a thread records which preset its latest
turn used (`mode`). While a reply streams, the composer's Send circle becomes
Stop for that thread. The transcript budget derives from the model's context
window when the inventory reports one (or an explicit context size), else stays
at 48k characters; a banner reports any turns trimmed from the next prompt.

The **composer** under the canvas is one surface for every prompt mode
(`StudioComposer.swift`, declarations in `StudioComposerSchema.swift`). An
**attachment well** shows the mode's slots — Image: input and reference images;
Video: start frame, end frame, audio; Music: source and timbre references;
Voice: reference audio; Vision and Audio tasks: their required input; Chat: a
per-turn image that stays behind the paperclip until attached — and every slot
takes a drop, a paste (⌘V), or a click to pick, storing straight into the draft
field the CLI flag reads. Under the prompt, a **chip strip** shows the mode's two
to four essentials (size, steps, seed, length, threshold, thinking) as menus, and
the **model chip** is the only model control: it lists `model list` rows filtered
to the mode's category, installed first, with "Auto" for the mode's default. The
chips and the inspector bind the same `StudioDraft`, so a value changed in one
shows in the other.

The **feed** above the composer (`StudioFeedCanvas.swift`, cards derived in
`StudioFeedCards.swift`) lists the mode's runs oldest first, newest beside the
composer. A finished run is a generation card: prompt, the chips it ran with
(read from its own command), every output in a grid of 236pt tiles (images,
video, 3D; audio gets the waveform player, text the Markdown renderer), and
Vary (rerun with a fresh, recorded seed), Rerun, Use as input, Quick Look,
Reveal, Copy, and Save to…; outputs drag out to Finder. Runs write to a
user-visible folder chosen by what the file is and which domain made it
(`StudioOutputLocation.swift`): `~/Pictures/mere.run/<Domain>` for pictures and
clips, `~/Music/mere.run/<Domain>` for audio, `~/Documents/mere.run/<Domain>`
for everything else, named `<slug-of-prompt>-<seed-or-short-id>.<ext>` with a
numeric suffix on collision. Settings ▸ General takes one root that overrides
all three (`mererun.app.outputRoot`). Nothing is migrated: Library rows keep the
paths they recorded, Application Support holds metadata only, and a destination
that cannot be created sends the run back to `App Outputs` with its sidecars and
one banner saying why. A run in flight is a
card that observes its `Job` directly — progress bar, "Denoising 15/24 · 0:41",
Cancel, and the log tail behind an Activity disclosure — and a queued run is a
row with Remove; both come from `JobStore`, not from the controller's foreground
mirror. A failed run leads with the last meaningful stderr line, keeps the log
behind "Show log", and offers Retry. Validation errors ("Prompt is required.")
render as a banner under the composer; readiness (missing model, missing CLI)
is a card at the bottom of the feed with "Get the model" and the pull's own
progress, so it never hides earlier work. Completion never moves the Library
selection; a result that finishes off-screen shows a "New result ↓" pill.
Picking a Library row scrolls to its card and outlines it briefly.

The input-first tasks — Vision ▸ Read, Find, Segment, and Track, and Audio
▸ Transcribe — render the **Analyze canvas** (`StudioAnalyzeCanvas.swift`,
`StudioAnalyzeViews.swift`) instead of the feed, because the answer belongs
beside the thing it is about rather than in a stream. It is one 940pt column:
an input strip naming the attached file with its dimensions or duration, a
Replace button that writes the same composer well, and a view switch whose
segments come from the task's own result kind (Boxes / Masks / JSON for Find
and Segment, Video / JSON for Track, Transcript / Timeline / JSON for
Transcribe); below it the input rendered large on the left — the image with
the result drawn over it, a video with its scrubber and per-object track spans,
audio with the waveform player — and a 360pt result column on the right
holding what the model found, the contextual next steps, and the prompt it ran
with. Results are read from the documents the CLI actually writes
(`StudioAnalyzeResults.swift`): `vision ground`'s normalized boxes, `vision
segment`'s pixel boxes with their mask PNGs, `vision track`'s per-frame
detections, `speech diarize`'s speaker turns, and the timestamped transcript
`speech transcribe` prints. Studio always asks for that document, passing
`--json-output` (and `--mask-output-dir` for the still tasks) beside the
annotated output. A run in flight, a queue, a failure, and readiness use the
feed's own cards above the result column, and earlier runs stay one click away
in the Library column, which also puts their input back in the composer. The
next steps open the sibling task carrying the input when the target accepts it
(Find ▸ "Segment these" keeps the picture; "Track in video" keeps only the
prompt, because Track needs a clip). The archetype is declared per task in
`StudioAnalyzeSchema.swift`, which also describes the Vision Lab, Audio, Text,
Earth, and Sound analysis tasks that still render their lab forms.

The **inspector** (⌥⌘I, the header's Inspector toggle, remembered per task under
`studio.inspectorTasks`) is a 300pt column rendered from the capability
contract. `StudioContractSchema.swift` binds each option
`MereRunCapabilityCatalog` declares to the `StudioDraft` field the app keeps it
in, and `ContractForm` (`StudioContractForm.swift`) draws it: the option's
`kind` picks the control (a field, a checkbox, a segmented control or pop-up
from its `choices`, a path well, a slider when its `range` has both ends and a
stepper when it does not), its `group` picks the section (Prompt, Inputs,
Output, Model & adapters, Sampling, Run), its `tier` decides whether it sits in
a section or under the collapsed "Advanced · N more", its `depends_on` hides it
until the option it needs carries a value, and a control at its `default_value`
emits no flag. A per-flag override registry keeps the composite editors the
contract cannot describe — the aspect presets with width × height and Swap, the
seed with Random and Reuse last, the steps and guidance sliders over the range
the mode's models use, seconds-or-frames, the model picker, the LoRA and
ACE-Step adapter rows, the mask and outpaint canvas, and the ordered MiniMax
references — and marks the attachments the composer's well owns so the
inspector never repeats them. Each section has Reset, and the header badge
counts the draft fields that differ from the mode's defaults.

Menus follow macOS convention: File ▸ New Chat (⌘N) and Import Receipt…, View ▸
Show Library (⌥⌘L), Show Inspector (⌥⌘I), Show Command View (⌥⌘C; Command
Console on other tasks), and the system sidebar toggle, Go ▸
every domain (⌘1–⌘9, then ⌥⌘1…) plus the current domain's tasks, Run ▸ Run
(⌘↩) and Stop (⌘.) acting on the current composer, Help ▸ Guide (⌘?). Settings
has General, Models, Server, and Advanced tabs. First run shows the Image empty
state with its "Get the model" path and a one-time dismissible banner.

The **Command view** (⌥⌘C on a prompt task, or the header's Command toggle)
is a 520pt column in the inspector's place — the two are never side by side —
showing the task's raw form from the same draft: every option the capability
contract declares for the template, under the contract's own groups — Prompt,
Inputs, Output, Model & adapters, Sampling, Run, and Options for a flag the
contract has yet to describe (`StudioCommandRows.swift`) — with the value the
argv carries (set rows first) or a switch for boolean flags, then "Will run"
with the masked command line, Copy, "Open in Terminal" (copies the command and
brings Terminal forward; the app never scripts another application), and Run.
Values are read-only until every task renders an editable contract form; the
chips and inspector edit them. Readiness cards' Details button opens it.

The **Command Console** window (`Window("Command Console")`) remains the raw
surface for every other template — template sidebar, editor, and run console in
three resizable panes. It opens from the header's Command toggle on non-prompt
tasks, ⌥⌘C there, Help ▸ Command Console anywhere, a Library row's "Edit
command…", and the adapter fallbacks for modes whose adapters are typed only in
the raw command. Opening it from the header carries the composer's draft into
the matching template; raising an already-open console only brings it forward,
so its edits stay. The console's own Run stays independent of the composer's,
and while the console is key the Run menu drives it while Go and Help keep
acting on the Studio window.

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
and source audio. The Command Console's Video templates contain
guided SCAIL-2, Cosmos3, mask-preparation, latent-export, and resident-session
workflows. Video ▸ Subjects hosts the SCAIL subject workflow as a three-stage
project board (Plan → Track → Animate) with a stage rail, a mask preview that
scrubs by frame and flips between masks and the driving clip, subject rows, and
stats read from the CLI's manifest, tracking, and quality reports. It keeps
multi-subject reference/selector authoring, preview and full-video SAM tracking,
immutable keyframe corrections, the continuity/profile controls (under each
stage's "More" row), plan.json persistence, and durable Library jobs with a job
bar for the running stage.

Text uses the same contract for native/MLX chat, code, embeddings,
anonymization, and text-LoRA training. Chat exposes typed text/JSON response
format, reasoning policy, context and KV controls, LoRA application, tool
permissions, and preflight. Text ▸ Embeddings adds vector norms and
cosine-similarity inspection; Text ▸ Anonymize shows original/protected PII
spans. Chat ▸ Train hosts the text trainer rather than a generic form. Laguna XS and
Inkling-Small are explicit model families; Inkling reasoning effort is available
in chat and training, while omitted target modules preserve the runtime's full
attention, MLP, expert, shared-outer, and unembedding training defaults.

Image uses the shared contract for generation/editing, LoRA training,
validation, dataset discovery, durable plans and dashboards, TripoSR,
TRELLIS.2, and InstantMesh. The primary Studio surface includes multi-reference
editing, structured prompts, LoRA catalog IDs or local adapters, Krea tuning,
preflight, and machine-readable progress. Image ▸ Train adds dataset previews,
preflight, launch/resume, loss metrics, samples, checkpoints, and run comparison
for Krea 2 and FLUX.2 Klein. Image ▸ Datasets renders validation artifacts,
candidate dataset diagnostics, and materialized plan paths.
The 3D domain is the workspace for TripoSR, native
TRELLIS.2 PBR reconstruction, and ordered 4/6-view InstantMesh. It includes
engine-specific controls, immutable output directories, embedded orbitable
Quick Look models, manifest statistics, and the shared progress/Library lifecycle.

Music is a production workspace, not a prompt-only wrapper. Studio exposes
quality planning, covers/repaint/flow edits, source and timbre-reference audio,
candidate ranking, LM planning, adapter stacks, stems, LRC, recipes, and DAW
delivery. Music ▸ Analyze adds standalone ACE-Step understanding with structured
results and Music ▸ Transcribe MuScriptor transcription with an embedded MIDI
piano roll; the resident ACE-Step server's health and lifecycle live under
Server ▸ Music server. Music ▸ Realtime is the Magenta RT2 session: a transport
with the live clock, the recording's waveform, Prompt A/B steering with a blend
slider, temperature, top-k, and guidance sent over the CLI's stdin protocol as
you release each control, the session log, and a job bar with Cancel and Log;
it re-attaches to a running session when you navigate back to it.
Music ▸ Train is the shared LoRA/LoKr trainer with dataset audio previews and live
loss events. The Command Console retains the complete raw command surface.
Audio ▸ Enhance and Audio ▸ Separate (also reachable as Music ▸ Separate) are
the restoration workspace for native AP-BWE and
UniverSR enhancement plus ViperX two-stem, four-stem, dereverb, and denoise
RoFormer workflows. It exposes model-specific compute/chunk controls, previews
source and outputs, and preserves manifests and every generated stem in Library.
The API key is injected through `MERERUN_API_KEY`, never placed in process
arguments.

Vision keeps the quick Read path in Studio while the Command Console exposes the
complete VLM/VFX family: multi-image captioning, LightOn/GLM/Infinity OCR,
grounding, text/box/point segmentation and tracking, camera capture, Buffalo-L
face analysis, native pose and optical flow, video depth, MoGe geometry, and
DA3 ordered multiview reconstruction. Coordinates remain typed, ordered CLI
arguments; machine-readable results and mask directories use explicit output
pickers. Image-to-3D lives in the 3D domain instead of being duplicated.
Vision ▸ Depth, Pose, Faces, Flow, Geometry, and Live host the former Vision Lab
inline. It renders face/pose overlays and dense optical-flow vectors, plays live
tracking and depth review videos, embeds geometry point clouds, and preserves
every JSON, EXR, mask, camera, and 3D sidecar as a durable Library artifact.

Runs is a sidebar domain over the public
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

Models ▸ Installed is a list-and-detail page. The content header's subtitle
reports the real inventory ("92 installed · 48 GB on this Mac", from `model list` and
`model storage`). The 320pt list shows installed models plus any model being
pulled, with a family chip row (Image, Chat, Vision, …) and a status dot: green
installed, accent pulling, yellow when the CLI reports the model as unsupported
on this Mac. Pull… opens a sheet of the models that are not installed yet; a
pull keeps its explicit third-party terms acceptance, live CLI output, and
cancellation with resumable partials. The detail column shows the model's
facts (store, source, last used and run count from the Library, manifest
verification from `model info`), a Health panel (latest quality gate and
manifest audit, with Run gate and Benchmark… routing to those tasks), a
Performance panel (last run length, unified-memory needs, latest benchmark),
the adapters whose base model it is (Use in <domain> applies one to the
composer, Train new… opens the trainer), and the runtime-settings editor and raw
`model info` output under two folds. Rows whose data the CLI or Library does
not have are omitted rather than faked. A job bar at the page bottom reports a
pull, MiniMax-H3 optimize/rebuild, or storage clean-up in flight (composer
pulls arrive through their Library row) with Cancel and Log. Reveal, Remove…,
refresh, opening the store, and clean-up stay on the page.

Models ▸ Health is the manifest audit and quality gate. Successful downloads
refresh the inventory in place. Manifest audit is a structured dry run, repair
requires confirmation and writes only missing known manifests, and
installed-model correctness/performance gates run as durable Library jobs with
JSON reports. Existing model browsing, storage cleanup, and runtime policy
remain in the Models and Server domains rather than being duplicated.

Server is a sidebar domain. **Server ▸ Serving**
owns API preflight/start/stop/restart, external-server reconnection, LAN/auth
safety, text and sidecar residency, load/unload and runtime policy, unified
memory/process CPU/Metal/thermal telemetry, observed request and cache/batching
traffic, typed Pi readiness/install/configure/session actions, copyable client
setup, and sanitized lifecycle activity. It polls the authenticated
`/runtime/status` contract and tolerates older payloads with missing additive
fields. App-owned servers and agent sessions remain durable Library runs; the
CLI/runtime remains the behavioral source of truth.

Voice ▸ Clone and Voice ▸ Voices host styled or cloned synthesis, reusable
profiles, reference recording, streaming feedback, and A/B playback; Audio ▸
Who Spoke and Audio ▸ Live host native Sortformer speaker diarization
with JSON/RTTM timelines and segment-tuning controls and live transcription.
Sound ▸ Video Foley, Condition, Encode, Decode, and Score cover video
Foley, conditioning, AE encode/decode, CLAP scoring, waveform review, NPY
metadata, and durable artifacts. The packaged app and embedded CLI carry the
microphone usage description and audio-input entitlement required by those
capture paths.

Plugins is a sidebar domain. It consumes the CLI's enriched
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
Earth is a sidebar domain for native Earth-observation inference, with Flood,
Fire, TESSERA, and OlmoEarth tasks. It covers TerraMind flood and fire tile inference
and the TESSERA v2 and OlmoEarth v1.2 encoders, names the tensors each input
bundle must carry before a run rather than after it, exposes engine-specific
controls (TESSERA output dimensions; OlmoEarth patch size, ground sample
distance, and space-time tokens), and preserves every produced safetensors file
as a durable Library artifact.

Models ▸ Locations is the store editor over `model location`. It shows the
writable store, read-only search roots, and explicit per-model bindings with
live availability, adds roots and bindings through a directory picker, reveals
any of them in Finder, and confirms before removing a root or a binding — so a
model kept on an external volume is registered without leaving the app.

Models ▸ Benchmarks runs the complete benchmark family as durable Library jobs
beside Models ▸ Health's manifest audit and quality gate: the fused Mere Lite and Mere
Comprehensive suites, the chat, code, and vision-language slices, tool-call and
tool-continuation evaluations, the Gemma4 KV and MTP and Qwen3.6 MTP decode
comparisons, Laguna DFlash, API workload replay, and fixture hashing. Each run
prints its JSON report to the Library row rather than to a file — the benchmarks
take no destination option — except the vision-language slice, whose
`--output-dir` collects the external lmms-eval run.

Audio ▸ Live is the live lane over `speech listen`. It enumerates capture
devices through the CLI, streams partial transcripts as the recognizer emits
them, and owns the child process directly so the operator can stop it, then
copy or save the transcript.

Server ▸ Serving has a Vision Grounding section with the same lifecycle the API server
has: preflight, app-owned start, stop, and restart, an honest reading of who
owns the process, a loopback-exposure warning when the endpoint would bind
beyond localhost without a key, and the live server log.

Plugins covers the full lifecycle: catalog details, out-of-PATH runs with
forwarded arguments, and rollback to a retained signed bundle behind a
confirmation.

The executable contract test requires every Command Console template and every
app-owned guide/config helper to resolve to a CLI help-verified capability.
The inverse coverage test also requires every command in the shared contract to
have an App-owned template or utility surface. A third test walks the CLI
command tree itself and requires every public leaf command to be either
cataloged or listed in `contractExemptCommandIDs` with the reason it stays
CLI-only, so a new CLI command cannot silently ship without a macOS path or a
recorded decision that it should not have one.

`StudioSnapshotTests` renders the shell for visual review without driving the
live app: every domain at its default task at 1280×820 in light and dark, plus
the Settings content and the Command Console, and fidelity renders at the
1440×900 mockup size for comparing against the design boards: the Main board
(Image ▸ Generate with a finished generation of two in-test images, a run held
open by the process seam mid-denoise, a queued run behind a concurrent model
pull, and the inspector open with two changed settings; then the same feed with
the Command view column), the composer with the boards' sample prompt
and an in-test image attached (Image ▸ Generate and Vision ▸ Find), the
Analyze board (Vision ▸ Find over a 1024×1024 in-test image with a seeded
`vision ground` document, and Audio ▸ Transcribe with a synthesized recording
and a timestamped transcript), Music ▸
Realtime mid-session (a run the process seam holds open, fed the CLI's frame
progress and steering echoes, with its recording synthesized on disk), and
Models ▸ Installed with a scripted model inventory (`model list`,
`capabilities`, `storage`, `info`, `runtime get`, and `adapter list` answered
from fixtures, plus Library rows for usage, a quality gate, a benchmark, and a
running composer pull). It is skipped unless
`MERERUN_STUDIO_SNAPSHOT_DIR` names a directory, so CI and a plain `swift test`
never render anything:

```
MERERUN_STUDIO_SNAPSHOT_DIR=/tmp/shell-shots swift test --filter StudioSnapshotTests
```

`StudioSnapshotRenderer` hosts each view in an `NSWindow` that is never ordered
on screen and captures it with `cacheDisplay`, so nothing appears on the Mac
running it. The controller uses a process runner that refuses every launch
(answering only the sidebar's status probe; for the Main board and Realtime
renders it holds the generation, pull, or session launches open, and for the
Main board and Models renders it answers the scripted `model list` and
`capabilities` commands) and
the Library is a temporary `library.json` seeded with fixture rows, so no CLI
process starts and the user's Library is never read or written. Two fidelity
gaps are deliberate: macOS 26 glass and scroll-edge effects only composite on
screen, so the renderer lifts glass content out and hides those effects (the
sidebar and toolbar draw as plain views), and the window keeps an opaque title
bar because a transparent one blanks every offscreen `ScrollView`.

The public `scripts/build_mere_run_app.sh` path produces a contributor/CI app
bundle and verifies its nested-code layout. Maintainer-only Developer ID
signing, notarization, stapling, DMG assembly, Sparkle feed generation, and
upload live in the separate private release-tools repository. Release proof
must validate the mounted/installed app and its embedded CLI, not only the
outer DMG.

Help exposes **Export Diagnostics…**, which writes a support report carrying app
and CLI versions, the resolved executable, machine shape, local server health,
and recent run outcomes. It contains no configuration values, API keys, or
tokens; command previews are already secret-masked when they are recorded.
Settings adds opt-in local crash and hang capture over MetricKit. It is off
until enabled, writes payloads to Application Support, never transmits them, and
lets the user reveal or delete them in one action.

The packaged app embeds Sparkle and exposes **Check for Updates…** in the app
menu. Release builds use the stable HTTPS appcast, automatic daily discovery,
an Ed25519-signed archive and feed, Developer ID verification, and
pre-extraction signature validation. Sparkle updates the complete app bundle,
including its embedded CLI, as one atomic unit.

Studio installs the Terminal CLI as a complete, versioned payload under
`~/Library/Application Support/mere.run/cli/`. The public `mere.run` command is
an atomic symlink to that payload. Studio records the destination, app build,
payload fingerprint, and installed assets in the
`~/Library/Application Support/mere.run/studio-cli-install.json` file.

After Studio starts, it synchronizes only an installation that still matches
this ownership receipt. Studio doesn't replace custom paths, package-manager
symlinks, or unmarked CLI copies. In **Settings**, use **Update CLI** to adopt
an unmarked copy or use **Repair CLI** after an owned payload changes. Studio
validates the staged command with `mere.run --version` before it updates the
public symlink.
