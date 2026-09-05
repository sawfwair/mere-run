# mere.run Studio for macOS

Optional SwiftUI studio wrapper around the public `mere.run` CLI.

The CLI (`Sources/MereRunCLI`, `Sources/MereRunCore`) is the behavioral source
of truth. The app translates UI state into CLI arguments, launches the public
executable as a child process, and renders what comes back. Do not duplicate
runtime logic here.

`MereRunContract` is the compile-time and machine-readable boundary between the
two products, and `mere.run catalog --json` emits the same document. App forms
use its typed options; tests prove the match in both directions.

## Targets

The Studio is three SwiftPM targets, so the model layer can be built and tested
without SwiftUI and the views can be rendered without the app's scenes:

- `StudioKit/` — library, no SwiftUI: the CLI resolver and
  environment, `ProcessRunner`, `Job`/`JobStore`/lanes/`ArtifactResolver`/progress,
  the command catalog (generated flags, `ArgumentBuilder`, `CommandDraft`), the
  Library store and its receipts, the conversation transcript, the navigation
  vocabulary (`StudioDomain`, `StudioTask`, `StudioDestination`), the declarative
  schemas the views render from, readiness, configuration, the serving monitor,
  the CLI installer, diagnostics, and crash reporting. Every one of those is
  unit-testable without hosting a view.
- `StudioUI/` — library, SwiftUI: the shell, `NavigationModel`, the
  boards (feed, inspector, Command view, analyze, converse, session, project,
  manage), `ContractForm`, the theme (which owns the bundled Caveat wordmark
  font), the controls, and the result renderers.
- `MereRunStudio/` — the `mere.run.app` executable:
  `MereRunApp`, the app delegate, the menu bar commands, the Settings scene, and
  the Sparkle wiring. Everything else it does, it does by composing the two
  libraries.

Declarations that cross a target boundary are `package`, which is exactly the
visibility they had while the Studio was one target; nothing became `public`.

Tests follow the same split: `StudioKitTests/` (model layer, including the argv
and default-draft fixtures), `StudioUITests/` (views, presenters, and the
offscreen snapshot harness), and `MereRunStudioTests/` (the executable, built as
the `MereRunAppTests` target). Shared test doubles live in `StudioTestSupport/`,
which nothing that ships depends on.

## Shape of the app

The window is one `NavigationSplitView`. The sidebar lists fifteen **domains**;
each domain has **tasks** in a segmented control at the top of the content
column. `StudioDestination` — one domain and one of its tasks — is the whole
navigation state. Three sheets remain, each a true task: the image mask editor,
Models ▸ Installed's Pull… catalog, and the Guide. Everything else that
interrupts is an alert or a confirmation dialog — third-party model terms,
removals, thread rename, Library delete, and a bad `mererun://` link.

Fifty-four tasks fall into three shapes:

- Twelve **prompt tasks** back a `StudioMode` and render the composer, a canvas,
  and the Library column: Image ▸ Generate, Video ▸ Generate, Music ▸ Compose,
  Sound ▸ Generate, Voice ▸ Speak, Chat ▸ Chat, Chat ▸ Code, Vision ▸ Read,
  Find, Segment, Track, and Audio ▸ Transcribe.
- Five of those twelve — Vision ▸ Read, Find, Segment, Track and
  Audio ▸ Transcribe — are input-first, so their canvas is the **Analyze**
  surface rather than the generation feed. Chat and Code get the **Converse**
  surface and a thread list in place of the Library.
- The other forty-two tasks host a full-height view of their own: the project
  boards (Video ▸ Subjects, the three Train tasks), the session pages
  (Music ▸ Realtime, Audio ▸ Live, Vision ▸ Live, Server ▸ Serving), the
  management pages (Models, Runs, Plugins, Voice ▸ Voices), and the analysis
  forms that have not moved to the Analyze surface yet.

Every task has an editable **Command** panel. A separate **Command Console**
window provides the complete command catalog. Both are
rendered from the capability contract, so a capability nobody has designed a
surface for is still reachable the day the contract declares it.

## Source map

- `StudioKit/StudioNavigation.swift`: `StudioDomain`, `StudioTask`,
  `StudioDestination`, and the `@SceneStorage` codecs. The per-window
  `NavigationModel` that drives them is `StudioUI/StudioNavigationModel.swift`.
- `StudioUI/StudioRootView.swift`: the `NavigationSplitView` shell, the content
  header (`StudioUI/StudioTaskControl.swift`), and the prompt workspace; hosts
  every task in the detail area.
- `StudioKit/StudioTypes.swift`: user-facing mode, draft, and request types.
- `StudioKit/CommandCatalog.swift`: `CommandTemplateID`, `CommandDraft`, and the
  `CommandTemplate` record type.
- `StudioKit/Catalog/`: one file per command category holding that category's
  `CommandTemplate` records and the function that builds each template's argv.
  `CommandFlags.swift` is generated from `MereRunCapabilityCatalog` by
  `./scripts/update-studio-command-flags.sh`, so every flag the app emits is a
  constant the shared contract declares and a renamed flag is a compile error;
  `ArgumentBuilder` appends positionals, switches, `--flag value` pairs,
  repeated options, and `--x` / `--no-x` pairs, and `optionUnlessDefault` drops
  a value the contract already declares as the CLI's default. `CommandDefaults`
  holds each template's starting draft, reading the contract's `default_value`
  where it declares one. `StudioKitTests/Fixtures/command-argv.txt` and
  `command-default-drafts.txt` pin both;
  `./scripts/update-studio-argv-fixture.sh` re-records them.
- `StudioKit/Jobs/`: the job model — `Job`, `JobStore`, `ArtifactResolver`,
  `ProcessRunner`, and the read-only `StudioJobMonitor`.
- `StudioKit/MereRunController.swift`: the facade views bind to.
- `StudioKit/StudioLibraryStore.swift`: local library persistence.

The declarative schemas live in StudioKit beside the model, and the views that
draw them in StudioUI, one file each side:
`StudioComposerSchema.swift` and `StudioComposer.swift`,
`StudioContractSchema.swift` and `StudioContractForm.swift`,
`StudioInspectorSchema.swift` and `StudioInspector.swift`,
`StudioAnalyzeSchema.swift` / `StudioAnalyzeResults.swift` and
`StudioAnalyzeCanvas.swift` / `StudioAnalyzeViews.swift`,
`StudioCommandRows.swift` and `StudioCommandView.swift`,
`StudioConsoleDraft.swift` and `StudioConsoleView.swift`,
`StudioFeedCards.swift` and `StudioFeedCanvas.swift`,
`StudioLibraryPresentation.swift` and `StudioLibraryPanel.swift`. That is what
makes a surface's rules testable without rendering it.

To open the offline handbook, in **Help**, select **mere.run Guide**. The
**Models** collection contains original recipes for 139 managed IDs, grouped
into 59 families. Use the search field to find a family or exact model ID.
The **Commands** collection contains command cookbooks.

Both collections use the CLI resource bundle. Reading guides requires neither
model weights nor a network connection. The app packaging script includes the
bundle with the embedded CLI and the CLI installed by Studio. Each recipe
records its sources and inference validation status.

The packaged app registers two typed local-launcher routes. The strict
`mererun://preview?path=…` route accepts one readable absolute artifact path and
may show Quick Look but must not import or mutate artifacts. The strict
`mererun://library/import?receipt=…` route accepts one readable absolute receipt
path; `StudioLibraryStore` validates the versioned receipt and referenced media,
owns persistence and deduplication, and the window's `NavigationModel` opens the
imported row. External launchers must never edit `library.json` directly.

## Shell

The sidebar lists the fifteen domains in four sections — Create (Image, Video,
Music, Sound, Voice, 3D), Converse (Chat), Understand (Vision, Audio, Text,
Earth), System (Models, Server, Runs, Plugins) — with the machine status cluster
as its only footer. The sidebar header is the wordmark (`mere` and a green
period in Caveat Medium, bundled under `Resources/Fonts` and registered at
launch by `MereRunTheme.Brand`), and the selected row is a solid accent pill
drawn by the row itself (the native `List` highlight is switched off; selection,
arrow keys, and VoiceOver are unchanged).

The footer pill reads "Ready · N models" once the status probe answers,
"Serving · N models" while the local server is up, and "Server unreachable" (in
red) if the probe never answers within six seconds, with "N running" on a second
line while jobs are in flight. It opens the **Activity popover**
(`StudioUI/StudioActivity.swift`), a 340pt panel the shell draws over the window from the
bottom-left: one row per running or queued job in the inference and utility
lanes (never a probe) with its progress and a stop control, over the app↔CLI
version handshake and a link into the Server page. With nothing running the same
panel shows the local server, the models root, and the resolved CLI path. It
reads the `JobStore` directly — the lanes for which rows exist, each `Job` for
its own progress — so nothing about the work in flight is mirrored on the
controller.

The sidebar toggle and task control share a 52pt header with the panel controls.
The split view does not add a separate toolbar row. Control-Command-S toggles the
sidebar, and the header background supports window dragging. Native traffic
lights remain available. The header sits at the top of the content column, beside
the Library (the window toolbar stays empty so the Library column runs to the
top of the window): one segmented pill for up to six tasks, or five segments
plus a "More" menu segment, which today only Vision's ten tasks reach. The
header's leading item is the domain glyph, title, and one-line subtitle;
trailing are the Library, Inspector, and Command toggles (Library and Inspector
appear on prompt tasks only). `StudioDestination` persists per window under
`studio.destination`; `studio.mode` still records the last prompt mode so its
draft and readiness survive a detour through a System task.

The Library column appears on the prompt tasks (`StudioTask.isPromptTask`);
every other task — Subjects, Realtime, Models, Train, the labs — takes the full
content width even inside a Create domain. Chat and Code fill that column with
their thread list instead (threads never file into the media Library). It is
filtered to the current domain by default with an All segment — a row is filed
under its command's domain (`CommandTemplateID.studioDomain`), so 3D meshes land
under 3D and benchmark reports under Models — and picking a row from another
domain switches the destination to it.

Beside the search field are a kind filter (All / Images / Video / Audio / Text,
plus "Favorites only") and a list-or-grid toggle; grid is three thumbnails
across with the title on hover. Thumbnails are the real thing per kind — the
picture, an `AVAssetImageGenerator` poster frame for video, a peak silhouette
for audio, the first line for a text result — decoded off the main actor and
cached by path, size, and modification date (`StudioUI/StudioLibraryThumbnail.swift`).
Rows carry a hover star (`StudioLibraryItem.isFavorite`, an additive optional
written as `nil` when unstarred), rename in place, and drag out to Finder or any
app. ⌘ and ⇧ click build a batch (`StudioLibrarySelection`) with a bar for
Reveal, Save to…, and Delete; Delete asks first and offers to move the run's
files to the Trash. Filtering and day-grouping live in `StudioLibraryPresenter`,
so both are testable without a view. The view mode, kind, and favorites filter
persist per window under `studio.libraryView`, `studio.libraryKind`, and
`studio.libraryFavorites`.

Each task retains its full draft and selected run through `StudioTaskSessions`.
Prompt modes preserve model, seed, dimensions, attachments, and sampling values;
specialist forms retain their typed settings. The versioned JSON store excludes
launch credentials and preserves unreadable files. `studio.drafts` remains a
migration fallback for older prompt-only scene state.

Menus follow macOS convention: File ▸ New Chat (⌘N) and Import Receipt…; View ▸
Show Library (⌥⌘L), Show Inspector (⌥⌘I), Show Command View (⌥⌘C), and the system sidebar toggle; Go ▸ every domain
(⌘1–⌘9, then ⌥⌘1…) plus the current domain's tasks; Run ▸ Run (⌘↩), Stop (⌘.),
Open Last Output (⇧⌘O), and Reveal Last Output in Finder (⇧⌘R), acting on the
current composer; Help ▸ mere.run Guide (⌘?), the mere.run link, Command
Console, and Export Diagnostics…. Settings has General, Models, Server, and
Advanced tabs. First run shows the Image empty state with its "Get the model"
path and a one-time dismissible banner; there is no Welcome sheet.

## Composer, feed, and Analyze

The **composer** under the canvas is one surface for every prompt mode
(`StudioUI/StudioComposer.swift`, declarations in `StudioKit/StudioComposerSchema.swift`). An
**attachment well** shows the mode's slots — Image: input and reference images;
Video: start frame, end frame, audio; Music: source and timbre references;
Voice: reference audio; Vision and Audio tasks: their required input; Chat: a
per-turn image that stays behind the paperclip until attached — and every slot
takes a drop, a paste (⌘V), or a click to pick, storing straight into the draft
field the CLI flag reads. Code and Sound ▸ Generate declare no slots. Under the
prompt, a **chip strip** shows up to four essentials (size, length, steps, seed,
threshold, task, voice mode, thinking) as menus with popover editors for custom
values; some modes show only the model chip. The **model chip** is the only
model control: it lists `model list` rows filtered to the mode's category,
installed first, with "Auto" for the mode's default. The chips and the inspector
bind the same `StudioDraft`, so a value changed in one shows in the other. ⌘↩
runs; while a conversation turn streams, the send circle becomes Stop.

The **feed** above the composer (`StudioUI/StudioFeedCanvas.swift`, cards derived in
`StudioKit/StudioFeedCards.swift`) lists the mode's runs oldest first, newest beside the
composer. A finished run is a generation card: prompt, the chips it ran with
(read from its own command), every output in a grid of 236pt tiles (images,
video, 3D; audio gets the waveform player, text the Markdown renderer), and
Vary (rerun with a fresh, recorded seed), Rerun, Use as input, Quick Look,
Reveal, Copy, and Save to…; outputs drag out to Finder. A run in flight is a
card that observes its `Job` directly — progress bar, "Denoising 15/24 · 0:41",
Cancel, and the log tail behind an Activity disclosure — and a queued run is a
row with Remove; both come from `JobStore`, not from a controller mirror. A
failed run leads with the last meaningful stderr line, keeps the log behind
"Show log", and offers Retry. Validation errors ("Prompt is required.") render
as a banner under the composer; readiness (missing model, missing CLI) is a card
at the bottom of the feed with "Get the model" and the pull's own progress, so
it never hides earlier work. Completion never moves the Library selection; a
result that finishes off-screen shows a "New result ↓" pill. Picking a Library
row scrolls to its card and outlines it briefly.

The input-first tasks render the **Analyze canvas**
(`StudioUI/StudioAnalyzeCanvas.swift`, `StudioUI/StudioAnalyzeViews.swift`) instead of the feed,
because the answer belongs beside the thing it is about rather than in a stream.
It is one 940pt column: an input strip naming the attached file with its
dimensions or duration, a Replace button that writes the same composer well, and
a view switch whose segments come from the task's own result kind (Boxes /
Masks / JSON for Find and Segment, Video / JSON for Track, Transcript /
Timeline / JSON for Transcribe); below it the input rendered large on the left —
the image with the result drawn over it, a video with its scrubber and
per-object track spans, audio with the waveform player — and a 360pt result
column on the right holding what the model found, the contextual next steps, and
the prompt it ran with. Results are read from the documents the CLI actually
writes (`StudioKit/StudioAnalyzeResults.swift`): `vision ground`'s normalized boxes,
`vision segment`'s pixel boxes with their mask PNGs, `vision track`'s per-frame
detections, `speech diarize`'s speaker turns, and the timestamped transcript
`speech transcribe` prints. Studio always asks for that document, passing
`--json-output` (and `--mask-output-dir` for the still tasks) beside the
annotated output. A run in flight, a queue, a failure, and readiness use the
feed's own cards above the result column, and earlier runs stay one click away
in the Library column, which also puts their input back in the composer. The
next steps open the sibling task carrying the input when the target accepts it
(Find ▸ "Segment these" keeps the picture; "Track in video" keeps only the
prompt, because Track needs a clip).

`StudioKit/StudioAnalyzeSchema.swift` declares the surface — the result views and the next
steps — for twenty-two tasks, seventeen of which still render their own form
inside their task rather than this canvas (Vision ▸ Depth, Pose, Faces, Flow,
Geometry, Live; Audio ▸ Who Spoke, Enhance, Separate; Text ▸ Embeddings,
Anonymize; the four Earth tasks; Sound ▸ Score and Condition). Migrating one is
a view change, not a design decision.

**Chat** is the Converse surface (`StudioUI/StudioConversationView.swift`,
`StudioUI/StudioThreadList.swift`). A **thread list** replaces the Library column there —
every chat and code thread, searchable, grouped Today / Earlier, with a compose
button (⌘N) — and threads never appear in the media Library. The thread header
carries the title, the model chip (the same filtered picker as the composer,
`StudioUI/StudioModelPicker.swift`), and a "System" chip that edits the system prompt in a
popover; changing either applies from the next turn, and every assistant turn
records the model, system prompt, and decode speed it ran with
(`StudioMessage.model` / `.systemPrompt` / `.tokensPerSecond`, additive optionals
in `library.json`). Under a reply: Copy, Retry, Branch, and "model · tok/s ·
time". Editing a user turn truncates the thread as before, or **Branch** starts
a new thread from that point (before a user turn, after an assistant turn). The
task control's **Code** is a preset inside the same thread list — the
`text code` command, its default model and system prompt, monospaced code
blocks with proportional prose — and a thread records which preset its latest
turn used (`mode`). The transcript budget derives from the model's context
window when the inventory reports one (or an explicit context size), else stays
at 48k characters; a banner reports any turns trimmed from the next prompt.

## Inspector, Command view, and Command Console

The **inspector** (⌥⌘I, the header's Inspector toggle, remembered per task under
`studio.inspectorTasks`) is a 300pt column rendered from the capability
contract. `StudioKit/StudioContractSchema.swift` binds each option
`MereRunCapabilityCatalog` declares to the `StudioDraft` field the app keeps it
in, and `ContractForm` (`StudioUI/StudioContractForm.swift`) draws it: the option's
`kind` picks the control (a field, a checkbox, a segmented control or pop-up
from its `choices`, a path well, a slider when its `range` has both ends and a
stepper when it does not), its `group` picks the section (Prompt, Inputs,
Output, Model & adapters, Sampling, Run), its `tier` decides whether it sits in
a section or under the collapsed "Advanced · N more", its `depends_on` hides it
until the option it needs carries a value, and a control at its `default_value`
emits no flag. A per-flag override registry keeps the fifteen composite editors
the contract cannot describe — the aspect presets with width × height and Swap,
the seed with Random and Reuse last, the steps and guidance sliders over the
range the mode's models use, seconds-or-frames, the model picker, the LoRA and
ACE-Step adapter rows, the mask and outpaint canvas, the ordered MiniMax
references, and the voice profile list — and marks the attachments the
composer's well owns so the inspector never repeats them. Each section has
Reset, and the header badge counts the draft fields that differ from the mode's
defaults.

The inspector shows only the flags the binding table maps to a draft field, so
no control can look live and change nothing. That makes it thin where the table
is thin: Read, Find, Segment, Track, and Code bind between one and five flags,
and the rest of their options are reached in the Command Console.

The **Command** panel (⌥⌘C or the header toggle) exposes the current task's
complete editable contract. It replaces the inspector and uses a 440-point
column when space permits, otherwise an overlay. The preview, validation, and
run use the edited arguments, including options absent from the simple controls.
Prompt controls and their mapped Command fields synchronize. Specialist Run
buttons retain Command edits while accepting later edits from their own forms.

The **Command Console** window (`Window("Command Console")`,
`StudioUI/StudioConsoleView.swift`) is the editable raw surface for every capability, in
three resizable panes: the catalog of templates by category, the selected
capability's form, and the run's log with its receipt and artifact
(`StudioUI/StudioConsoleLog.swift`). The middle pane is the same `ContractForm`, drawn
with the flag rather than the label at the head of each row and with numbers
typed rather than dragged, so the console has no per-command view of its own:
`StudioConsoleDraft` keeps one value per flag, `StudioConsoleCommand` reads a
template's own argv into those values and builds the argv back out of them, and
the eyebrows, controls, dependencies, positional arguments and "Will run" block
all come from `MereRunCapabilityCatalog`. Nothing is filtered by tier and
nothing is compared against a default: the console emits a flag exactly when the
draft holds a value, because what it shows is what it runs.
`StudioConsoleDraftTests` holds the identity that makes that safe: for every
template in the catalog, seeding from its default command and rebuilding
produces the same command. Options the contract does not describe go in Extra
arguments; the Custom template has no capability and keeps the catalog's raw
argument editor, the one editor the console still writes by hand.

The console opens from Help ▸ Command Console, a Library row's **Edit command…**,
and adapter fallbacks. Opening it from a task carries that task's command;
 a Library row reopens on the exact argv its run launched
(`StudioLibraryItem.commandArguments`, an additive optional, so the console can
set options no `CommandDraft` field carries); raising an already-open console
only brings it forward, so its edits stay. A console run is a normal inference
job — same queue, progress, artifact resolution and Library row — and while the
console is key the Run menu drives it while Go and Help keep acting on the
Studio window.

## Focus, compare, and continue

Click an image or **Focus** on its result card to inspect it in the workspace.
**Compare** selects another result and links zoom and pan. The settings area
shows differences between the recorded commands. **Continue with…** opens a
new draft for editing, reference guidance, video, image understanding, or
segmentation. The resulting run records its parent; the original stays in Library.
**Save copy…** copies before replacing a destination and treats saving onto the
source as a no-op.

## Jobs, artifacts, and output

`StudioAppSession` attaches `StudioLibraryStore` to job events and owns the
serving monitor for the lifetime of the app. Views select jobs; they do not own
completion recording. Cancellation and interrupted sessions have distinct
Library statuses. Console Stop and the Run menu act on the Console's selected
job; a chat's Stop acts on that thread's turn.


`JobStore` owns every child process the app launches behind the
`MereRunProcessRunning` seam (`Process()` appears only in
`StudioKit/Jobs/ProcessRunner.swift` and the synchronous `CLIBootstrapInstaller` version
probe), with three lanes: `inference` for Studio runs (capped at two with a FIFO
queue), `utility` for the hand-built CLI reads and writes behind
`utilityCommandResult` (capped at four, FIFO), and `probe` for readiness and
`status --json` probes (never queued, deduplicated by key so a repeated probe
joins the one in flight and a probe with stale Settings is superseded). A
`JobRequest` is either a catalog command (template plus draft, which drive
validation and output detection) or raw arguments (`JobRequest.utility` /
`.probe`, which skip preflight and capture complete stdout and stderr for the
submitter). The store publishes one observable `Job` per job (state, status,
progress, log, live output, artifacts, result), raw output chunks through
`events`, and a lossless `completions` stream, and it retains the last fifty
finished jobs per lane so utility churn cannot evict a run the user is reading.
`StudioRootView` keeps the Library rows current from those two streams.

`ArtifactResolver` reads, in order, (1) the CLI's `--receipt` line — the final
stdout NDJSON object `{"event":"result","exit":0,"outputs":[…]}`, whose first
entry is the primary artifact and whose sidecars carry a `role` (`detections`,
`masks`, `recipe`, …); (2) the output kind `MereRunCapabilityCatalog` declares
for the capability together with the `--output` path the request asked for, once
it exists; (3) the stdout path contract and `fileExists` probing, kept as the
fallback for the commands that print no receipt and for an older CLI. Roles
reach the UI on `Artifact.sidecarRole` / `roleLabel` and are persisted on
`StudioLibraryItem.artifactRoles`, so a result surface labels a sidecar instead
of guessing from its extension; sidecars found by probing are labelled from the
recorded draft fields that located them (`StudioArtifactRole.inferred`).

The app appends `--receipt` and `--progress-json` to the launched argv for the
capabilities the contract lists in `receiptCapabilityIDs` (nine) and
`progressJSONCapabilityIDs` (five), never on a `--preflight` run (the CLI
rejects `--receipt --preflight`, and a preflight has no progress to stream) and
never in the "Will run" preview or a Library row's command, which stay the
command a person would type. A run that prints a receipt skips the 350 ms output
poll; every other command with an output file keeps it.

Runs write to a user-visible folder chosen by what the file is and which domain
made it (`StudioKit/StudioOutputLocation.swift`): `~/Pictures/mere.run/<Domain>` for
pictures and clips, `~/Music/mere.run/<Domain>` for audio,
`~/Documents/mere.run/<Domain>` for everything else, named
`<slug-of-prompt>-<seed-or-short-id>.<ext>` with a numeric suffix on collision.
The suffix is derived, not random, so the path the Command view previews is the
path the run writes. Settings ▸ General takes one root that overrides all three
(`mererun.app.outputRoot`). Nothing is migrated: Library rows keep the paths
they recorded, Application Support holds metadata only, and a destination that
cannot be created sends the run back to `App Outputs` with its sidecars and one
banner saying why.

`MereRunController` is the facade views bind to. It snapshots Settings and the
CLI launch into a `JobRequest` for every lane, awaits utility and probe jobs on
behalf of their callers (readiness results are evaluated against the request
current at completion), mirrors the foreground inference job into the published
compatibility fields some management views still read, and owns
the template selection the console opens on and the persisted settings.

## Domains

**Image** covers generation and editing, LoRA training, validation, dataset
discovery, durable plans and dashboards. Image ▸ Generate includes
multi-reference editing, structured prompts, LoRA catalog IDs or local adapters,
Krea tuning, and preflight. Image ▸ Train adds dataset previews, preflight,
launch and resume, loss metrics, samples, checkpoints, and run comparison for
Krea 2 and FLUX.2 Klein. Image ▸ Datasets renders validation artifacts,
candidate dataset diagnostics, and materialized plan paths.

**Video** ▸ Generate uses model-family-aware controls: LTX uses `--quality` and
`--output-mode`, while native MiniMax-H3 exposes its exact `17n+5` frame
cadence, adaptive or explicit denoising schedule, weight-residency policy,
exact/balanced/maximum denoise acceleration, and ordered Ref2VA image, video,
and audio references, without emitting incompatible LTX flags. Its attachment
well takes a start image, an end keyframe, and source audio. Video ▸ Subjects is
the SCAIL subject flow as a three-stage project board (Plan → Track → Animate)
with a stage rail, a mask preview that scrubs by frame and flips between masks
and the driving clip, subject rows, and stats read only from the CLI's manifest,
tracking, and quality reports. It keeps multi-subject reference and selector
authoring, preview and full-video SAM tracking, immutable keyframe corrections,
the continuity and profile controls under each stage's "More" row, `plan.json`
persistence, and durable Library jobs with a job bar for the running stage. The
guided SCAIL-2, Cosmos3, mask-preparation, latent-export, and resident-session
commands live in the Command Console.

**Music** is a production surface, not a prompt-only wrapper: quality planning,
covers, repaint and flow edits, source and timbre-reference audio, candidate
ranking, LM planning, adapter stacks, stems, LRC, recipes, and DAW delivery.
Music ▸ Analyze adds standalone ACE-Step understanding with structured results;
Music ▸ Transcribe adds MuScriptor transcription with an embedded MIDI piano
roll; Music ▸ Separate shares the restoration surface with Audio. Music ▸
Realtime is the Magenta RT2 session: a transport with the live clock, the
recording's waveform, Prompt A/B steering with a blend slider, temperature,
top-k, and guidance sent over the CLI's stdin protocol as you release each
control, the session log, and a job bar with Cancel and Log; it re-attaches to a
running session when you navigate back to it. Music ▸ Train is the shared
LoRA/LoKr trainer with dataset audio previews and live loss events. The resident
ACE-Step server's health and lifecycle live under Server ▸ Music server.

**Sound** ▸ Generate and Video Foley produce effects; Condition, Encode, Decode,
and Score cover conditioning, AE encode and decode, CLAP scoring, waveform
review, NPY metadata, and durable artifacts.

**Voice** ▸ Speak, Clone, and Voices host styled or cloned synthesis, reusable
profiles, reference recording, streaming feedback, and A/B playback.

**3D** is the domain for TripoSR, native TRELLIS.2 PBR reconstruction, and
ordered 4- and 6-view InstantMesh. It has engine-specific controls, immutable
output directories, embedded orbitable Quick Look models, manifest statistics,
and the shared progress and Library lifecycle. It runs the `image reconstruct-3d`
family; the `vision image-to-3d` aliases stay CLI-only rather than being
duplicated under Vision.

**Chat** covers native and MLX chat and code with typed text/JSON response
format, reasoning policy, context and KV controls, LoRA application, tool
permissions, and preflight. Chat ▸ Train hosts the text trainer rather than a
generic form. Laguna XS and Inkling-Small are explicit model families; Inkling
reasoning effort is available in chat and training, and omitted target modules
preserve the runtime's full attention, MLP, expert, shared-outer, and
unembedding training defaults.

**Vision** covers the whole VLM and VFX family: multi-image captioning,
LightOn/GLM/Infinity OCR, grounding, text/box/point segmentation and tracking,
camera capture, Buffalo-L face analysis, native pose and optical flow, video
depth, MoGe geometry, and DA3 ordered multiview reconstruction. Read, Find,
Segment, and Track are the Analyze tasks; Depth, Pose, Faces, Flow, Geometry,
and Live host the lab form that renders face and pose overlays and dense optical
flow vectors, plays live tracking and depth review video, embeds geometry point
clouds, and preserves every JSON, EXR, mask, camera, and 3D sidecar as a durable
Library artifact. Coordinates stay typed, ordered CLI arguments; machine-readable
results and mask directories use explicit output pickers.

**Audio** ▸ Transcribe is the Analyze task over `speech transcribe`. Who Spoke
is native Sortformer diarization with JSON and RTTM timelines and
segment-tuning controls. Enhance and Separate are the restoration surface for
native AP-BWE and UniverSR enhancement plus ViperX two-stem, four-stem,
dereverb, and denoise RoFormer workflows, with model-specific compute and chunk
controls, source and output previews, and every generated stem kept in the
Library. Audio ▸ Live is the live lane over `speech listen`: it enumerates
capture devices through the CLI, streams partial transcripts as the recognizer
emits them, and owns the child process directly so the operator can stop it,
then copy or save the transcript. The packaged app and embedded CLI carry the
microphone usage description and audio-input entitlement those capture paths
require.

**Text** ▸ Embeddings adds vector norms and cosine-similarity inspection;
Text ▸ Anonymize shows original and protected PII spans.

**Earth** is native Earth-observation inference, with Flood, Fire, TESSERA, and
OlmoEarth tasks. It covers TerraMind flood and fire tile inference and the
TESSERA v2 and OlmoEarth v1.2 encoders, names the tensors each input bundle must
carry before a run rather than after it, exposes engine-specific controls
(TESSERA output dimensions; OlmoEarth patch size, ground sample distance, and
space-time tokens), and preserves every produced safetensors file as a durable
Library artifact.

**Models ▸ Installed** is a list-and-detail page. The content header's subtitle
reports the real inventory ("92 installed · 48 GB on this Mac", from `model list`
and `model storage`). The 320pt list shows installed models plus any model being
pulled, with a family chip row (Image, Chat, Vision, …) and a status dot: green
installed, accent pulling, yellow when the CLI reports the model as unsupported
on this Mac. Pull… opens a sheet of the models that are not installed yet; a
pull keeps its explicit third-party terms acceptance, live CLI output, and
cancellation with resumable partials. The detail column shows the model's facts
(store, source, last used and run count from the Library, manifest verification
from `model info`), a Health panel (latest quality gate and manifest audit, with
Run gate and Benchmark… routing to those tasks), a Performance panel (last run
length, unified-memory needs, latest benchmark), the adapters whose base model it
is (Use in `<domain>` applies one to the composer, Train new… opens the
trainer), and the runtime-settings editor and raw `model info` output under two
folds. Rows whose data the CLI or Library does not have are omitted rather than
faked. A job bar at the page bottom reports a pull, MiniMax-H3 optimize or
rebuild, or storage clean-up in flight (composer pulls arrive through their
Library row) with Cancel and Log. Reveal, Remove…, refresh, opening the store,
and clean-up stay on the page.

**Models ▸ Locations** is the store editor over `model location`. It shows the
writable store, read-only search roots, and explicit per-model bindings with
live availability, adds roots and bindings through a directory picker, reveals
any of them in Finder, and confirms before removing a root or a binding — so a
model kept on an external volume is registered without leaving the app.

**Models ▸ Health** is the manifest audit and quality gate. Manifest audit is a
structured dry run, repair requires confirmation and writes only missing known
manifests, and installed-model correctness and performance gates run as durable
Library jobs with JSON reports. Successful downloads refresh the inventory in
place.

**Models ▸ Benchmarks** runs the complete `model benchmark` family as durable
Library jobs: the fused Mere Lite and Mere Comprehensive suites, the chat, code,
and vision-language slices, tool-call and tool-continuation evaluations, the
Gemma4 KV and MTP and Qwen3.6 MTP decode comparisons, Laguna DFlash, API
workload replay, and fixture hashing. Each run prints its JSON report to the
Library row rather than to a file — the benchmarks take no destination option —
except the vision-language slice, whose `--output-dir` collects the external
lmms-eval run.

**Models ▸ Adapters** lists adapter catalogs and local adapters and applies one
to a domain's composer or opens its trainer.

**Server ▸ Serving** is one operational page over the local API and the resident
model lanes: preflight, app-owned start, stop and restart, external-server
reconnection, LAN and auth safety, text and sidecar residency, load/unload and
runtime policy, unified-memory, process-CPU, Metal and thermal telemetry,
observed request and cache/batching traffic, typed Pi readiness, install,
configure and session actions, copyable client setup, and sanitized lifecycle
activity. A Vision Grounding section gives `vision serve` the same lifecycle as
the API server — preflight, start, stop, restart, an honest reading of who owns
the process, a loopback-exposure warning when the endpoint would bind beyond
localhost without a key, and the live server log. It polls the authenticated
`/runtime/status` contract and tolerates older payloads with missing additive
fields. App-owned servers and agent sessions stay durable Library runs. Open
WebUI and `world serve` are catalog templates without a section on this page;
they run from the Command Console.

**Runs** is the domain over the public `executor` and `run` contracts. It
discovers local durable reports, lists Relay jobs, polls typed inspection state,
shows artifact inventories, reveals local runs, and exposes verified fetch,
cancellation, and immutable Relay retry. Client-side Relay profile setup and
device sign-in also go through the CLI; Studio streams the approval URL but
never handles the credential itself.

**Plugins** consumes the CLI's enriched plugin snapshot, shows installed path,
version, and manifest verification, offers channel selection and copyable pinned
install commands, confirms install or update, runs the plugin's fixed doctor
verb, and rolls back to a retained signed bundle behind a confirmation. Plugin
implementations stay out of process.

Hugging Face tokens, API keys, and the Open WebUI admin password cross the
process boundary through environment variables (`MERERUN_API_KEY` and its
siblings) instead of appearing in argv, in both the typed surfaces and the
console.

## Product boundaries

Relay remains the control plane for node identity, placement, scheduling policy,
model distribution, worker lifecycle, and fleet telemetry; Studio owns the
creator's run inbox and links to the Relay console instead of copying those
schemas or controls. Diorama is the separate first-class Worlds app and owns
durable world projects, navigation, exploration, saved routes, review, and
`.diorama` bundles; Studio owns only the typed local `world serve` runtime
endpoint, authentication, model selection, status, and the handoff to
`https://diorama.mere.run`. Graph Studio owns workflow authoring and execution.
Each of these is recorded as a named exemption in `contractExemptCommandIDs`
with the reason it stays CLI-only.

## Coverage and tests

- `StudioUITests/StudioTypesTests.testEverySharedCLICapabilityHasAnAppOwnedSurface`
  asserts set equality between the app's capability IDs (every template plus six
  app-owned utilities: `guide` and the five `config` commands) and the
  contract's, so a newly cataloged CLI command cannot ship without a macOS path,
  and an app template cannot name a capability the contract does not declare.
- `Tests/MereRunCLITests/CapabilityCatalogTests.everyPublicCLICommandIsCatalogedOrExplicitlyExempt`
  walks the CLI command tree itself and requires every public leaf command to be
  either cataloged or listed in `contractExemptCommandIDs` with the reason it
  stays CLI-only. It also rejects a stale exemption, so the list cannot rot.
- `StudioUITests/NavigationModelTests.testEveryCommandTemplateMapsToADomain`
  asserts that every
  command template files into exactly one domain and that every domain owns at
  least one command. There is no assertion that a capability maps to a single
  *task*; the Command Console is what guarantees every capability a home.
- `StudioKitTests/CommandContractGuardTests` builds argv from a sweep of maximal
  and variant drafts for every local template and asserts that the subcommand
  path matches the contract and that every emitted flag is one the contract
  declares.
- `StudioKitTests/CommandFlagsGenerationTests` regenerates
  `StudioKit/Catalog/CommandFlags.swift` from the contract and diffs it against
  the committed file.
- `StudioKitTests/CommandArgumentGoldenTests` and `CommandDefaultDraftTests` pin
  every template's argv and starting draft against recorded fixtures, so a
  refactor that changes a command line has to say so.

`StudioUITests/StudioSnapshotTests` renders the shell for visual review without
driving the live app: every domain at its default task at 1440×820 in light and dark, plus
the Settings content, and fidelity renders at the 1440×900 mockup size for
comparing against the design boards — the Command Console on `image generate`,
the Main board (Image ▸ Generate with a finished generation of two in-test
images, a run held open by the process seam mid-denoise, a queued run behind a
concurrent model pull, and the inspector open with two changed settings; then
the same feed with the Command view column), the Library column in list, grid,
mixed-kind, and batch states, the Activity popover over those jobs and idle, the
composer with the boards' sample prompt and an in-test image attached
(Image ▸ Generate and Vision ▸ Find), the Analyze board (Vision ▸ Find over a
1024×1024 in-test image with a seeded `vision ground` document, and
Audio ▸ Transcribe with a synthesized recording and a timestamped transcript),
Chat with the boards' sample thread, Music ▸ Realtime mid-session (a run the
process seam holds open, fed the CLI's frame progress and steering echoes, with
its recording synthesized on disk), Video ▸ Subjects at each stage of a seeded
three-subject project, and Models ▸ Installed with a scripted model inventory
(`model list`, `capabilities`, `storage`, `info`, `runtime get`, and
`adapter list` answered from fixtures, plus Library rows for usage, a quality
gate, a benchmark, and a running composer pull). It is skipped unless
`MERERUN_STUDIO_SNAPSHOT_DIR` names a directory, so CI and a plain `swift test`
never render anything:

```
MERERUN_STUDIO_SNAPSHOT_DIR=/tmp/shell-shots swift test --filter StudioSnapshotTests
```

`StudioUITests/StudioSnapshotRenderer` hosts each view in an `NSWindow` that is
never ordered
on screen and captures it with `cacheDisplay`, so nothing appears on the Mac
running it. The controller uses a process runner that refuses every launch
(answering only the sidebar's status probe; for the fidelity renders it holds
the generation, pull, or session launches open and answers the scripted
inventory and readiness commands) and the Library is a temporary `library.json`
seeded with fixture rows, so no CLI process starts and the user's Library is
never read or written. Two fidelity gaps are deliberate: macOS 26 glass and
scroll-edge effects only composite on screen, so the renderer lifts glass
content out and hides those effects (the sidebar draws as a plain view), and the
window keeps an opaque title bar because a transparent one blanks every
offscreen `ScrollView`.

The harness renders reliably but is not yet run-to-run reproducible: six boards
— the composer, Realtime, Activity, Models, Plugins, and Subjects — draw elapsed
time and relative dates, so re-rendering the same commit changes about twenty of
the sixty-seven shots. Treat it as a tool for looking at a surface, not as a
comparison gate; it cannot tell a real regression from the clock until those
strings are frozen.

## Packaging, updates, and support

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
