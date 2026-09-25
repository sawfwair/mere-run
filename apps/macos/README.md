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
  `MereRunApp`, the app delegate, the menu bar commands, the menu bar extra, the
  Settings scene, and the Sparkle wiring. Everything else it does, it does by composing the two
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

Fifty-six tasks use six surface archetypes:

- Twelve **prompt tasks** back a `StudioMode` and render the composer, a canvas,
  and the Library column: Image ▸ Generate, Video ▸ Generate, Music ▸ Compose,
  Sound ▸ Generate, Voice ▸ Speak, Chat ▸ Chat, Chat ▸ Code, Vision ▸ Read,
  Find, Segment, Track, and Audio ▸ Transcribe.
- Five of those twelve — Vision ▸ Read, Find, Segment, Track and
  Audio ▸ Transcribe — are input-first, so their canvas is the **Analyze**
  surface rather than the generation feed. Chat and Code get the **Converse**
  surface and a thread list in place of the Library.
- Contract-backed Generate and Analyze tasks use the shared task
  workspace (`StudioUI/StudioTaskWorkspace.swift`). They cover Sound's Foley,
  Condition, Encode, Decode, and Score; Music's Analyze, Transcribe, and Separate;
  Vision's Depth, Pose, Faces, Flow, and Geometry; Audio's Who Spoke, Enhance,
  and Separate; Text's Embeddings and Anonymize; Image's Datasets; 3D's From
  image; and all four Earth tasks.
- The remaining tasks use their own Session, Project, or Manage surface, except Text ▸
  Decisions and Text ▸ Classify, which keep their editors and result panes. These include the
  three Train projects, Video ▸ Subjects, Music ▸ Realtime, Audio ▸ Live,
  Vision ▸ Live, Voice ▸ Voices, Models, Server, Runs, and Plugins.

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
- `StudioKit/StudioPromptTaskController.swift` and its extensions: active prompt
  state, task and conversation transitions, and run preparation. The shell
  delegates these operations and retains navigation, layout, focus, and dialogs.
- `StudioKit/StudioTaskSessions.swift`: inactive drafts, selection memory,
  Command overrides, and persistence. See
  [prompt workspace ownership](../../docs/internals/studio-prompt-workspace.md).
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
`StudioLibraryPresentation.swift` and `StudioLibraryPanel.swift`,
`StudioTaskSchema.swift` and `StudioTaskComposer.swift` / `StudioTaskInspector.swift`.
That is what makes a surface's rules testable without rendering it.

Every task declares a **surface archetype** (`StudioKit/StudioArchetypes.swift`:
Generate, Converse, Analyze, Session, Project, Manage), with the words and glyph
its empty surface shows (`StudioTaskPresentation`). Mode-less Generate and
Analyze tasks use one **shared task workspace**
(`StudioUI/StudioTaskWorkspace.swift`): the archetype's canvas over
`StudioTaskComposer`, with `StudioTaskInspector` in the inspector column. Its
draft is a `StudioTaskDraft` (`StudioKit/StudioTaskDraft.swift`): the chosen
template plus the same per-flag `StudioConsoleDraft` the Command view edits, so
the well, the chips, the inspector, the Command view, Library restoration, and
the argv read one value. The task's other variants are parked in it as they were
left, so switching back — from the variant chip or by picking another variant's
Library row — restores that variant's views, second picture, or cameras. The well's slots, the chips, and the inspector sections
come from the template's contract (`StudioKit/StudioTaskSchema.swift`); the
destination is filled at submit time by
`StudioOutputLocation.destination(for:)`; the prompt controller, shared
workspace, and task-specific pages submit runs through
`StudioKit/StudioTaskRunner.swift`. Bespoke result views
live under `StudioUI/Renderers/` and register by `(view, document)` in
`StudioResultRenderers` (a finished feed card asks the same registry for a
rendering in place of, or under, its output grid); a task that can say more
about its input file than its name registers an input view the same way in
`StudioInputRenderers`; the Session pages share their transport chrome from
`StudioUI/StudioSessionControls.swift`. Sound ▸ Video Foley and Condition and 3D ▸
From image (Generate); Sound ▸ Encode, Decode, and Score, Music ▸ Analyze and
Transcribe, Vision ▸ Depth, Pose, Faces, Flow, and Geometry, Audio ▸ Who Spoke,
Enhance, and Separate, Text ▸ Embeddings and Anonymize, Image ▸ Datasets, and
the four Earth tasks (Analyze) render on it; Vision ▸ Live and Audio ▸ Live
(Session) and Voice ▸ Voices (Manage) render their own pages over the same task
draft. The SFX Lab, Music Tools, Vision Lab, Voice, Audio Tools, Utility Lab,
3D Creation, and Geo Lab pages they replaced are gone. Text ▸ Decisions keeps
its question editor; Video ▸ Subjects, Music ▸ Realtime, Models, Server, Runs,
and Plugins keep task-specific surfaces.

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

The footer pill reads "Ready · N models" once the status probe answers (the
probe runs when the model inventory or the CLI settings change, not on a timer),
"Serving" while the API server answers (the reading the Server page
and the menu bar extra take from `StudioLocalServer`, not the slower status
poll), and "CLI not responding" (in red) if the probe never answers within six
seconds, with "N running" on a second
line while jobs are in flight. It opens the **Activity popover**
(`StudioUI/StudioActivity.swift`), a 340pt panel the shell draws over the window from the
bottom-left: one row per running or queued job in the inference and utility
lanes (never a probe) with its progress and a stop control, over the app↔CLI
version handshake and a link into the Server page. A row for one of Studio's own
CLI reads names the work ("System · Checking models"), never the subcommand.
With nothing running the same panel shows the local server and the models root;
the resolved CLI path is the footer's tooltip. When `status --json` reports
`modelLocationIssues` (a registered drive that did not answer in time, usually
because macOS is waiting on its removable- or network-volume access prompt, or
one macOS denied), the pill's dot turns yellow and the panel adds a row naming
the drive that opens Privacy & Security ▸ Files & Folders. It
reads the `JobStore` directly — the lanes for which rows exist, each `Job` for
its own progress — so nothing about the work in flight is mirrored on the
controller.

The **menu bar extra** (`StudioUI/StudioMenuBarPanel.swift`, on by default,
switched off in Settings ▸ Server) keeps the API server within reach with no
Studio window open. Its glyph is the app icon's Caveat "m." with the period as a
power light: filled while a server answers, hollow while none does. The panel
shows the server's state and address with Start, Stop, and copy-endpoint
controls and why it stopped when it exits on its own; a Stop row for a vision,
music, or world server Studio started while it runs; **This Mac** — CPU and the
server's decode rate as tiles with two minutes of sparkline history, and one
memory bar splitting the server's footprint from everything else and free, with
a tinted thermal warning when the Mac is throttling (`StudioMachineMonitor` reads
Mach host statistics every two seconds; the decode rate is the change in
generated tokens between endpoint polls); the resident text models (each with
Unload), sidecars, and a Load menu of the server's other text models; the same
job rows as the Activity popover, and Open Studio, Server Settings…, and Quit.
It reads the same `StudioLocalServer` the Server page drives, so the two never
disagree. Quit asks first while a server Studio started or an inference job is
still running, since every child process ends with the app; the app delegate owns
the session, so that holds with no Studio window. With the extra in place,
mere.run leaves the Dock while no window is open and returns when one opens.
Settings ▸ Server ▸ Menu bar and startup turns off the extra or the Dock hiding,
starts the API server when mere.run opens (unless something already answers on
the endpoint), and registers mere.run as a login item (`SMAppService`; only the
packaged app can register). A server Studio started that exits on its own posts
a notification while mere.run is in the background.

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

The Library column appears on the prompt tasks and on tasks that have moved onto
the shared task workspace (`StudioTask.showsPromptChrome`); every other task —
Subjects, Realtime, Models, Train, and Decisions — takes the full content width even
inside a Create domain. Chat and Code fill that column with
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
files to the Trash. A batch of exactly two finished image runs adds **Compare**
to the bar and the context menu, which opens the older run in the result
workspace with the newer beside it. Search matches a run's title, kind, prompt,
model (the name the app shows or the exact id, from the thread, the recorded
draft, or a legacy row's `--model` argument); a whole status word ("failed",
"running") narrows to that status and the rest of the query must still match,
so "failed harbor" is the failed harbor runs, and a fragment like "ed" matches
nothing.
Filtering and day-grouping live in `StudioLibraryPresenter`, so both are
testable without a view. The view mode, kind, and favorites filter persist per
window under `studio.libraryView`, `studio.libraryKind`, and
`studio.libraryFavorites`.

A row's context menu offers **Use these settings** (also on a finished card's
icon row and beside Retry on a failed card): the run's task opens with its
recorded prompt, model, and options in the composer, ready to tweak and run
again. `StudioLibraryDraftRestoration` reads the recorded command back through
the same contract bindings the Command view uses, replaces the task's parked
draft and any Command view override, and records the run as the draft's parent.
An input-first command's file argument lands first, through `replaceInput`, so
the `--box`, `--point`, `--init-frame`, and `--end-frame` that follow it come
back onto the picture rather than being cleared with the input change.
Only options a page control binds come back; a Command view extra with no
control (an option the composer never shows) is not restored, and a row whose
command the composer does not build (an upscale or edit run from the Console)
does not offer the action at all (`StudioLibraryDraftRestoration.canRestore`).
Run again and Edit command… stay for an exact rerun or a raw edit.

Each task retains its full draft and selected run through `StudioTaskSessions`.
Prompt modes preserve model, seed, dimensions, attachments, and sampling values;
task-specific forms retain their typed settings. The versioned JSON store excludes
launch credentials and preserves unreadable files. Prompt edits update task
sessions synchronously. `studio.drafts` remains a migration source for earlier
prompt-only scene state; importing it preserves unvisited tasks and gives full
session drafts precedence.

Menus follow macOS convention: File ▸ New Chat (⌘N) and Import Receipt…; View ▸
Show Library (⌥⌘L), Show Inspector (⌥⌘I), Show Command View (⌥⌘C), and the system sidebar toggle; Go ▸ every domain
(⌘1–⌘9, then ⌥⌘1…) plus the current domain's tasks; Run ▸ Run (⌘↩), Stop (⌘.),
Open Last Output (⇧⌘O), and Reveal Last Output in Finder (⇧⌘R), acting on the
current composer; Window ▸ Open Studio and Command Console (⇧⌘C); Help ▸
mere.run Guide (⌘?), the mere.run link, and Export Diagnostics…. ⌥⌘C is always
the task's Command view, disabled on the few tasks without one. Settings has
General, Models, Server, and Advanced tabs; its path settings are pickers with
Choose…, Reveal, and Reset, and the Server tab's endpoint and key
apply on Apply or Return, not per keystroke. First run shows the Image empty state with its "Get the model"
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
prompt, a **chip strip** shows up to four contract essentials (size, length,
duration, steps, seed, resolution, task, voice mode, thinking) as menus with
popover editors for custom values; some modes show only the model chip. The **model chip** is the only
model control: it lists `model list` rows filtered to the mode's category,
installed first, with "Auto" for the mode's default. A fresh draft starts on the
model the user made the task's default in Models ▸ Installed ("Use for Chat by
default", kept per task in `StudioTaskSessions`), else the CLI's recommendation
for Chat and Code, else the template default. Every surface names a model
through `StudioModelNaming`: the inventory's title when `StudioModelStore` has
published one (`StudioModelTitles`, set in the environment by each window root
and passed to presenters explicitly), else a name formatted from the id, with
the exact id in the tooltip. The chips and the inspector bind the same `StudioDraft`, so a value
changed in one shows in the other. ⌘↩ runs; while a conversation turn streams,
the send circle becomes Stop.

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
failed run leads with the line the CLI marked as the error (else the last
meaningful stderr line; a `Searched:` list of model locations is never it), keeps the log behind
"Show log", and offers Retry. Validation errors ("Prompt is required.") render
as a banner under the composer; readiness (missing model, missing CLI) is a card
at the bottom of the feed, so it never hides earlier work. The card speaks
plainly ("LTX-2 Fast isn't on this Mac yet.") and carries the next step for its
state: **Get the model** with the pull's own progress (a publisher's terms are
acknowledged in a sheet first, the same one the composer uses), beside
**Choose another model**, the composer's own picker; the picker with **Open in
Models** when the Mac cannot run the model; **Check again** and the picker when
the check itself failed, with the CLI's last line kept as a muted detail so a
wrong model location stays diagnosable; and **Check the model** before the first
check, which has its own neutral state (`ModelReadinessState.notChecked`,
`StudioReadinessActions`). A run that failed because its model is not on this
Mac says so plainly and offers **Get the model** on its card instead of the
CLI's error line. Completion never moves the Library selection; a
result that finishes off-screen shows a "New result ↓" pill. Picking a Library
row scrolls to its card and outlines it briefly.

The input-first tasks render the **Analyze canvas**
(`StudioUI/StudioAnalyzeCanvas.swift`, `StudioUI/StudioAnalyzeViews.swift`) instead of the feed,
because the answer belongs beside the thing it is about rather than in a stream.
It uses up to 940pt of canvas width: an input strip naming the attached file with its
dimensions or duration, a Replace button that writes the same composer well, and
a view switch whose segments come from the task's own result kind (Boxes /
Masks / JSON for Find and Segment, Video / JSON for Track, Transcript /
Timeline / JSON for Transcribe); below it the input rendered large on the left —
the image with the result drawn over it, a video with its scrubber and
per-object track spans, audio with the waveform player — and a 360pt result
column on the right holding what the model found, the contextual next steps, and
the prompt it ran with. When the Library leaves less room, the result stacks
below the input and the view switch moves under the input strip. Results are
read from the documents the CLI actually writes
(`StudioKit/StudioAnalyzeResults.swift`): `vision ground`'s normalized boxes,
`vision segment`'s pixel boxes with their mask PNGs, `vision track`'s per-frame
detections, `speech diarize`'s speaker turns, and the timestamped transcript
`speech transcribe` prints. Studio always asks for that document, passing
`--json-output` (and `--mask-output-dir` for the still tasks) beside the
annotated output. A run in flight, a queue, a failure, and readiness use the
feed's own cards above the result column, and earlier runs stay one click away
in the Library column, which also puts their input back in the composer. The
next steps open the sibling task carrying the input when the target accepts it
(Find ▸ "Segment these" keeps the picture and draws what Find found as Segment's
box prompts; "Track in video" keeps only the prompt, because Track needs a clip).

Segment and Track take their prompts on the picture rather than as typed
coordinates (`StudioUI/StudioRegionPromptEditor.swift`, the geometry and the CLI
text in `StudioKit/StudioRegionPrompts.swift`). Over the input, a drag draws a
box, a click adds a point, and Option-click adds a negative point; a Box /
Point / Negative / Clear toolbar picks what a click does. What a press starts
is one pure decision (`StudioRegionPress.press(tool:hit:optionHeld:)`): a
point, or the selected box's corner handle, always takes the press (select and
move, or resize); a box takes it only with the Box tool, so with the Point or
Negative tool a click inside a box adds the point there — inside a box is where
a refining point goes — and a drag inside it draws another box. The selected
box is solid with a white halo and corner handles, the selected point has a
ring; Delete (or Forward Delete) removes the selection and Escape clears it.
Those keys arrive through one local `NSEvent` key-down monitor
(`StudioRegionKeyMonitor`, owned by the layer whose prompt was pressed last and
installed only while it holds a selection) rather than SwiftUI focus, because a
click that starts the drawing gesture never makes the layer first responder;
`StudioRegionKeyCommand` in StudioKit decides the key and ignores every key
while a text field is being edited, so typing in the composer is never
affected. A prompt's numbered
tag flips or slides to stay on the picture. Each prompt is a VoiceOver element
("Box 1, coffee cup, 120 by 80 at 40, 30"). The prompts live on the draft
(`StudioDraft.visionRegionPrompts`) in the input's own pixels and become the
command's `--box` / `--point` values, so the composer, the Command view, and
the run all read one set (a `--box` typed in the Command view appears on the
picture through the same binding table); a drawn prompt satisfies the task's
prompt requirement, and the composer starts empty with a placeholder rather
than a prefilled prompt, so a drawing runs on its own. Replacing the input, by
any route, clears the prompts and frames drawn on the previous one. A photo is shown
upright, as the well shows it, while its prompts and the CLI's result boxes are
kept in the file's stored pixels — the space the CLI decodes without the EXIF
transform — with `StudioImageOrientation` mapping between the two for all eight
orientations. Track shows its clip as a frame scrubber
(`StudioUI/StudioTrackFrameEditor.swift`, frames decoded with
`AVAssetImageGenerator`): "Prompts here" makes the frame in view the prompt
frame the tracker seeds on (`--init-frame`), and "End here" sets the optional
last frame (`--end-frame`); the full sentence is each button's help, and the
buttons fall back to their icons when the column is too narrow for the titles.
The CLI then tracks the whole clip from frame 0 through the end frame, not
from the prompt frame, so the range line under the scrubber reads "Prompts on
frame 10 · tracks frames 0–30" and the scrubber shades that span. The picture
itself takes the height the column has above the composer
(`StudioAnalyzeMediaLayout`), so a portrait input fits the visible area instead
of running under the composer. Once a tracked clip exists it plays in place,
with "Adjust prompts and frames" bringing the scrubber back.

`StudioKit/StudioAnalyzeSchema.swift` declares the surface — the result views and the next
steps — for every input-first task. Sound ▸ Score, Encode, and Decode render on
this canvas through the shared task workspace (Score's result is the CLAP gauge,
Encode's the `.npy` header, Decode's the decoded audio), as do Music ▸ Analyze
and Transcribe, Vision ▸ Depth, Pose, Faces, Flow, and Geometry, and Audio ▸
Who Spoke, Enhance, and Separate. Text ▸ Embeddings, Text ▸ Anonymize, and
Image ▸ Datasets render on it too: the typed text or the folder, plan file, or
nothing a Datasets variant takes on the left, and the cosine matrix, the
protected text and spans, the candidate folders (each with "Train on it"), the
run plan report, or the validation artifacts as the result panel's rows
(`StudioUI/Renderers/`). The four Earth tasks reach it through the shared task
workspace too, with a checklist of the tensors their bundle needs in the input
column. Every contract-backed input-first task renders on this canvas. Text ▸
Decisions keeps its question editor and answer pane.
A view that is about the result rather than the input — Points, Vectors, Depth,
Scene — takes the input column from a renderer registered in
`StudioUI/Renderers/StudioResultRenderers.swift` (`canvasRendering`), the same
registry the result panel asks for its rows.

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

With the thinking chip on, a reply's reasoning is kept beside the answer
(`StudioMessage.reasoning`) and shown as a collapsed **Thinking** disclosure
above it; while the model is still inside its reasoning block the disclosure
reads "Thinking…" live. A turn that failed says why on one line — the last
meaningful line of the run's stderr, or the preflight message when the run
never started (`StudioMessage.failureReason`, via `StudioFailureSummary`) — with
Retry beside it and the run's last stderr lines behind **Show log**
(`StudioMessage.logTail`). A thread whose reply was cut off when Studio closed
shows the same row. Reasoning, reason, and log are display-only: the transcript
renders only `content`, and a failed turn is never replayed at all, so none of
it reaches the next prompt. The transcript follows new output only while the
reader is at the bottom; scrolling up stops the following and a **Jump to
latest** pill brings it back. Deleting a thread from its context menu asks
first, naming the thread.

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
and the rest of their options are reached in the Command Console. Segment and
Track also bind `--box`, `--point`, `--init-frame`, and `--end-frame`, but as an
external override like the attachment well: the canvas is their editor, so the
inspector never shows them as text while a Command-view edit still flows back
into the drawing.

The **Command** panel (⌥⌘C or the header toggle) exposes the current task's
complete editable contract. It replaces the inspector and uses a 440-point
column when space permits, otherwise an overlay. Each row is headed by the
option's label with its flag beneath in small monospace, over the same typed
controls the Console draws. The preview, validation, and
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

The console opens from Window ▸ Command Console (⇧⌘C), a Library row's **Edit command…**,
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
**Compare** selects another result and links zoom and pan; batching two image
rows in the Library and choosing Compare lands in the same view with the pair
already side by side. The settings area
shows differences between the recorded commands. **Continue with…** opens a
new draft for editing, reference guidance, video, image understanding, or
segmentation. The resulting run records its parent; the original stays in Library.
**Save copy…** copies before replacing a destination and treats saving onto the
source as a no-op. Library **Save to…** uses the same replacement behavior for
one file. Saving several artifacts into a folder preserves existing files and
saves each shared source once. Missing files and copy failures are reported.
Selecting a different Library row closes the focused result; returning from a
task detour keeps focus on the same result. Closing focus returns keyboard focus
to the composer.

## Jobs, artifacts, and output

`StudioAppSession` attaches `StudioLibraryStore` to job events and owns the
serving monitor for the lifetime of the app. Views select jobs; they do not own
completion recording. Cancellation and interrupted sessions have distinct
Library statuses. Console Stop and the Run menu act on the Console's selected
job; a chat's Stop acts on that thread's turn.


`JobStore` owns every child process the app launches behind the
`MereRunProcessRunning` seam (`Process()` appears only in
`StudioKit/Jobs/ProcessRunner.swift` and the synchronous `CLIBootstrapInstaller` version
probe), with four lanes: `inference` for Studio runs (capped at two with a FIFO
queue), `utility` for the hand-built CLI reads and writes behind
`utilityCommandResult` (capped at four, FIFO), `probe` for readiness and
`status --json` probes (never queued, deduplicated by key so a repeated probe
joins the one in flight and a probe with stale Settings is superseded), and
`service` for the API, vision, and music servers Studio starts (never queued; a
server runs until it is stopped, so it holds no inference slot and takes no
console, Library row, or completion notification). A
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
(`mererun.app.outputRoot`). A task on the shared task workspace (Music ▸
Transcribe, the Vision and Audio tasks) has no path field:
`StudioOutputLocation.destination(for:)` names its output after the input in
the domain's folder when the run starts, with its sidecars beside it. Its draft
never keeps a destination Studio named: a fresh draft starts without one, and a
saved draft is read without any that sits in one of Studio's folders (under the
root, a per-media `mere.run` folder, or App Outputs). A destination typed into
the Command view elsewhere keeps its folder, with `-2`, `-3`… added while that
path exists or a submitted run holds it, so no run writes over another's. The
Command view's "Will run" shows the destination the run will write. The
task-specific pages use the same domain roots: training adapters are filed under
the domain they train for, Decisions uses `outputDirectoryURL`, and Music ▸
Realtime, Video ▸ Subjects, and the audio recorder use timestamped file names
from `specialistFile`. A task-specific or
Command view run is prepared the same way a prompt run is
(`StudioOutputLocation.preparing`): the folder is created, or the run moves to
`App Outputs` and the shell's banner says why; a path a submitted run holds is
reserved, so the next proposal in the same second steps aside. Nothing is migrated:
Library rows keep the paths
they recorded, Application Support holds metadata only, and a destination that
cannot be created sends the run back to `App Outputs` with its sidecars and one
banner saying why.

`MereRunController` is the facade views bind to. It snapshots Settings and the
CLI launch into a `JobRequest` for every lane, awaits utility and probe jobs on
behalf of their callers (readiness results are evaluated against the request
current at completion), mirrors the foreground inference job into the published
compatibility fields some management views still read, and owns
the template selection the console opens on and the persisted settings. Those
settings are `UserDefaults` keys (`mererun.app.cliPath`, `modelsRoot`,
`hubCache`, `workingDirectory`, `runtimeHost`, `runtimePort`) except the
Settings ▸ Server API key, which `StudioSecretStore` keeps as a generic
password in the login Keychain (service `run.mere.app`, account
`runtimeAPIKey`; `KeychainSecretStore` is the one Keychain owner in StudioKit).
A key an earlier version saved under `mererun.app.runtimeAPIKey` is moved into
the Keychain at launch and the defaults key is removed only after that write
succeeds; if the Keychain refuses, the key still applies for the session, the
defaults value stays for the next launch to retry, and a banner says so
(`runtimeAPIKeyStorageNotice`). A save the Keychain refuses is never written to
`UserDefaults` instead.

## Domains

**Image** covers generation and editing, LoRA training, validation, dataset
discovery, durable plans and dashboards. Image ▸ Generate includes
multi-reference editing, structured prompts, LoRA catalog IDs or local adapters,
Krea tuning, and preflight. Image ▸ Train adds dataset previews, preflight,
launch and resume, loss metrics, samples, checkpoints, and run comparison for
Krea 2 and FLUX.2 Klein; Klein's per-target ranks are rows of module suffix and
rank rather than a typed `suffix=rank` list. Image ▸ Datasets is one Analyze task
over three commands — Discover, Validate, and Run plan — picked by the
Operation chip. Discover takes a folder in the well and lists the candidate
datasets it found with their counts and problems
(`StudioKit/StudioTextDatasetResults.swift`, `StudioUI/Renderers/StudioDatasetCandidates.swift`);
"Train on it" on a row opens Image ▸ Train with that folder as the dataset.
Validate takes no input and lists the artifacts it wrote. Run plan takes a plan
file and renders the report (`StudioKit/StudioRunPlanReport.swift`,
`StudioUI/Renderers/StudioRunPlanReportView.swift`): a preflight's steps,
resolution, batch, rank, learning rate, checkpoint and preview cadence,
schedule, memory switches, dataset counts, model, and output, read from the
CLI's typed envelope, or a materialized run's files, each revealable in
Finder; Preflight is a chip and the run directory for Materialize is an
Output-section row in the inspector. The three Train pages are Project
surfaces over a task draft (`StudioKit/StudioTrainingRun.swift`,
`StudioUI/StudioTrainingView.swift`): the dataset folder and an optional resume
checkpoint are attachment wells, the base model is the shared model picker over
the trainer's inventory category, the template's options sit in the sections
the page always had (numbers the contract gives no range are typed), and the
adapter's destination is routed rather than typed — named after the dataset and
filed under the domain it trains for, with events, samples, and checkpoints
beside it. A chosen recipe decides the options it governs (`--width`, `--model`,
`--rank`, and the rest are left off the command line unless typed), and a Klein
launch sets a checkpoint and a preview every 250 steps when neither was chosen.
Runs go through the task runner, so Stop, the Library row, and the root's
Command view share the draft; the pages' saved drafts import once.

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
persistence, and durable Library jobs with a job bar for the running stage. A
subject's precise selectors and each keyframe correction are drawn with the same
region editor Segment uses (`StudioUI/StudioSubjectSelectorEditor.swift`): the
reference selector on the reference image in its own pixels, and the driving
selector or correction on the driving frame center-cropped to the plan's
width × height, which is the space `video prepare-masks` segments in; the
drawing is written back to the plan's box and point text. The
guided SCAIL-2, Cosmos3, mask-preparation, latent-export, and resident-session
commands live in the Command Console.

**Music** is a production surface, not a prompt-only wrapper: quality planning,
covers, repaint and flow edits, source and timbre-reference audio, candidate
ranking, LM planning, adapter stacks, stems, LRC, recipes, and DAW delivery.
Music ▸ Analyze and Music ▸ Transcribe are Analyze tasks on the shared task
workspace: the recording goes in the composer's well (or is dropped on the
canvas), the settings live in the inspector, and the result column shows what
the run found. Analyze reads the command's JSON as tempo, key, meter,
language, caption, and lyrics under its Analysis view, with the model's reply
and audio codes folded away when a run kept them
(`StudioUI/Renderers/StudioMusicAnalysisRenderer.swift`); Transcribe draws the
MIDI it wrote on a piano roll under Notes, with Quick Look and Reveal for the
file (`StudioUI/Renderers/StudioPianoRollRenderer.swift`), and its expected
instruments are picked in the inspector from the list the CLI prints with
`--list-instruments` (a plain field when the list cannot be read). The
transcription is named after the recording and filed by what it is — a MIDI
file under `~/Music/mere.run/Music`, a JSON or JSON Lines event list under
`~/Documents/mere.run/Music` — with its musical-context document beside it
as `<name>-context.json`; none is asked for when musical context is off.
Music ▸ Separate shares the restoration surface with Audio. Music ▸
Realtime is the Magenta RT2 session: a transport with the live clock, the
recording's waveform, Prompt A/B steering with a blend slider, temperature,
top-k, and guidance sent over the CLI's stdin protocol as you release each
control, the session log, and a job bar with Cancel and Log; it re-attaches to a
running session when you navigate back to it. Music ▸ Train is the shared
LoRA/LoKr trainer with live loss events; its dataset is a clip list
(`StudioUI/StudioMusicManifestEditor.swift`, `StudioKit/StudioMusicTrainingManifest.swift`)
rather than a hand-written manifest: add audio files or a folder of clips with
matching `.txt` captions (or drop them in), caption each clip, add lyrics where
they matter, play any clip, and see the trainer's own checks in the page's words.
Start training writes `<adapter>.dataset.jsonl` beside the adapter, one record
per line the way `music train-adapter --dataset` reads it; manifests made
elsewhere import, and the clip list exports. The clip list is the task draft's
`--dataset` editor, and ACE-Step's checkpoint root (a folder chooser), decoder,
VAE, and text-encoder folders sit under the model picker. The resident ACE-Step server's
health and lifecycle live under Server ▸ Music server.

**Sound** ▸ Generate and Video Foley produce effects, with Woosh renoise as the
model's default, one amount on a slider, or one amount per step (the task
inspector's Renoise editor; a per-step schedule that does not match the step
count is refused before the run starts). Video Foley is a Generate task on the
shared task workspace: the clip goes in the well, the prompt in the composer,
and the finished card plays the picture over the waveform it produced.
Condition (prompt to conditioning tensors, shown as their header), Encode
(audio to `.npy` latents, shown as the tensor header), Decode (latents back to
audio), and Score (the CLAP gauge over the audio) run on the same workspace, so
the Library, "Use these settings", Stop, readiness, and output routing behave
as they do for every other task.

**Voice** ▸ Speak is styled or cloned synthesis: attaching a reference recording
to its composer well switches it to clone mode, and its inspector picks a saved
voice. Voices (`StudioUI/StudioVoicesView.swift`) is the Manage surface for
those saved voices: a list, a detail that plays the reference and shows its
transcript, Delete behind a confirmation, and New voice — a name, a reference
recording in an attachment well (chosen, dropped, or recorded with Record…),
an optional transcript, and a language — run as `speech profile create`
through the task runner. Any audio attachment well offers Record… in its
context menu (`StudioUI/StudioAudioRecorder.swift`); the recording is filed
with the task's domain.

**3D** ▸ From image is a Generate task on the shared task workspace for
TripoSR, native TRELLIS.2 PBR reconstruction, and ordered 4- and 6-view
InstantMesh. The Engine chip picks the template; the well takes one picture
(or InstantMesh's ordered views), the inspector shows each engine's own
controls from the contract, and every run lands in a fresh directory under the
3D folder. Results are feed cards with the mesh in an orbitable Quick Look
tile and the manifest's vertex, triangle, and PBR voxel counts under it
(`StudioKit/StudioMeshSummary.swift`, `StudioUI/Renderers/StudioMeshSummaryRow.swift`).
InstantMesh's ordered views are reordered in the inspector, and its optional
calibrated cameras are edited per view there
(`StudioUI/StudioInstantMeshCameraEditor.swift`,
`StudioKit/StudioCameraDocuments.swift`) — a 3 × 4 camera-to-world pose and
`fx, fy, cx, cy` — checked as the CLI checks them, saved as a content-named
file the draft's `--cameras` points at, and imported and exported as files.
An InstantMesh run without four or six views, or with a camera file that does
not match them, is refused with the reason before anything is created
(`StudioKit/StudioCommandChecks.swift`). It runs the `image reconstruct-3d`
family; the `vision image-to-3d` aliases stay CLI-only rather than being
duplicated under Vision.

**Chat** covers native and MLX chat and code with typed text/JSON response
format, reasoning policy, context and KV controls, LoRA application, tool
permissions, and preflight. Chat ▸ Train hosts the text trainer over its task
draft: the JSONL dataset, evaluation prompts, and a resume checkpoint are
attachment wells, the base model comes from the text-chat inventory, and a
local model path sits under it. Laguna XS and Inkling-Small are explicit model families; Inkling
reasoning effort is available in chat and training, and omitted target modules
preserve the runtime's full attention, MLP, expert, shared-outer, and
unembedding training defaults.

**Vision** covers the whole VLM and VFX family: multi-image captioning,
LightOn/GLM/Infinity OCR, grounding, text/box/point segmentation and tracking,
camera capture, Buffalo-L face analysis, native pose and optical flow, still
(Marigold V2) and video depth, MoGe geometry, and DA3 ordered multiview
reconstruction. Read, Find, Segment, Track, Depth, Pose, Faces, Flow, and
Geometry are Analyze tasks. The last five run on the shared task workspace
(`StudioUI/StudioTaskWorkspace.swift`): the picture (or two, or an ordered set)
in the composer's well, the variant as a chip — Faces: Detect, Embed, Compare,
Batch; Depth: still, video; Geometry: single, multi-view — the contract's options
in the inspector, and one `StudioTaskDraft` behind the well, the chips, the
Command view, and the argv. Their renderers live under `StudioUI/Renderers/`:
Faces draws boxes, then the five landmarks per face in the Points view and reads
the embedding, comparison, and batch documents by name; Pose draws its landmarks
over the picture with one row per subject; Flow draws the field as
direction-colored vectors with its motion statistics; Depth shows the preview
PNG (or the review clip) from the run's directory; Geometry embeds Quick Look
over the point cloud with the depth and normal previews in a strip. Every JSON,
EXR, mask, camera, and 3D sidecar stays a durable Library artifact, and the
destination is named by routing rather than a path field. Faces ▸ Embed and
Compare choose their face by clicking it on the picture in the inspector once a
Detect run has drawn boxes on that image (a number field remains for a picture
nobody has detected faces in); Compare offers one picker per picture. Geometry's
multi-view variant edits optional calibrated cameras per view in the inspector —
image size, normalized focal length and center, and a world-to-camera rotation
and translation — with the CLI's own checks (positive size and focal length, a
proper rotation, and an image size equal to the image's decoded size, which new
cameras take from the image), writing its saved draft file into `--cameras`
whenever cameras are on; a camera file that does not match the views, or will
not read, refuses the run with the reason rather than letting the model estimate
cameras (`StudioCommandChecks`). A camera file the draft names from elsewhere — a
restored run, the Command view — is read into the editor, not overwritten. At
submit the runner copies the file beside
the run's output directory as `<folder>.cameras.json` and points `--cameras`
there, so the run's folder is self-contained. Camera files import and export.
Faces are numbered from one everywhere the picture is read; the flag counts
from zero.

**Vision ▸ Live** is a Session page (`StudioUI/StudioLiveTrackSession.swift`) over
`vision track-live`: Start/Stop, the camera (this Mac's cameras by name, in the
order the CLI numbers them) and the model as chips in the transport row, the
things to track one per line, the capture's progress and log while it runs, and
the annotated clip with its track spans once it lands in the Library; the
settings column holds the rest of the contract. Runs go through the task runner,
which keeps the camera-access prompt in front of the CLI; Stop while macOS is
still asking cancels the run, so the capture does not start once access is
granted. Stop ends the capture without a clip; the session ends on its own after
the duration.

**Audio** ▸ Transcribe, Who Spoke, Enhance, and Separate are Analyze tasks on
the shared task workspace. Who Spoke runs `speech diarize` with native
Sortformer or Nemotron 3 (the model chip), the output format and the Nemotron
input buffer as chips, and the threshold, minimum segment, and merge gap in
the inspector; a JSON or RTTM timeline is drawn as one lane per speaker over
the recording (`StudioUI/Renderers/StudioSpeakerTimeline.swift`) and as the
panel's turn rows, with Save timeline…. Enhance runs `audio enhance` (AP-BWE
or UniverSR from the model chip, compute as a chip, the UniverSR bandwidth,
ODE, guidance, chunk, and seed controls in the inspector) and plays the
enhanced file; Separate — under Audio and under Music — runs `music separate`
(ViperX two-stem, four-stem, dereverb, and denoise from the model chip) and
lists every stem with its own player
(`StudioUI/Renderers/StudioStemsList.swift`), read from the manifest the CLI
writes. Audio ▸ Live (`StudioUI/StudioLiveListenSession.swift`) is the Session
surface over `speech listen` and `speech diarize-live`: Start submits the task
draft through the task runner as an inference job with a Library row, the
transport row carries the operation, microphone, options, and model chips,
events stream into the transcript or the speaker activity as they arrive, and
Stop — the page's, the Library's, or the menu's ⌘. — interrupts the CLI the way
Ctrl-C does (terminating it if it does not finish). The runner adds the
session's `--jsonl --quiet` at launch, so a session the Command view runs streams
into the page too, and the page adopts it as it starts. The session belongs to
the controller, so leaving the page loses nothing; when it ends, its text is written to the Audio folder and becomes the
Library row's artifact, so the row reads like Transcribe's. The packaged
app and embedded CLI carry the microphone usage description and audio-input
entitlement those capture paths require.

**Text** ▸ Embeddings and Text ▸ Anonymize are Analyze tasks whose input is
typed on the canvas: Embeddings takes one text per line and shows each vector's
norm with the cosine similarity of every pair
(`StudioUI/Renderers/StudioEmbeddingsMatrix.swift`); Anonymize takes the paste
as one text and shows it beside the protected text with every span the filter
marked (`StudioUI/Renderers/StudioAnonymizationSpans.swift`), or the protected
text alone in its Text view. Both file their JSON under Text. Text ▸ Decisions
(`StudioUI/StudioLayaDecisionView.swift`, `StudioKit/StudioDecisions.swift`) builds
the Laya request instead of asking for one: the text to judge, then ordered
choice, score (levels lowest first), and yes-or-no questions, with optional
option descriptions and editable ids derived from each question. It checks the
request the way `text decide` does, writes it beside the run's output in the Text
folder, and reads the result back as each question's answer, probabilities, and
confidence, with what was cut to fit; **Check fit** runs `--preflight` and shows
each question's token fit. The handbook example loads with one click, and request
JSON imports and exports.

Text ▸ Classify (`StudioUI/StudioGLiNERClassificationView.swift`,
`StudioKit/StudioClassifications.swift`) builds the GLiNER2.5 Decide request
from text and ordered label tasks. Each task can carry descriptions, an optional
prompt, and a multi-label threshold. The result pane shows selected labels and
all scores in the task's label order. **Check fit** reports the input token
count. Like Decisions, Classify saves its draft for the Command panel and can
import or export request JSON.

Text ▸ Extract (`StudioUI/StudioGLiNERExtractionView.swift`,
`StudioKit/StudioExtractions.swift`) edits GLiNER entity names, relation
names, structured fields, and joint classification tasks. **Check fit** reports
the schema and token count. **Extract** shows span offsets and confidence,
relation pairs, and structured records. The page imports and exports the same
JSON request used by `mere.run text extract`. Both GLiNER pages offer
overlapping chunks for long text.

**Earth** is native Earth-observation inference, with Flood, Fire, TESSERA, and
OlmoEarth tasks — TerraMind flood and fire tile inference and the TESSERA v2
and OlmoEarth v1.2 encoders — each an Analyze task on the shared task
workspace. The well takes the safetensors tile bundle; the input column reads
the bundle's header (never the tensors behind it) against the tensors the
command requires and ticks each one off with its dtype and shape, so a missing
`DEM` or an unpaired `S1_ASC` is caught before the run in the words the
command would refuse it with, and an empty well names what a bundle must carry
(`StudioKit/StudioEarthInputRequirement.swift`,
`StudioUI/Renderers/StudioEarthInputChecklist.swift`, registered by template in
`StudioUI/Renderers/StudioInputRenderers.swift`). The result panel shows the
written safetensors file's header — the logits or embedding tensor, its shape,
and the writer's metadata — with the command's JSON as the second view. The
inspector holds the model, Preflight, TESSERA's output dimensions as the picker
of the widths the command accepts (`StudioUI/StudioEarthControls.swift`), and
OlmoEarth's patch size, ground sample distance, and space-time tokens. Outputs
are named after the bundle under the Earth folder, every run is a Library row,
and the Geo Lab page's saved drafts seed the task drafts once.

**Models ▸ Installed** is a list-and-detail page. In a narrow window, select a
model to open its full-width details. Choose **All models** or press Escape to
return to the list. The content header's subtitle
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
folds. The header's tags name the tasks the model runs by default ("Default for
Image" for a template default, "Your default for Code" for a choice), and the
More menu offers **Use for `<task>` by default** for each task whose model chip
lists the model; turning one on moves that task's composer onto the model at
once (a model the user picked by hand in a parked draft stays), turning it off
returns the task to the built-in default, and a choice whose model has since
left the inventory is ignored rather than left pointing at nothing. Rows whose data
the CLI or Library does not have are omitted rather than
faked. A job bar at the page bottom reports a pull, MiniMax-H3 optimize or
rebuild, or storage clean-up in flight with Cancel and Log. Downloads started
from either the composer or Models use the same job, including queued downloads.
You can leave Models and return to its progress and cancellation controls.
After failure or cancellation, use **Download log** for details and **Pull…**
to retry. The latest failed download stays listed so you can also retry from its
details.
Pulling the same model again reuses its active download. After a download
finishes, Studio refreshes inventory and readiness for your current selection.
The composer and Models share one inventory; a failed refresh reports an error
and keeps the previous inventory for that location. Reveal, Remove…, refresh, opening the store,
and clean-up stay on the page.

**Models ▸ Locations** is the store editor over `model location`. It shows the
writable store, read-only search roots, and explicit per-model bindings with
live availability, adds roots and bindings through a directory picker (the
bound model is chosen from the inventory), reveals
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

The **Server** domain has one task per resident server: Serving (the API),
Music server, and Vision server.

**Server ▸ Serving** is one operational page over the local API and the resident
model lanes, in six sections — Overview, Models, Telemetry, Clients, Activity,
Configuration — under one header that carries the state, the last operation's
result, and the only Start, Stop, Restart, and Preflight controls: text and
sidecar residency, load/unload and runtime policy, unified-memory, process-CPU,
Metal and thermal telemetry with observed request and cache/batching traffic,
typed Pi readiness (read when Clients opens), install, configure and session
actions, copyable client setup, LAN and auth safety, and sanitized lifecycle
activity. Configuration offers Restart to apply while Studio's server runs. It polls the authenticated
`/runtime/status` contract and tolerates older payloads with missing additive
fields. The API server has one app-wide owner, `StudioKit/StudioLocalServer.swift`,
shared with the menu bar extra: it keeps the serve options, starts `api serve` on
the Settings endpoint with the Keychain key, adopts a server started from the
Command Console, restarts only after the old process has exited, and reads its
phase (stopped, starting, running, stopping, running outside Studio, stopped
unexpectedly) from the job and the endpoint monitor together.

**Server ▸ Vision server** and **Server ▸ Music server**
(`StudioUI/StudioResidentServerViews.swift`, `StudioUI/StudioMusicServerView.swift`)
run `vision serve` and `music serve` (and `world serve`, which has no page) through
`StudioKit/StudioServiceProcess.swift`, the same service-lane owner the API
server's process uses. Both pages start, stop, and restart their server; show why
it stopped and its live log; and use the task's Command view edits. Vision server
also offers preflight. Music server edits its checkpoint and adapter stack.
Neither server is a Library run, and the menu bar lists each while it runs, with
Stop. A server started from the Command Console goes to the same service lane —
the console still shows its log — and its owner adopts it; a `--preflight` run
stays an ordinary console run. Agent sessions stay durable Library runs. Open
WebUI is a Docker launcher, not a resident server, and runs from the Command
Console.

**Runs** is the domain over the public `executor` and `run` contracts. It
discovers local durable reports, lists Relay jobs, polls typed inspection state,
shows artifact inventories, reveals local runs, and exposes verified fetch,
cancellation, and immutable Relay retry. `run inspect --json` answers in one of
three shapes (a Relay job, a local graph run, or the inspection envelope around
an image run, a transcription run, a training directory, a report, or a plan);
`StudioKit/StudioSpecialistResults.swift` decodes each and the page shows the
state, what went wrong, the facts, each step with its state, and the outputs
with Reveal, with the raw report behind a disclosure. Client-side Relay profile setup and
device sign-in also go through the CLI; Studio streams the approval URL but
never handles the credential itself.

**Plugins** consumes the CLI's enriched plugin snapshot, shows installed path,
version, and manifest verification, offers channel selection and copyable pinned
install commands, confirms install or update, runs the plugin's fixed doctor
verb, and rolls back to a retained signed bundle behind a confirmation. Plugin
implementations stay out of process.

Hugging Face tokens, API keys (including the Settings ▸ Server key the
`status --json` probe sends), and the Open WebUI admin password cross the
process boundary through environment variables (`MERERUN_API_KEY` and its
siblings) instead of appearing in argv, in both the typed surfaces and the
console; the command preview and Library rows show argv only, so the value
never appears there either.

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
the region-prompt editor with boxes and points drawn on that image and Segment
and Track on the Analyze board with prompts drawn and Track's frame scrubber
over an in-test clip,
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

Displayed time is frozen. Around each render `StudioSnapshotRenderer.render`
sets `StudioDisplayClock.fixedDate` to its reference date and injects the same
instant as the `studioReferenceDate` environment value, and the snapshot
fixtures date their jobs, threads, library rows, and usage records relative to
that reference. The elapsed counters on running rows read
`referenceDate ?? context.date` inside their `TimelineView`, so the boards that
show elapsed time or a relative date render the same strings on every run
instead of following the wall clock, and generated file names use the same
instant.

What is not frozen is how long the harness waits. `render` pumps the main run
loop for a fixed wall-clock budget (`settle`, 1.5 seconds by default, plus a
0.3-second pass after the offscreen preparation) to let layout, `.task` work,
and in-flight animations land, so a loaded machine can still capture a board
mid-settle. Together with the offscreen substitutions above — lifted glass
content, hidden scroll-edge effects, an opaque title bar — that makes the shots
comparable by eye across runs, not a byte-for-byte pixel gate.

`StudioKitTests/StudioLiveAcceptanceTests` is the other gated harness: it runs
the real CLI against the models installed on the Mac, without the app. Each
test builds a flow's command the way its page does (`StudioCommandAdapter` for
composer tasks, a `StudioTaskDraft` through `StudioTaskRunner.prepare` for
tasks on the shared task workspace, or a task-specific page's `CommandDraft`),
runs it,
and decodes the output with the page's decoder, asserting what the page would
show — Decisions, Segment with a drawn box and points on a generated photo
(with an EXIF-rotated copy), Track's prompt frame and range, Find handing its
boxes to Segment, Faces, Depth, multi-view geometry and InstantMesh cameras,
Who Spoke, Music analyze and transcribe, instruments, and the training manifest, run plans and
`run inspect`, Sound's renoise through the task draft and the runner, a CLAP
score decoding for the gauge, the Woosh latents round trip through Encode and
Decode, a short Video Foley run on a synthesized clip, a chat turn with thinking shown, and a failed
turn's one-line reason. Inputs are drawn, synthesized, or generated with the CLI
into the run directory; nothing binary is committed. It is skipped unless
`MERERUN_LIVE_ACCEPTANCE_DIR` names a directory, and each test skips on its own
when a model it needs is not installed (`mere.run model list --json`). The CLI
is the package's debug build, or `MERERUN_LIVE_CLI`. Every step's argv, stdout,
stderr, and exit code are kept under `<dir>/<flow>/`, and `<dir>/summary.log`
gets one line per flow. A full pass loads a dozen models and takes several
minutes; one flow at a time is `--filter StudioLiveAcceptanceTests/test04`:

```
swift build
MERERUN_LIVE_ACCEPTANCE_DIR=/tmp/live swift test --filter StudioLiveAcceptanceTests
```

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
