# macOS Studio v2: review and plan

Reviewed on September 3, 2026 against `main` at `1ef4fda3` (v0.50.0).
`apps/macos/MereRunStudio` is 39,446 lines across 57 files; the test target is
5,532 lines and 244 tests. The review covered the shell and navigation, the
prompt-to-result loop, the nineteen specialist workspaces, and the code
architecture, each read in full.

## Verdict

v1 is two products in one window. The first is the prompt-first Studio: a
grouped sidebar, one composer, a canvas, a library. It is well made and the
review keeps almost all of it. The second is a general-purpose GUI over the CLI
command tree, bolted onto the first as **22 modal sheets** with 14 different
fixed window sizes, reached from a second navigation system in the sidebar
footer and from up to six contextual buttons in the mode header.

The second product is what feels clunky and confusing. It was produced by a
policy, recorded in `macos-studio-capability-review.md`, that every CLI command
must have a "first-class workspace," and by reading "first class" as "its own
sheet." v2 keeps the goal and changes the reading: every capability is a peer
destination reached the same way as every other, and what varies is the
interaction shape it declares, not how far away it lives. "Lab", "Workshop",
"Tools", and "Advanced" are not categories in v2.

### Evidence, by area

Shell and navigation (`StudioRootView.swift`, `StudioSidebar.swift`):

- 19 `.sheet` modifiers on the root plus three nested sheets (Models → Health,
  Models → Locations, Options popover → mask editor). Sidebar rows "Serving &
  Agents", "Runs", "Plugins", "Models", "Geospatial" each open a sheet rather
  than navigate.
- The root view holds 41 `@State` values; 20 are presentation flags. Sheet
  handoffs use relay flags such as `openTrainingAfterMusicTools` because only
  one sheet can be presented at a time.
- The shell is a hand-rolled `HStack` with hard-coded widths. There is no
  `NavigationSplitView`, `.toolbar`, `.inspector`, or `NavigationStack` in the
  app, so no native sidebar collapse, column resize, or toolbar overflow.
- The only accent-colored primary button in the shell opens the server console.
  "Geospatial" wraps to two lines in a 2×2 footer grid.
- Menus: View contains only system items. No File menu, no New Chat, no Go menu.
  Tooltips advertise ⇧⌘G for Geospatial; nothing binds it. ⇧⌘S and ⇧⌘P are
  taken for Serving and Plugins against macOS convention.
- Menu **Run ▸ Run** executes the Advanced template, not the composer prompt.
- The readiness overlay's "Details" button, the Library "Edit" action, and
  applying an adapter in five modes all open the raw CLI editor.

Creation loop (`StudioComposer.swift`, `StudioCanvas.swift`):

- The Options popover is a 920-line hand-written view. Control count by mode:
  Find/Segment/Track 1, Sound FX 4, Chat about 27, Create Image about 26, Music
  about 30. CLI vocabulary appears as primary copy: "Preflight only",
  "Machine-readable progress", "Sigma shift", "KV scheme", "17n+5".
- Switching modes replaces the draft, so a typed prompt and attachments are lost.
- The model quick-picker lists every installed model regardless of mode; a
  second free-text "Model" field lives in the popover, and each keystroke there
  fires a readiness probe that greys out Send.
- Running state is one global flag mirrored from a "foreground" session. The
  progress overlay covers whatever item is selected; Stop cancels the newest
  run, not the one on screen; the overlay follows you into another mode.
- A missing prompt raises a full-canvas red "Needs attention" overlay. An actual
  run failure shows a "Failed" badge and the raw stdout plus stderr in
  monospace. Severity is inverted.
- Completion force-selects the finished item three separate ways, interrupting
  whatever you were reviewing.
- "Edit" on a result opens the Advanced editor. There is no Vary, Use as input,
  batch count, or reload-into-composer.
- Outputs land in `~/Library/Application Support/MereRun/App Outputs/` with
  template-and-timestamp names. No visible folder, no Save or Export.
- Chat threads share the Library list with images. Model and system prompt are
  captured only when a thread is created; later turns run with the composer's
  current model and the thread never records which.

Specialist workspaces (the 19 sheets, about 15,000 lines):

- Three different skeletons (segmented picker + two columns; task rail + three
  columns; single scroll of cards), form columns from 380 to 470pt, rails from
  170 to 210pt with eyebrows "TOOLS", "WORKSPACE", "SPECIALISTS", or none.
- Naming suffixes: Lab, Studio, Tools, Console, Center. Title, button label,
  and file name disagree for four of them.
- No specialist job can be cancelled, although `cancel(requestID:)` exists.
  Music Tools "Stop server" calls the global cancel and kills whatever is
  running. Job state is `@State` in the sheet, so closing a sheet mid-download
  orphans the progress UI while the process continues.
- Library rows are attributed to the wrong mode: Geospatial → Read Image,
  benchmarks and API server → Chat, adapter pulls → Chat.
- `MereBanner`, which DESIGN.md names as the only notice component, is used once
  in the entire specialist layer. Seven other idioms carry errors and status.
- About 1,800 to 2,000 lines are near-identical scaffolding: nine copies of a
  labeled text field, seven of a metric tile, seven of `timestampedOutput`, five
  task rails, 19 sheet headers, three inline copies of the submit-to-Library
  body, two runtime-settings editors, two loss charts.
- Eight of the surfaces (Geospatial, Audio, Health benchmarks, 3D, Music Tools,
  Vision Lab, SFX Lab, Utility Lab) are forms over a command family whose shape
  the contract already declares. Their real value is the bespoke result
  renderers (pose overlay, flow field, piano roll, CLAP gauge, PII spans).

Architecture (`MereRunController.swift`, `CommandCatalog.swift`,
`MereRunRootView.swift`):

- `CommandDraft` has 481 fields; `arguments(from:)` is an 1,800-line switch with
  1,024 raw flag literals. The typed contract in `MereRunContract` (127
  capabilities, 1,206 options) is used as data at one site. Three flags have
  already drifted from the contract.
- `MereRunRootView.swift` is misnamed: 12 lines of root wrapper, then about
  5,250 lines of Advanced form views bound directly to `controller.draft`, plus
  Settings.
- The controller mixes process launching, the Advanced editor's draft, the
  foreground console mirror, cross-run state, readiness, and seven persisted
  settings.
- The CLI is invoked through four front doors: composer adapter, Advanced
  draft, `StudioSpecialistRunner`, and 37 `utilityCommandResult` sites with
  hand-built argv that bypass queueing, progress, Library, and cancellation.
- Artifacts are detected by `fileExists` probing on a 350 ms poll plus stdout
  path heuristics. The contract declares each command's output kind; the app
  does not read it.
- Adding a `StudioMode` touches about 30 sites. Adding a workspace touches the
  root view in three places plus a local runner, result view, and task enum.
- None of the 244 tests exercise a view. About 1.1 MB of SwiftUI has no
  coverage, and there are no snapshot tests.

### What v1 got right (v2 keeps all of it)

- `StudioMode` as the product vocabulary: title, glyph, template, accepted
  types, placeholder, serif empty-state line, and example prompts in one place.
- The grouped sidebar with the sliding bronze selection pill and ⌘1–⌘9.
- The status cluster: one dot, one phrase, detail in a popover.
- The composer's shape: prompt first, attach, model pill, Options, ⌘↩.
- Empty states with example chips that fill but never run.
- The canvas state machine including the in-place "Get the model" readiness
  path and pull progress rendered in the canvas.
- Markdown renderer tolerant of unclosed fences; waveform with seek and
  VoiceOver; streaming caret and thinking dots; context-trim banner.
- Library persistence: lenient decoding, corrupt-file quarantine, atomic writes,
  day grouping, keyboard navigation, hover Quick Look, the receipt import
  contract.
- `RunSession` isolation, the two-lane queue, `progressByRequestID`,
  `cancel(requestID:)`, the lossless `runCompletions` stream.
- `MereRunContract` as a shared target with drift tests in both directions;
  `mere.run catalog --json`.
- The process seam (`MereRunProcessRunning`), secrets through environment
  variables, argv masking, `terminateAllProcesses` on quit.
- The theme tokens, control styles, Reduce Motion handling, deep-link strictness,
  CLI installer receipts, diagnostics with no secrets, opt-in MetricKit.
- The bespoke result renderers inside the labs, and Voice Studio's user-language
  task names ("Create / Transcribe / Listen Live / Who Spoke / Voices").

## The v2 design

One navigation. Every capability is a peer, reached the same way. What varies
is the **surface archetype** a task declares, which decides the shell it gets.

| Archetype | Shape | Tasks |
|---|---|---|
| Generate | Prompt first, feed of results, composer pinned at the bottom | Image, Video, Music, Sound, Speak, 3D |
| Converse | Thread list, turns, per-turn model | Chat, Code |
| Analyze | Input first, prompt optional, result renderer specific to the output | Read, Find, Segment, Track, Transcribe, Who Spoke, Depth, Pose, Faces, Flow, Geometry, Earth tiles, Embeddings, Anonymize, Enhance, Separate, CLAP score |
| Session | Long-lived process with transport controls and live state | Realtime music, Live listen, API server, Vision serve, Agent |
| Project | Multi-stage with persisted state, dashboards, compare | SCAIL subjects, LoRA training, run comparison |
| Manage | List and detail with actions and confirmations | Models, Adapters, Plugins, Runs, Locations, Voice profiles |

Every sidebar row is a **domain**. Every domain has **tasks** in a task control.
Every task declares its archetype, its capability schema, and any bespoke result
renderers; the archetype supplies the shell. Training is not a destination, it
is a Project task inside Image, Text, and Music. Benchmarks and Health are Manage
tasks inside Models. Geospatial is a domain like any other.

A capability nobody has designed a surface for yet still appears as a task in
its domain, rendered by the generic contract form under the Analyze or Manage
archetype. That is the "no bespoke UI yet" state, not a separate tier. It is
also what makes coverage honest: every contract capability maps to exactly one
(domain, task) pair, and the coverage test asserts that mapping instead of the
current template-per-capability check.

There is no separate Advanced or Console place. Every task has a **Command
view** (⌥⌘C) that shows the same draft as the raw contract form plus the exact
argv, editable, with Run. Same destination, different display mode.

### 1. Shell and navigation

`NavigationSplitView(sidebar, content, inspector)` with one `StudioDestination`
selection and one `sheet(item:)` for the few true task sheets.

```
CREATE      Image · Video · Music · Sound · Voice · 3D
CONVERSE    Chat
UNDERSTAND  Vision · Audio · Text · Earth
SYSTEM      Models · Server · Runs · Plugins
```

Tasks per domain (archetype in parentheses):

- Image: Generate, Edit (Generate); Validate dataset, Discover dataset (Analyze);
  Train LoRA (Project).
- Video: Generate (Generate); Subjects (Project, the SCAIL flow).
- Music: Compose (Generate); Realtime (Session); Analyze, Transcribe (Analyze);
  Train (Project).
- Sound: Generate, Video Foley (Generate); Score, Condition, Encode, Decode
  (Analyze).
- Voice: Speak, Clone (Generate); Voices (Manage).
- 3D: From image (Generate) with engine picker TripoSR / TRELLIS.2 / InstantMesh.
- Chat: threads with a Code preset (Converse); Train text LoRA (Project).
- Vision: Read, Find, Segment, Track, Depth, Pose, Faces, Flow, Geometry
  (Analyze); Live track (Session).
- Audio: Transcribe, Who Spoke, Enhance, Separate (Analyze); Live listen
  (Session).
- Text: Embeddings, Anonymize (Analyze).
- Earth: Flood, Fire, TESSERA, OlmoEarth (Analyze).
- Models: Installed, Locations, Health, Benchmarks (Manage); Pull is a job.
- Server: API, Vision, Agent, Open WebUI (Session) plus runtime settings; absorbs
  the Music resident server and the Models runtime-settings editor so each
  exists once.
- Runs: local durable runs and Relay jobs (Manage).
- Plugins: catalog and lifecycle (Manage).

Shell rules:

- Footer carries the status cluster only. No buttons.
- The sidebar header is the brand wordmark: "mere" in Caveat Medium with the
  green dot, replacing the serif "mere.run" text. Serif stays for display
  moments in the canvas.
- The task control sits in a real `.toolbar`: mode glyph and title leading, task
  control center, Library, Inspector, and Command toggles trailing. Nothing else.
- System pages keep their existing internal rails and drop the fixed frame and
  the Done button. Models' Health and Locations become tasks, not nested sheets.
- Sheets remain only for true tasks: third-party terms (alert), mask editor,
  Relay sign-in, rename.
- Menus: File ▸ New Chat ⌘N, Import Receipt…; View ▸ Library ⌥⌘L, Inspector
  ⌥⌘I, Command ⌥⌘C, Sidebar ⌃⌘S; Go ▸ every domain (⌘1–⌘9, then ⌥⌘1…) and
  every task of the current domain; Run ▸ Run ⌘↩, Stop ⌘., Open, Reveal, acting
  on the current task; Help ▸ Guide ⌘?, mere.run, Export Diagnostics. Retire
  ⇧⌘S and ⇧⌘P.
- First run: no Welcome sheet. The Image empty state with "Get the model" is
  the onboarding, plus a one-time dismissible banner.
- Settings gains tabs: General, Models, Server, Advanced.

### 2. Composer and inspector

Always visible: the prompt; an **attachment well** whose slots are declared per
mode (Image: input and references; Video: start frame, end frame, audio; Music:
source and timbre references; Voice: reference audio), each accepting drop,
paste, and picker with a thumbnail; a **chip strip** of the two to four
essential parameters ("1024×1024 ▾", "4 steps", "Seed random", "Auto model");
Run.

- One model control: a picker filtered to the mode's model category, showing
  installed / ready / download state per row, "Auto" by default. The popover
  text field goes away.
- The **inspector** (⌥⌘I, remembered per mode) replaces both the Options
  popover and the docked Advanced panel. It renders from a declarative schema
  with tiers: `essential` becomes chips, `standard` becomes inspector sections
  (Prompt, Inputs, Output, Model & Adapters, Sampling), `expert` collapses under
  "Advanced" at the bottom. Sections have Reset; the toggle shows a
  modified-count badge. The same schema renders the task's Command view, so the two
  cannot drift.
- Studio never shows CLI plumbing. Preflight, JSON report, progress JSON,
  timings, stats, require-installed, skip-recipe, and DAW bundle live in the
  Command view only.
- Per-mode drafts (`[StudioMode: StudioDraft]`) so switching modes never loses
  work. Model-family switches (LTX vs. MiniMax, ACE tasks) are an explicit Task
  picker at the top of the inspector, never a silent draft rewrite.
- Dimensions: aspect presets with swap, pixel steppers behind them. Seed:
  Random toggle, number field, "Reuse last".

### 3. Canvas and Library

Every mode is a **feed of generations**, oldest at top, composer pinned at the
bottom. This is the shape Chat already has.

- A generation is one request with N outputs. Running and queued generations are
  cards in the feed with their own progress, Cancel, and (queued) Remove. No
  full-canvas overlay, no selection yanking; a "New result ↓" pill instead.
- Card actions: click the prompt to reload it, Vary (new seed), Rerun, Use as
  input, Open, Reveal, Quick Look, Save to…, Delete. Click an output for a
  focused large view in place.
- Vision modes render natively: boxes over the image, masks composited, track
  overlays on the scrubber.
- Failure card: one-line summary from the last meaningful stderr line, a "Show
  log" disclosure, Retry. Validation errors appear inline under the composer.
  The full-canvas overlay is reserved for readiness (missing model, missing
  CLI).
- Library stays a column, filtered to the current mode by default with an All
  filter, and a Kind filter. Grid and list views, favorites, multi-select,
  drag-out from rows, inline rename, Delete offers to trash the files.
  Selecting a row from another mode switches the mode.
- Rows show real thumbnails: poster frame for video, mini waveform for audio.
- Outputs go to a visible folder, default `~/Pictures/mere.run/<Mode>/`,
  configurable, named `<prompt-slug>-<seed>.<ext>`. Application Support holds
  metadata only.

### 4. Converse

Chat and Code become one Converse surface, typed separately from generations.

- A thread list replaces the Library column here. ⌘N starts a thread.
- Thread header shows model and system prompt as editable fields; changes are
  recorded on the turn where they took effect, and every turn stores the model
  that produced it.
- Code is a preset (system prompt, monospaced code blocks with proportional
  prose) inside Converse, not a second thread pool.
- Edit-turn offers branch as well as truncate. Context budget derives from the
  model's context size rather than a fixed 48k characters.
- `Thread` and `Generation` are distinct types. Threads do not appear in the
  media Library.

### 5. Jobs

One `Job` model replaces the foreground mirror and the utility path.

```swift
enum JobLane { case inference /* cap 2, FIFO */, utility /* cap 4 */, probe /* dedupe */ }
enum JobState { case queued, running(since: Date), finished(exit: Int32, at: Date), cancelled, preflightFailed(String) }

@MainActor final class Job: ObservableObject, Identifiable {
    let id: JobID; let request: JobRequest; let displayCommand: String
    @Published var state: JobState
    @Published var progress: StudioRunProgress?
    @Published var log: LogRing
    @Published var liveText: String
    @Published var artifacts: [Artifact]
}

@MainActor final class JobStore: ObservableObject {
    func submit(_ request: JobRequest) -> JobID
    func cancel(_ id: JobID)
    func send(_ text: String, to id: JobID) throws
    func result(for id: JobID) async -> JobResult
    let completions: PassthroughSubject<JobResult, Never>
}
```

- Everything that runs longer than about two seconds is a job, including model
  pulls, plugin installs, and live listening, so it is cancellable, visible,
  and survives navigation. `utilityCommandResult` becomes a utility-lane job
  awaited inline; list and status reads keep it.
- `JobStore` owns the Library write-through. The root view no longer writes
  library rows from `onReceive`.
- Views observe a `Job` by id. `isRunning`, `logs`, `liveOutputText`,
  `currentProgress`, and `lastOutputURL` leave the controller.
- An **Activity** popover on the status cluster lists every in-flight and
  queued job with Stop per row.
- Library mode attribution derives from the template's category, not a
  hand-passed value.

### 6. Tasks, archetypes, and forms

```swift
protocol StudioTask {
    static var id: StudioTaskID { get }            // (domain, task)
    static var title: String { get }
    static var symbol: String { get }
    static var archetype: SurfaceArchetype { get }  // generate, converse, analyze, session, project, manage
    static var capabilities: [String] { get }       // contract capability IDs this task covers
    associatedtype Draft: Codable & Equatable
    static func initialDraft(seed: Seed?, defaults: CapabilityDefaults) -> Draft
    static func requests(from draft: Draft) throws -> [JobRequest]
    static var renderers: [ResultRenderer] { get }  // bespoke output views, may be empty
    @ViewBuilder static func controls(draft: Binding<Draft>) -> Controls   // optional bespoke inputs
}
```

- Each archetype is one shell: `GenerateShell` (composer, feed), `ConverseShell`
  (thread list, turns), `AnalyzeShell` (input well, optional prompt, result
  column), `SessionShell` (transport bar, live panel, log), `ProjectShell`
  (stage rail, dashboard, compare), `ManageShell` (list, detail, action bar).
  All six share the header, the job bar (progress, Cancel, Reveal, Open in
  Library, log popover), `MereBanner`, and the Command view.
- A task with no bespoke `controls` gets `ContractForm(capability:draft:)`,
  which renders `MereRunCapabilityOption.kind` (`string`, `integer`, `number`,
  `boolean`, `file`, `directory`, `choice`) to controls. The contract gains
  additive `defaultValue`, `group`, `tier`, `range`, and `dependsOn` metadata so
  disclosure and conditional fields come from data. A per-flag override
  registry covers the composite editors (ordered views, SCAIL subjects, MiniMax
  references, dimensions, mask brush).
- A task with no bespoke `renderers` gets the shared result view (Quick Look,
  media players, JSON table, text). Existing bespoke renderers (pose overlay,
  flow field, piano roll, CLAP gauge, PII spans, embeddings matrix, loss chart)
  move under `Renderers/` and register by extension or JSON shape.
- The task registry drives the sidebar task control, the Go menu, Library kind
  titles, and the coverage test.
- `MereBanner` is the one notice. `.merePrimary` and `.mereSecondary` are the
  only button styles. Task names are user-language nouns and verbs; Lab, Studio,
  Console, Center, Tools, and Advanced leave the UI.

### 7. Architecture

```
apps/macos/
  StudioKit/      library, no SwiftUI: CLI resolver and environment, ProcessRunner,
                  Job/JobStore/lanes, ArgumentBuilder + per-category catalog,
                  ArtifactResolver, ProgressParser, Library store, transcript,
                  readiness, config, serving monitor, installer
  StudioUI/       library, SwiftUI: Shell, Modes (one descriptor file per mode),
                  Domains (one folder each, tasks inside), Archetypes, Forms, Renderers
  MereRunStudio/  executable: App, AppDelegate, Commands, Settings, Sparkle
```

- `ArgumentBuilder` takes contract-derived flag constants so an unknown flag is
  a compile error. `arguments(from:)` and `defaultDraft()` are deleted;
  `CommandTemplate` records stay as data. `CommandDraft` survives only as the
  persisted wire struct behind `StudioLibraryItem.commandDraft`.
- `ModeDescriptor` per mode collapses the ten switches in `StudioMode`,
  `StudioCommandAdapter.makeRequest`, `StudioDraft.reset`, and
  `syncAdvanced` into one mapping per mode.
- The CLI emits a final NDJSON `{"event":"result","outputs":[…]}` line and
  extends `--progress-json` to every long-running command. `ArtifactResolver`
  prefers the receipt, then the contract's output kind, and keeps `fileExists`
  only as a fallback for old CLIs. The 350 ms poll goes.
- `MereRunRootView.swift` is deleted. The Command view is `ContractForm` plus an
  argv preview and a job console, scoped to the current task.
- Snapshot tests render key views offscreen with `ImageRenderer` in light and
  dark, regular and compact, so visual review never requires driving the live
  app.

## Sequence

Each PR leaves `./scripts/check.sh` green and the app shippable. S is under 300
lines, M is 300 to 1,000, L is several M PRs. The judgment-heavy steps are A1
to A5, B1, and C1; the rest is mechanical and clearly specified.

### Phase A: foundation and shell (the visible win)

| # | PR | Size |
|---|---|---|
| A0 | Contract guard: build argv from a maximal draft and assert it is a subset of the contract; add the three drifted flags to the contract | S |
| A1 | Extract `Job`, `JobStore`, `JobLane` from `RunSession` and the queue; controller becomes a facade with its published API intact | M |
| A2 | Route `utilityCommandResult`, readiness probes, and server status through job lanes | M |
| A3 | Library write-through moves into `JobStore`; delete the root view's `onReceive` wiring and the four inline runner copies | M |
| A4 | `NavigationModel` + `StudioDestination` (domain, task); `NavigationSplitView` shell with a real toolbar and task control; System sheets become pages; footer reduced to status; Go menu and shortcuts. Existing lab views are re-hosted unchanged as tasks in their domains so nothing is lost while the archetype shells are built | L (3 PRs) |
| A5 | Views observe `Job` by id; Activity popover; delete the foreground mirror and global running overlay; fix Run menu gating | M |
| A6 | Per-task drafts; Library filtered by domain with row-to-task switching; delete the Welcome sheet | S |

### Phase B: the Generate and Converse archetypes

| # | PR | Size |
|---|---|---|
| B1 | Option schema with tiers; inspector renders from it; chip strip; purge CLI plumbing from the designed surfaces; delete the Options popover | L (2 PRs) |
| B2 | Attachment well with per-task slots; one filtered model picker | M |
| B3 | Feed canvas: generation cards, inline running and queued cards, failure card, inline validation | L (2 PRs) |
| B4 | Library grid, filters, favorites, thumbnails, visible output folder and naming | M |
| B5 | Converse shell: thread list, ⌘N, per-turn model, Code as preset, branch on edit | M |

### Phase C: the remaining archetypes and the Command view

| # | PR | Size |
|---|---|---|
| C1 | `StudioTask` protocol, task registry, shared header and job bar, `MereBanner` everywhere; `AnalyzeShell` with Earth as the template task | M |
| C2 | Contract gains `defaultValue`, `group`, `tier`, `range`, `dependsOn`; CLI emits result receipts and progress JSON for long commands | M (CLI) |
| C3 | `ContractForm`; migrate the schema-shaped tasks (Audio, Vision, Sound analysis, Text, Benchmarks, 3D, Music analyze and transcribe) onto `AnalyzeShell` and `ManageShell`, native result rendering for Vision (boxes, masks, track overlays) | L (one S per task) |
| C4 | `SessionShell` (Realtime, Live listen, Server) and `ProjectShell` (Subjects, Training, compare); fold Voice Studio into Voice and Audio; remove the header lab buttons | L (one S–M each) |
| C5 | `ArgumentBuilder` with contract flag constants; split the catalog by category; delete `arguments(from:)` and `defaultDraft()` | L (mechanical, per category) |
| C6 | Command view on every task from `ContractForm` plus argv preview; delete `MereRunRootView.swift`; move Settings; coverage test asserts every capability maps to one (domain, task) | L (per category) |
| C7 | `ArtifactResolver` on receipts; delete the poll and the six view-side artifact strategies | M |

### Phase D: structure and ship

| # | PR | Size |
|---|---|---|
| D1 | Split into `StudioKit` / `StudioUI` / `MereRunStudio`; app-lifetime serving monitor | M |
| D2 | Offscreen snapshot tests for the shell, each archetype in its empty, running, result, and failed states, light and dark, regular and compact | M |
| D3 | Update `apps/macos/README.md`, `PRODUCT.md`, `DESIGN.md`, and the capability review to the domain-task-archetype policy; release 2.0 | S |

Compatibility held throughout: the `library.json` row shape (additive
optionals), `UserDefaults` keys under `mererun.app.*`, `@SceneStorage` keys
under `studio.*`, the `mererun://` deep-link contract, and the Raycast receipt
format.

## Decisions to confirm

1. **Vision as one domain with nine tasks** rather than v1's four sidebar rows
   (Read, Find, Segment, Track). One row keeps the sidebar to 15 rows; four
   rows keep the shortcuts people already have.
2. **Code as a Converse preset.** The alternative is a Code row that shares the
   thread list.
3. **Library as a column, not a destination.** A full-page Library is a small
   addition later if wanted.
4. **Output folder default.** `~/Pictures/mere.run/<Domain>/` for images and
   video, `~/Music/mere.run/<Domain>/` for audio, or one root for everything.
5. **Training placement.** Modeled as a Project task inside Image, Chat, and
   Music. The alternative is one Training task under Models, since adapters
   are model artifacts.
