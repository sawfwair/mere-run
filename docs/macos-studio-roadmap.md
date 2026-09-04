# MereRun macOS Studio historical review and v1.0 roadmap

This page preserves a historical source review of `apps/macos/MereRunStudio`
(approximately 6,500 lines of code) against the CLI source of truth in
`Sources/MereRunCLI` and the bundling pipeline. The implementation update
identifies which findings no longer describe the product.

> **Implementation update (September 2026):** This document preserves the
> original pre-parity audit and sequencing rationale. The capability gaps in the
> historical findings no longer describe the product, and neither does its
> vocabulary: the Advanced surface, the sheets, and the mode header's lab
> buttons were all retired by [macOS Studio v2](./macos-studio-v2.md), which
> replaced them with domain-and-task navigation. The app consumes the shared
> machine-readable 127-command capability contract and has executable drift
> tests for every template it can run. Studio provides typed text, image, video,
> music, speech, sound, vision and VFX, Earth-observation, adapter, run, world,
> setup, model, model-location, plugin, benchmark, Open WebUI, and API
> workflows; Graph Studio and Node remain explicit external boundaries, now
> recorded as named exemptions in the CLI coverage test rather than as silent
> absences. Structured receipts, file pickers, validation, retry and
> resume controls are implemented. See
> [`apps/macos/README.md`](../apps/macos/README.md) for the
> implemented surface. On July 27, 2026, the release pipeline produced a Developer
> ID-signed and Apple-notarized app and DMG, both passed Gatekeeper and stapler
> validation, and the installed app completed a real image-generation workflow
> through its embedded CLI. The remaining findings and phases are retained as the
> historical audit that motivated the implementation. They are not a statement
> of product capability or release readiness.

> **Closure update (August 30, 2026):** Every item in this document has been
> re-checked against the sources rather than taken on trust. All eight historical
> quick wins are closed — two by a different route than proposed, noted in the
> table below. Phases 0 through 3 are closed. Phase 4's two remaining items,
> opt-in MetricKit capture and Export Diagnostics, are now implemented, closing
> the phase. The one deliberate deferral is Phase 4's sandbox posture, which the
> risks section records as a product decision rather than an oversight.
> Capability coverage against the CLI is tracked separately in
> [`macos-studio-capability-review.md`](./macos-studio-capability-review.md).

## Historical verdict (superseded by the implementation update)

At the time of the original audit, the app was **not 5% finished — it was a
solid ~45% v0 skeleton** with misleading breadth and shallow depth. The bones
were real (a clean `@MainActor` controller, an injectable process layer, a
34-command catalog, a persisted library, a capable Models sheet, and ~1,000 LOC
of tests). The missing work was concentrated exactly where "finished" lives:
it could not ship (ad-hoc signed, no notarization, a camera-permission crash),
and core modes dead-ended (audio/video showed an icon, chat was one-shot, and
downloads showed a wall of logs). Those claims are historical and superseded by
the implementation update.

### Dimension scorecard

| Dimension | State | Headline problem |
|---|---|---|
| Architecture and orchestration | **55%** | Output detected by scanning stdout for paths that `fileExists`, not the CLI's `--json` or `--quiet` contract; engine hard-capped at one run; UTF-8 chunks silently dropped |
| UX and interaction | **45%** | Speak, Music, and Video dead-end on an icon; chat single-shot; spinner-only progress; library has no search or delete; Read Image opens on a blocked action |
| Feature parity compared with the CLI | **62%** | `sfx`, `music realtime`, `music analyze`, `config`, `status`, `plugin`, `open-webui`, `benchmark`, and `guide` have no GUI; chat omits tools and vision; voice cloning unwired |
| Native platform | **28%** | Ad hoc signed, no entitlements, no `NSCameraUsageDescription` while shipping `vision track-live` (TCC termination); child processes orphaned on quit; hardcoded dark theme; no accessibility coverage |
| Visual and polish | **55%** | Strong identity, but 1,210-pixel Advanced panel placed in a 560-pixel viewport; invalid `Image(systemName: "finder")`; no semantic tokens; native controls clash in light mode |
| Build and distribution | **30%** | Hand-built bundle, executables under `Contents/Resources` (notarization failure), no DMG or notarization step despite the README claim, and no macOS CI |

---

## Historical release blockers (resolved)

1. **Camera TCC crash.** `vision track-live` → `SAM31CameraCapture` opens `AVCaptureDevice`, but the generated `Info.plist` has no `NSCameraUsageDescription`. Because the CLI runs as a child of `MereRun.app`, TCC attributes camera access to the app bundle → process is terminated with no prompt. _(scripts/build_mere_run_app.sh, VisionTrackLiveCommand.swift:79)_
2. **Gatekeeper rejection.** `codesign --force --sign -` (ad-hoc), no hardened runtime, no entitlements, no `notarytool`/`stapler`. Any downloaded copy gets the "damaged / can't be opened" error. _(scripts/build_mere_run_app.sh:94)_
3. **Notarization-fatal layout.** The CLI binary, `llama.framework`, `mlx` bundle, and `vendor/ds4/ds4-server` are copied into `Contents/Resources/`. `notarytool` rejects executable Mach-O under Resources.
4. **Orphaned processes.** No `applicationShouldTerminate` hook — quitting mid-run (especially `api serve`) leaves the child CLI running. `terminate()` only fires on explicit Stop.
5. **README mismatch.** The README advertises a "signed and notarized macOS DMG," but no pipeline produces a DMG.

---

## Historical quick wins (all verified closed)

Each item below was re-checked against the sources on 2026-08-30. The result is
recorded beside it. Two were closed by a different route than the audit
proposed, which is noted rather than hidden.

| Quick win | State | Where |
|---|---|---|
| Read Image default opens a blocked action | Closed by the alternative route: `inspect` and `caption` use a VLM the CLI auto-downloads, so they are no longer gated by the managed catalog | `StudioTypes.swift` `capabilityRequirement` |
| Invalid `Image(systemName: "finder")` | Closed — no occurrences remain | — |
| Force `.preferredColorScheme(.dark)` | Obsolete. Superseded by Phase 4: `MereRunTheme` resolves every token per effective appearance, so forcing dark would now be wrong | `MereRunTheme.swift` |
| UTF-8 chunk loss | Closed — `IncrementalUTF8Decoder` retains partial trailing bytes and backs off to a codepoint boundary | `MereRunController.swift` |
| Silent library persistence failures | Closed — `lastPersistenceError` is published | `StudioLibraryStore.swift` |
| Prompt bar clipping | Closed — `.windowResizability(.contentMinSize)` plus an explicit minimum frame | `MereRunApp.swift` |
| Missing `NSCameraUsageDescription` | Closed — camera and microphone usage strings are inserted, with matching entitlements, and `track-live` is gated behind `AVCaptureDevice.requestAccess` | `scripts/build_mere_run_app.sh`, `scripts/MereRun.entitlements` |
| README DMG claim | Closed — no unsupported DMG claim remains | `README.md` |

## Historical quick wins (original text)

- **Read Image default:** Change `readImageAction` default and reset from
  `.inspect` to `.ocr`, or register the auto-download VLM in the capability
  catalog. At audit time, the mode opened on a permanently blocked action.
  _(StudioTypes.swift:190,202)_
- **Invalid SF Symbol:** `Image(systemName: "finder")` is not a real symbol; renders as a missing glyph in 3 places. Use `magnifyingglass`/`folder`. _(StudioRootView.swift:733, MereRunRootView.swift:382, StudioModelsView.swift:403)_
- **`.preferredColorScheme(.dark)`** at the WindowGroup root so native pickers/steppers/sheets stop rendering light-on-dark against the custom dark panels.
- **UTF-8 data loss:** buffer raw `Data` per stream and retain incomplete trailing bytes instead of dropping whole chunks when a codepoint straddles a read. _(MereRunController.swift:267-277)_
- **Library error visibility:** publish a `lastPersistenceError` instead of the empty `catch` so silent history loss is detectable. _(StudioLibraryStore.swift:135-137)_
- **`.windowResizability(.contentMinSize)`** + explicit minimum frame so the prompt bar can't be clipped. _(MereRunApp.swift:14)_
- **Add `NSCameraUsageDescription`** to the plist insert block and hide `visionTrackLive` until permission is gated.
- **Correct the README:** Remove the DMG claim until the pipeline produces a notarized artifact.

---

## Original phased roadmap

### Phase 0 — Platform correctness and releasable pipeline _(3–4 weeks; foundational)_
Make the bundle structurally valid, signable, notarizable, Gatekeeper-clean, crash-free on permissions, and CI-built.
- **TCC and camera safety:** Add usage strings, and gate `track-live` behind
  `AVCaptureDevice.requestAccess` before launch.
- **Signing and notarization:** Add `MereRun.entitlements`, use Developer ID
  with `--options runtime --timestamp`, sign from the inside out, and run
  `scripts/package-macos.sh` for DMG creation, notarization, and stapling.
- **Bundle layout** — frameworks → `Contents/Frameworks`, helper executables → `Contents/MacOS`/`Helpers`; update `CLIResolver.bundledCandidates()`.
- **macOS packaging** — `scripts/package-macos.sh`; per-PR `build_mere_run_app.sh debug` job; version from git tag + commit count.
- **Process lifecycle** — `applicationShouldTerminate` kills all children; UTF-8 fix; README correction.

**Exit:** downloaded DMG opens clean (`spctl --assess` passes); `track-live` prompts for camera; `notarytool` accepts; quit kills all children; CI publishes a notarized DMG.

> **Note:** Start Developer ID certificate and `notarytool` credential
> provisioning on day one. This external dependency gates the pipeline and can
> be reproduced only on a clean machine.

### Phase 1 — Run engine and CLI contract refactor _(4–6 weeks; architectural foundation)_
Replace heuristic orchestration and the single-run ceiling so later features sit on contracts, not guesses.
- **Structured result channel** — consume `--quiet` path lines / `--json` (already emitted by ~14 commands) instead of `detectOutputURL`'s last-40-line `fileExists` scan; decode into typed `RunResult`/artifacts; collapse the dual poll/per-chunk triggers into one debounced off-main path.
- **Per-run session model** — `RunSession` keyed by id (own process, log buffer, status, immutable request) behind a coordinator with N concurrent slots; lift `guard !isRunning`; cancel/remove/reorder queued items; per-run logs (the shared `logs` is wiped every run).
- **Decouple editing from running:** `startRun` consumes
  `request.template/draft`. Use per-surface editing drafts so a dequeuing run
  does not overwrite Advanced input.
- **State ownership and testability:** Move runtime server parameters, such as
  host, port, and key, into dedicated persisted state. At audit time, the
  Models sheet used the transient command draft. Inject a CLI locator and file
  system abstraction, and add URLProtocol stub and real-binary runner tests.

**Exit:** Artifacts resolve from the CLI contract, including directory and
multi-file results. Two modes run concurrently with isolated logs. Editing does
not overwrite active runs. Models load and unload targets the real runtime.
Tests cover resolution, detection, the runner, and HTTP.

> Land the CLI contract and the session model as **separate incremental drops behind existing behavior** to avoid a long-lived branch on the controller everything depends on.

### Phase 2 — Media and progress UX _(3–4 weeks)_
Make every mode's output actually usable.
- **Inline playback** — AVKit `VideoPlayer` for `.mp4/.mov`; `AVAudioPlayer` transport (play/scrub/time/waveform) for `.wav/.mp3/.m4a`; extend `StudioOutputFileKind.classify` for audio/video.
- **Determinate progress** — parse ModelPull's `[id] NN% done/total (speed/s)` stderr lines into a progress model; `ProgressView(value:)` with bytes/speed/ETA; collapse repeated `\r` updates; same for step-based generation.
- **File ingestion** — `.dropDestination` on canvas + prompt for each mode's accepted types; paste-image for createImage/readImage; Read Image default fix; finder-symbol fix.
- **Library management** — `.searchable`, mode/status/date filter chips, section grouping, context-menu Delete/Rename (+ backing store APIs); scope canvas selection to the active mode.

**Exit:** Audio and video play inline. Pulls and generations show determinate
progress. Drop and paste work. Read Image works on first launch. The library is
searchable and manageable.

### Phase 3 — Conversational depth and CLI parity _(5–7 weeks)_
Turn shallow modes into full experiences; surface the missing CLI families.
- **Multi-turn chat/code + vision + tools** — app-side conversation model (CLI is stateless): user/assistant bubbles, persistent composer, New chat, per-message copy/retry/edit, re-send accumulated context; image attachment for vision-chat; expose `--tools/--tool-loop/--allow-shell-exec/--thinking` in Advanced.
- **Voice cloning** — Speak gets a profile picker (`speech profile list`), style/clone toggle, ref-audio attach + save-as-profile; wire `mode/profile/refAudio/refText/saveProfile/language` into the Advanced template.
- **Option-coverage parity** — one shared per-mode option schema rendered in both surfaces (chat temp/max-tokens, image CFG/strength, transcription language/timestamps/backend, video duration/fps/frames/variant/end-image); make Advanced responsive + synced to the active mode.
- **Missing families** — `sfx generate`/`sfx video` as Studio modes (+ `ae/clap/condition` Advanced); `music analyze` (Advanced) and scoped `music realtime`; `config` (HF token) in Settings; `status --json` as a live status pill; Advanced templates for `plugin`/`open-webui`/`benchmark`.

**Exit:** chat keeps context with history UI + vision; voice cloning end-to-end from Speak; Studio options have real depth; sfx/analyze/config/status usable from the GUI.

### Phase 4 — Native polish, accessibility, and distribution hardening _(4–5 weeks)_

> **Verified 2026-08-30.** Menus, `SceneStorage` restore, `UNUserNotificationCenter`
> completion notifications, Quick Look, the semantic light/dark token system, the
> first-run welcome, Sparkle with an EdDSA appcast, and the app-to-CLI version
> handshake are all implemented. The two outstanding items in this phase —
> opt-in MetricKit capture and Export Diagnostics — are now implemented as
> `StudioCrashReporter` (off until enabled in Settings, written locally, never
> transmitted) and `StudioDiagnostics` (Help → Export Diagnostics, secret-free).
> Phase 4 is closed.
- **Native affordances** — real menus (File/View/Help, New Window, restore window/tabs removed by replacing `.newItem`); SceneStorage state restore; `UNUserNotificationCenter` completion notifications; Quick Look.
- **Theming system** — semantic color tokens + a real light theme; themed Button/Toggle/Field styles; spacing/radius/type scales; `.ultraThinMaterial`/`NSVisualEffectView` for the chromeless title bar; one shared error/banner component (genuine failures in red, not warning yellow).
- **Accessibility:** Add `.accessibilityLabel` and `value` to all icon buttons
  and status pills. Announce readiness and run state, use relative text styles
  for Dynamic Type, and complete a VoiceOver pass.
- **Onboarding and lifecycle:** Add a first-run welcome and guided starter-model
  download through the CLI `guide`. Use Sparkle with an EdDSA appcast so the
  bundle and CLI update atomically. Add an app-to-CLI version handshake,
  opt-in MetricKit crash capture, and Export Diagnostics.

**Exit:** standard menus/windows/notifications/Quick Look; correct light+dark with Dynamic Type; VoiceOver pass; onboarding downloads a starter model; Sparkle delivers a signed update.

---

## Original key risks

- **Camera permission is subtle:** the usage string must live in the *app* bundle (TCC blames the parent), and the problem reproduces only on a fresh machine. A development machine can appear fixed while a release remains broken.
- **Bundle relocation** is coupled to `CLIResolver` paths and the framework search paths baked into the CLI/ds4 binaries; moving executables can break dylib loading. Test the relocated, signed bundle on a clean machine.
- **The concurrent-engine refactor** touches the controller that both UIs and
  all tests depend on. Land it incrementally behind existing behavior.
- **`--json`/`--quiet` coverage is ~14 commands today** — audit each subcommand before deleting `detectOutputURL` entirely; some modes may still need the path-line fallback.
- **Multi-turn threading lives in the app** (stateless CLI) — handle context-window growth / token budgeting or long chats silently truncate.
- **Two parallel UIs** are a standing drift liability; the shared per-mode option schema (Phase 3) is what prevents it — skipping it makes the inconsistency permanent.
- **Sandbox posture is deferred** (notarized-unsandboxed first). Mac App Store would require rearchitecting the `/usr/local/bin` installer, raw-string paths (→ security-scoped bookmarks), and localhost HTTP — an explicit product decision, not a late surprise.
