# MereRun macOS Studio — Review & Roadmap to v1.0

_Verified multi-agent review of `Sources/MereRunApp` (~6,500 LOC) against the CLI source of truth (`Sources/MereRunCLI`) and the bundling pipeline. Findings below were adversarially re-checked against source; refuted/adjusted claims were dropped or down-graded._

> **Implementation update (2026-07-27):** This document preserves the original
> pre-parity audit and sequencing rationale. The capability gaps described below
> are no longer the current product state. The app now consumes the shared
> machine-readable 88-command capability contract and has executable drift tests
> for every local Advanced template. Studio provides typed Text, Image, Video,
> Music, Speech, SFX, Vision/VFX, adapter, run, world, setup, model, plugin,
> benchmark, Open WebUI, and API workflows; Graph Studio and Node remain explicit
> external boundaries. Structured receipts, file pickers, validation, retry and
> resume controls are implemented. See
> [`Sources/MereRunApp/README.md`](../Sources/MereRunApp/README.md) for the
> current surface. Distribution validation is tracked separately and the
> historical release blockers below remain applicable until an installed,
> Developer ID-signed and notarized build is proven.

## Verdict

The app is **not 5% finished — it's a solid ~45% v0 skeleton** with misleading breadth and shallow depth. The bones are real (a clean `@MainActor` controller, an injectable process layer, a 34-command catalog, a persisted library, a capable Models sheet, ~1,000 LOC of tests). What's missing is concentrated exactly where "finished" lives: it **can't ship** (ad-hoc signed, no notarization, a guaranteed camera-permission crash), and core modes **dead-end** (audio/video show an icon, chat is one-shot, downloads show a wall of logs).

### Dimension scorecard

| Dimension | State | Headline problem |
|---|---|---|
| Architecture & orchestration | **55%** | Output detected by scanning stdout for paths that `fileExists`, not the CLI's `--json`/`--quiet` contract; engine hard-capped at one run; UTF-8 chunks silently dropped |
| UX & interaction | **45%** | Speak/Music/Video dead-end on an icon; chat single-shot; spinner-only progress; library has no search/delete; Read Image opens on a blocked action |
| Feature parity vs CLI | **62%** | `sfx`, `music realtime`/`analyze`, `config`, `status`, `plugin`, `open-webui`, `benchmark`, `guide` have zero GUI; chat omits tools/vision; voice cloning unwired |
| Native platform | **28%** | Ad-hoc signed, no entitlements, no `NSCameraUsageDescription` while shipping `vision track-live` (TCC kill); child processes orphaned on quit; hardcoded dark; no a11y |
| Visual & polish | **55%** | Strong identity, but 1210px Advanced panel jammed in a 560px viewport; invalid `Image(systemName: "finder")`; no semantic tokens; native controls clash in light mode |
| Build & distribution | **30%** | Hand-rolled bundle, executables under `Contents/Resources` (notarization-fatal), no DMG/notarization step despite README promising one, no macOS CI |

---

## Ship-blockers (must fix before any distribution)

1. **Camera TCC crash.** `vision track-live` → `SAM31CameraCapture` opens `AVCaptureDevice`, but the generated `Info.plist` has no `NSCameraUsageDescription`. Because the CLI runs as a child of `MereRun.app`, TCC attributes camera access to the app bundle → process is terminated with no prompt. _(scripts/build_mere_run_app.sh, VisionTrackLiveCommand.swift:79)_
2. **Gatekeeper rejection.** `codesign --force --sign -` (ad-hoc), no hardened runtime, no entitlements, no `notarytool`/`stapler`. Any downloaded copy gets the "damaged / can't be opened" error. _(scripts/build_mere_run_app.sh:94)_
3. **Notarization-fatal layout.** The CLI binary, `llama.framework`, `mlx` bundle, and `vendor/ds4/ds4-server` are copied into `Contents/Resources/`. `notarytool` rejects executable Mach-O under Resources.
4. **Orphaned processes.** No `applicationShouldTerminate` hook — quitting mid-run (especially `api serve`) leaves the child CLI running. `terminate()` only fires on explicit Stop.
5. **README lie.** README advertises a "signed and notarized macOS DMG"; no pipeline produces a DMG at all.

---

## Quick wins (independent, low-risk, land immediately)

- **Read Image default:** change `readImageAction` default/reset from `.inspect` to `.ocr`, or register the auto-download VLM in the capability catalog — the mode currently opens on a permanently blocked action. _(StudioTypes.swift:190,202)_
- **Invalid SF Symbol:** `Image(systemName: "finder")` is not a real symbol; renders as a missing glyph in 3 places. Use `magnifyingglass`/`folder`. _(StudioRootView.swift:733, MereRunRootView.swift:382, StudioModelsView.swift:403)_
- **`.preferredColorScheme(.dark)`** at the WindowGroup root so native pickers/steppers/sheets stop rendering light-on-dark against the custom dark panels.
- **UTF-8 data loss:** buffer raw `Data` per stream and retain incomplete trailing bytes instead of dropping whole chunks when a codepoint straddles a read. _(MereRunController.swift:267-277)_
- **Library error visibility:** publish a `lastPersistenceError` instead of the empty `catch` so silent history loss is detectable. _(StudioLibraryStore.swift:135-137)_
- **`.windowResizability(.contentMinSize)`** + explicit minimum frame so the prompt bar can't be clipped. _(MereRunApp.swift:14)_
- **Add `NSCameraUsageDescription`** to the plist insert block and hide `visionTrackLive` until permission is gated.
- **Correct the README** DMG claim until the pipeline produces a notarized artifact.

---

## Phased roadmap

### Phase 0 — Platform correctness & releasable pipeline _(3–4 wk · foundational, non-negotiable first)_
Make the bundle structurally valid, signable, notarizable, Gatekeeper-clean, crash-free on permissions, and CI-built.
- **TCC & camera safety** — add usage strings; gate `track-live` behind `AVCaptureDevice.requestAccess` before launch.
- **Signing & notarization** — `MereRun.entitlements` (notarized-unsandboxed first); Developer ID + `--options runtime --timestamp`, inside-out signing; `scripts/package-macos.sh` (DMG → `notarytool submit --wait` → `stapler staple`).
- **Bundle layout** — frameworks → `Contents/Frameworks`, helper executables → `Contents/MacOS`/`Helpers`; update `CLIResolver.bundledCandidates()`.
- **macOS packaging** — `scripts/package-macos.sh`; per-PR `build_mere_run_app.sh debug` job; version from git tag + commit count.
- **Process lifecycle** — `applicationShouldTerminate` kills all children; UTF-8 fix; README correction.

**Exit:** downloaded DMG opens clean (`spctl --assess` passes); `track-live` prompts for camera; `notarytool` accepts; quit kills all children; CI publishes a notarized DMG.

> ⚠️ Start Developer ID cert + `notarytool` credential provisioning on **day one** — it's an external dependency that gates the whole pipeline and only reproduces on a clean machine.

### Phase 1 — Run engine & CLI-contract refactor _(4–6 wk · architectural foundation)_
Replace heuristic orchestration and the single-run ceiling so later features sit on contracts, not guesses.
- **Structured result channel** — consume `--quiet` path lines / `--json` (already emitted by ~14 commands) instead of `detectOutputURL`'s last-40-line `fileExists` scan; decode into typed `RunResult`/artifacts; collapse the dual poll/per-chunk triggers into one debounced off-main path.
- **Per-run session model** — `RunSession` keyed by id (own process, log buffer, status, immutable request) behind a coordinator with N concurrent slots; lift `guard !isRunning`; cancel/remove/reorder queued items; per-run logs (the shared `logs` is wiped every run).
- **Decouple editing vs running** — `startRun` consumes `request.template/draft`; per-surface editing drafts so a dequeuing run never clobbers Advanced input.
- **State ownership & testability** — runtime server params (host/port/key) become dedicated persisted state (Models sheet currently piggybacks the transient command draft); inject CLI-locator + filesystem abstraction; URLProtocol stub + real-binary runner tests.

**Exit:** artifacts resolved from the CLI contract (incl. directory/multi-file); two modes run concurrently with isolated logs; editing never clobbered; Models load/unload targets the real runtime; new tests cover resolution/detection/runner/HTTP.

> Land the CLI contract and the session model as **separate incremental drops behind existing behavior** to avoid a long-lived branch on the controller everything depends on.

### Phase 2 — Headline media & progress UX _(3–4 wk · compounding value)_
Make every mode's output actually usable.
- **Inline playback** — AVKit `VideoPlayer` for `.mp4/.mov`; `AVAudioPlayer` transport (play/scrub/time/waveform) for `.wav/.mp3/.m4a`; extend `StudioOutputFileKind.classify` for audio/video.
- **Determinate progress** — parse ModelPull's `[id] NN% done/total (speed/s)` stderr lines into a progress model; `ProgressView(value:)` with bytes/speed/ETA; collapse repeated `\r` updates; same for step-based generation.
- **File ingestion** — `.dropDestination` on canvas + prompt for each mode's accepted types; paste-image for createImage/readImage; Read Image default fix; finder-symbol fix.
- **Library management** — `.searchable`, mode/status/date filter chips, section grouping, context-menu Delete/Rename (+ backing store APIs); scope canvas selection to the active mode.

**Exit:** audio/video play inline; pulls/generations show a real bar; drop/paste works; Read Image works on first launch; library is searchable & manageable.

### Phase 3 — Conversational depth & CLI parity _(5–7 wk)_
Turn shallow modes into full experiences; surface the missing CLI families.
- **Multi-turn chat/code + vision + tools** — app-side conversation model (CLI is stateless): user/assistant bubbles, persistent composer, New chat, per-message copy/retry/edit, re-send accumulated context; image attachment for vision-chat; expose `--tools/--tool-loop/--allow-shell-exec/--thinking` in Advanced.
- **Voice cloning** — Speak gets a profile picker (`speech profile list`), style/clone toggle, ref-audio attach + save-as-profile; wire `mode/profile/refAudio/refText/saveProfile/language` into the Advanced template.
- **Option-coverage parity** — one shared per-mode option schema rendered in both surfaces (chat temp/max-tokens, image CFG/strength, transcription language/timestamps/backend, video duration/fps/frames/variant/end-image); make Advanced responsive + synced to the active mode.
- **Missing families** — `sfx generate`/`sfx video` as Studio modes (+ `ae/clap/condition` Advanced); `music analyze` (Advanced) and scoped `music realtime`; `config` (HF token) in Settings; `status --json` as a live status pill; Advanced templates for `plugin`/`open-webui`/`benchmark`.

**Exit:** chat keeps context with history UI + vision; voice cloning end-to-end from Speak; Studio options have real depth; sfx/analyze/config/status usable from the GUI.

### Phase 4 — Native polish, accessibility & distribution hardening _(4–5 wk · ship v1.0)_
- **Native affordances** — real menus (File/View/Help, New Window, restore window/tabs removed by replacing `.newItem`); SceneStorage state restore; `UNUserNotificationCenter` completion notifications; Quick Look.
- **Theming system** — semantic color tokens + a real light theme; themed Button/Toggle/Field styles; spacing/radius/type scales; `.ultraThinMaterial`/`NSVisualEffectView` for the chromeless title bar; one shared error/banner component (genuine failures in red, not warning yellow).
- **Accessibility** — `.accessibilityLabel`/`value` on all icon buttons & status pills; announce readiness/run state (color-only today); relative text styles for Dynamic Type; full VoiceOver pass.
- **Onboarding & lifecycle** — first-run welcome + guided starter-model download (wire the CLI `guide`); Sparkle (EdDSA appcast) so bundle + CLI update atomically; app↔CLI version handshake; opt-in MetricKit crash capture + Export Diagnostics.

**Exit:** standard menus/windows/notifications/Quick Look; correct light+dark with Dynamic Type; VoiceOver pass; onboarding downloads a starter model; Sparkle delivers a signed update.

---

## Key risks

- **Camera permission is subtle:** the usage string must live in the *app* bundle (TCC blames the parent), and only reproduces on a fresh machine — easy to "fix" on the dev box and still ship broken.
- **Bundle relocation** is coupled to `CLIResolver` paths and the framework search paths baked into the CLI/ds4 binaries; moving executables can break dylib loading. Test the relocated, signed bundle on a clean machine.
- **The concurrent-engine refactor** touches the controller both UIs and all tests depend on — land it incrementally behind existing behavior.
- **`--json`/`--quiet` coverage is ~14 commands today** — audit each subcommand before deleting `detectOutputURL` entirely; some modes may still need the path-line fallback.
- **Multi-turn threading lives in the app** (stateless CLI) — handle context-window growth / token budgeting or long chats silently truncate.
- **Two parallel UIs** are a standing drift liability; the shared per-mode option schema (Phase 3) is what prevents it — skipping it makes the inconsistency permanent.
- **Sandbox posture is deferred** (notarized-unsandboxed first). Mac App Store would require rearchitecting the `/usr/local/bin` installer, raw-string paths (→ security-scoped bookmarks), and localhost HTTP — an explicit product decision, not a late surprise.
