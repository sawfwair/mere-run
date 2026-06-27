# MereRun macOS Studio — Completion Plan (v1.0)

This document enumerates **all remaining work** to take the MereRun macOS Studio
(`Sources/MereRunApp`) from its current state to a shippable, signed/notarized **v1.0**,
sequenced across a **26-week** delivery calendar.

It is the execution companion to [`docs/macos-studio-roadmap.md`](./docs/macos-studio-roadmap.md)
(the review + phased plan). Where the roadmap explains *why*, this document specifies
*what is left*, *in what order*, *with what acceptance criteria*.

---

## 1. Current state

Already landed in this branch (do **not** re-plan these — they are done and validated:
`swiftlint --strict` clean, full build, 867 tests pass, app bundle codesigns valid):

- **Phase 0 — Platform correctness & release pipeline (complete).** Camera TCC gate + usage
  strings, hardened-runtime `scripts/MereRun.entitlements`, notarization-safe bundle layout
  (executables in `Contents/Helpers`, none in `Contents/Resources`), `codesign --deep`,
  `scripts/package-macos.sh` (DMG → notarize → staple), `.github/workflows/macos-release.yml`,
  CI bundle smoke, process-cleanup delegate, incremental UTF-8 decoder, version-from-git,
  README correction, and the 8 quick wins.
- **Phase 2 — Media & progress (complete).** Inline AVKit video + AVAudioPlayer audio,
  determinate progress (bytes/speed/%), drag-and-drop, library search/delete/rename,
  mode-scoped canvas selection.
- **Phase 3 — partial.** `sfx` Studio mode + Advanced sfx templates; voice-cloning flags on
  `speech synthesize` (Advanced).
- **Phase 4 — partial.** Completion notifications, app↔CLI version handshake, HF-token
  setting, library-error banner, accessibility labels, Run/Help menu, first-run welcome.

Everything below is **remaining**.

---

## 2. Remaining workstreams

Effort key: **S** ≈ 1–2 days · **M** ≈ 3–5 days · **L** ≈ 1–2 weeks · **XL** ≈ 2–4 weeks.

### WS-1 — Run engine & CLI contract (Phase 1) · XL

The single-run, heuristic-output controller is the structural ceiling. This must precede
deep feature work that assumes concurrency and reliable results.

| ID | Task | Effort | Files | Acceptance |
|----|------|--------|-------|------------|
| 1.1 | **Structured result channel** — consume `--quiet` path lines / `--json` / `--json-output` (already emitted by ~14 commands) instead of `detectOutputURL`'s last-40-line `fileExists` scan; decode into a typed `RunResult`/artifact list. | L | `MereRunController.swift`, new `RunResult.swift` | Image/dir/multi-file outputs resolve from the CLI contract; `detectOutputURL` fallback only for commands without a contract; unit tests cover JSON + path-line + fallback. |
| 1.2 | **Per-run session model** — a `RunSession` keyed by id (own process handle, log buffer, status, immutable captured request) behind a coordinator with N concurrent slots. | XL | new `RunSession.swift`, `RunCoordinator.swift`, `MereRunController.swift` | Two modes (e.g. chat + image) run concurrently with isolated logs; queued items can be cancelled/reordered. |
| 1.3 | **Decouple editing vs running** — `startRun` consumes `request.template`/`request.draft`; per-surface editing drafts so a dequeuing run never clobbers Advanced input. | M | `MereRunController.swift`, `StudioRootView.swift`, `MereRunRootView.swift` | Editing in one surface is never overwritten by a run starting in the other (regression test). |
| 1.4 | **Per-run log buffers** replacing the single shared `logs` wiped each run; surfaces select which run's log to show. | M | `MereRunController.swift`, `StudioRootView.swift`, `MereRunRootView.swift` | Starting a run no longer erases a prior run's console; logs are per-session. |
| 1.5 | **Runtime-server param ownership** — move host/port/api-key of the running runtime into dedicated persisted controller state; stop deriving the Models-sheet endpoint from the transient command draft. | M | `MereRunController.swift`, `StudioModelsView.swift`, settings | Load/unload targets the actually-running runtime regardless of last command draft. |
| 1.6 | **Injected CLI-locator + filesystem abstraction** (mirroring `MereRunProcessRunning`) so resolution fallbacks and output detection are unit-testable; add a `URLProtocol` stub for the Models HTTP and a real-binary integration test for the runner. | M | `MereRunController.swift`, tests | CLI resolution, output detection, runner pipe/termination, and HTTP load/unload are covered by tests. |

### WS-2 — Conversational depth (Phase 3) · L

| ID | Task | Effort | Acceptance |
|----|------|--------|------------|
| 2.1 | **Multi-turn chat/code** — app-side conversation model (ordered turns per session; the CLI is stateless), user/assistant bubbles in the canvas, persistent composer, "New chat", per-message copy/retry/edit; re-send accumulated system+turns each run with context-window budgeting. | L | Chat/code keep context across turns with a message-history UI; long conversations truncate gracefully, not silently. |
| 2.2 | **Vision chat** — image attachment for vision-capable chat models (CLI `--image`) in Studio. | M | An image can be attached to a chat turn and is sent via `--image`. |
| 2.3 | **Agentic tool loop (Advanced)** — surface `--tools`/`--tool-loop`/`--allow-shell-exec`/`--sandbox-dir`/`--thinking` for `text chat`. | M | Tool-loop flags are reachable and correctly built in the Advanced editor. |
| 2.4 | **Studio voice-clone UI** — a profile picker listing saved profiles (`speech profile list`), style/clone toggle, reference-audio attach with save-as-profile, wired into the Speak mode (the Advanced flags already exist). | M | Voice cloning is reachable end-to-end from the Speak mode. |

### WS-3 — CLI feature parity (Phase 3) · L

| ID | Task | Effort | Acceptance |
|----|------|--------|------------|
| 3.1 | **`music analyze`** (Advanced, audio→JSON) and **`music realtime`** scoped as a dedicated interactive experience (live audio + control panel). | L | `music analyze` runs from Advanced; `music realtime` has a documented interactive surface. |
| 3.2 | **`sfx ae` / `sfx clap` / `sfx condition`** as Advanced templates. | M | The full sfx family is reachable from Advanced. |
| 3.3 | **`config`** (HF endpoint, other keys) surfaced in Settings (HF token already done); **`status --json`** as a live status pill in the Studio top bar (server up?, loaded model, installed count). | M | Status pill reflects `status --json`; config values are editable in Settings with masking. |
| 3.4 | **`plugin`** (list/install/doctor), **`open-webui` quickstart**, **`model benchmark`** as Advanced templates; wire **`guide`** into a help panel. | M | Each command is invokable from the GUI; guide powers in-app help. |
| 3.5 | **Shared per-mode option schema** rendered in both Studio and Advanced (single source of truth) — chat temperature/max-tokens, image CFG/strength, transcription language/timestamps/backend, video duration/fps/frames/variant/end-image. | L | Studio options expose real depth; no drift between surfaces (schema-driven). |
| 3.6 | **Responsive Advanced panel** — single resizable column when docked vs full 3-pane detached; pre-select the template matching the active Studio mode and share draft state. | L | The 1210px panel no longer scrolls inside the rail; "Advanced" deepens the current task. |

### WS-4 — Native polish & accessibility (Phase 4) · L

| ID | Task | Effort | Acceptance |
|----|------|--------|------------|
| 4.1 | **Semantic theme tokens + real light theme** — background/content/accent/success/warning/danger/separator backed by asset-catalog `Color(light:dark:)`; remove the forced `.preferredColorScheme(.dark)` once light is real. | L | App renders correctly in light and dark; native controls match. |
| 4.2 | **Themed control styles** — one `PrimaryButton`/`ToggleStyle`/`Field`; replace `.roundedBorder` in the runtime editor; spacing/radius/type scales on `MereRunTheme`; refactor inline `.system(size:)` and literal paddings. | M | Consistent themed controls app-wide; no native light-on-dark mismatch. |
| 4.3 | **Material/vibrancy** — `NSVisualEffectView`/`.ultraThinMaterial` for the chromeless title-bar region. | S | The window reads as native macOS depth. |
| 4.4 | **One error/notification banner component** across all three surfaces; genuine failures (non-zero exit) in red, not warning yellow; size sheets to content. | M | Errors are consistent and correctly colored everywhere. |
| 4.5 | **Accessibility pass** — `.accessibilityLabel`/`value` on remaining icon controls and status pills (color-only state announced); relative text styles for **Dynamic Type**; full VoiceOver pass through the prompt-first flow. | L | VoiceOver navigates the full flow; UI scales with Dynamic Type. |
| 4.6 | **Quick Look** (`QLPreviewPanel`/space) for selected outputs and library items. | M | Space-bar Quick Look works on outputs and library rows. |
| 4.7 | **Window/menu/restore** — File (New Window, Open…, Reveal Output) + View toggles (Library/Advanced/Models) menus driven by shared UI state; restore window/tab support; persist mode+draft via `SceneStorage`. | M | Standard menus work; relaunch restores the last mode and layout. |
| 4.8 | **Clipboard paste** — paste-image-from-clipboard for createImage/readImage with a drop-target highlight. | S | Pasting an image populates the attachment. |

### WS-5 — Distribution hardening (Phase 4) · M (+ credential-gated)

| ID | Task | Effort | Acceptance |
|----|------|--------|------------|
| 5.1 | **Developer ID signing run** — set `MERERUN_CODESIGN_IDENTITY` in CI; the scripts already sign with `--options runtime` + entitlements. *(Credential-gated.)* | S | `spctl --assess` passes on a downloaded DMG from a clean machine. |
| 5.2 | **Notarization** — `MACOS_*`/`MERERUN_NOTARY_*` GitHub secrets; `package-macos.sh` already submits + staples. *(Credential-gated.)* | S | `notarytool` accepts the bundle; stapled DMG opens with no Gatekeeper prompt. |
| 5.3 | **Auto-update** — Sparkle is **out of scope for this OSS repo** (AGENTS.md forbids hosted-service surfaces). Ship the app↔CLI version handshake (done) and document the maintainer-side update channel; if auto-update is wanted, it lives in the distribution layer. | S | Version mismatch is surfaced in-app; update channel documented. |
| 5.4 | **MetricKit diagnostics** — opt-in `MXCrashDiagnostic` capture + an "Export diagnostics" action bundling logs + app/CLI versions (no remote upload). | M | Users can export a diagnostics bundle for bug reports. |

### WS-6 — Cross-cutting (integration, QA, release) · L

| ID | Task | Effort | Acceptance |
|----|------|--------|------------|
| 6.1 | **Performance** — off-main output detection (done partially), large-library virtualization, image/preview cache eviction, memory profiling during long video/music renders. | M | No main-thread hitches; bounded memory on long runs. |
| 6.2 | **Test depth** — UI tests for the prompt-first flow; snapshot tests for canvas states (empty/running/output/readiness); coverage for the session engine and conversation model. | L | Critical flows have automated coverage; CI runs them. |
| 6.3 | **Docs** — user guide for the Studio, updated screenshots, CHANGELOG, and `docs/` cross-links. | M | Studio is documented for end users. |
| 6.4 | **Beta + release hardening** — TestFlight-style external beta (Developer ID DMG), crash triage, polish backlog burn-down, release checklist. | L | A signed/notarized v1.0 DMG ships with a clean Gatekeeper assessment. |

---

## 3. 26-week schedule

Two-week sprints. Phase 0 and Phase 2 are already complete, so the calendar covers the
remaining engine, parity, polish, distribution, and release-hardening work, including
integration, QA, beta, and buffer.

| Wk | Focus | Workstream items | Exit |
|----|-------|------------------|------|
| 1–2 | Engine: result contract | 1.1, 1.6 (locator/fs seams + tests) | Outputs resolved from CLI contract; new test seams in place. |
| 3–4 | Engine: session model | 1.2 (coordinator + N slots) | Two runs execute concurrently with isolated state. |
| 5–6 | Engine: state hygiene | 1.3, 1.4, 1.5 | Editing/running decoupled; per-run logs; runtime endpoint owned. |
| 7–8 | Chat depth | 2.1 (multi-turn), 2.2 (vision chat) | Conversation threading + vision attach shipped. |
| 9–10 | Speech + tools | 2.4 (voice-clone UI), 2.3 (tool loop) | Voice cloning end-to-end; agentic flags in Advanced. |
| 11–12 | Option schema + Advanced | 3.5, 3.6 | One option schema across surfaces; responsive Advanced synced to mode. |
| 13–14 | CLI parity I | 3.3 (config/status pill), 3.2 (sfx ae/clap/condition) | Status pill live; full sfx family in Advanced. |
| 15–16 | CLI parity II | 3.1 (music analyze/realtime), 3.4 (plugin/open-webui/benchmark/guide) | Remaining CLI families surfaced; guide-powered help. |
| 17–18 | Theming | 4.1 (tokens + light), 4.2 (control styles), 4.3 (material) | Correct light/dark; consistent themed controls. |
| 19–20 | Native + a11y | 4.4 (banner), 4.5 (accessibility + Dynamic Type), 4.6 (Quick Look), 4.7 (menus/restore), 4.8 (paste) | VoiceOver pass; menus/restore/Quick Look done. |
| 21 | Distribution | 5.1, 5.2 (Dev ID + notarization once secrets exist), 5.4 (diagnostics) | Notarized DMG validated on a clean machine. |
| 22 | Performance | 6.1 | No hitches; bounded memory under load. |
| 23–24 | QA + tests | 6.2, 6.3 | UI/snapshot tests green in CI; Studio documented. |
| 25 | Beta | 6.4 (external beta, crash triage) | Beta feedback triaged; release checklist drafted. |
| 26 | Release hardening + buffer | 6.4, backlog burn-down | Signed/notarized **v1.0** ships; Gatekeeper clean. |

> Sequencing rationale: the engine (WS-1) precedes everything because determinate progress,
> per-mode history, and multi-turn chat all depend on the structured result channel, per-run
> sessions, and decoupled editing state. Parity and polish compound on top; distribution +
> beta + hardening close the cycle. Pull accessibility/light-mode earlier if enterprise or
> accessibility requirements surface.

---

## 4. Credential / infrastructure-gated items

These cannot be completed from the codebase alone; the scripts/CI are written to activate
the moment the secrets exist:

1. **Developer ID Application certificate** → `MERERUN_CODESIGN_IDENTITY` (local + CI secret
   `MACOS_CERT_P12_BASE64` / `MACOS_CERT_PASSWORD`).
2. **notarytool credentials** → `MERERUN_NOTARY_APPLE_ID` / `MERERUN_NOTARY_PASSWORD`
   (app-specific) / `MERERUN_NOTARY_TEAM_ID` (or a stored `MERERUN_NOTARY_PROFILE`).
3. **GitHub Actions secrets** for `.github/workflows/macos-release.yml`.
4. **Auto-update hosting** (appcast + signing keys) if pursued — distribution-layer, out of
   scope for this OSS repo per `AGENTS.md`.

Provision the Developer ID certificate and notarytool credentials **first**; they gate the
entire release pipeline and only reproduce failures on a fresh machine, not the dev box.

---

## 5. Risks & mitigations

- **Engine refactor breadth (WS-1).** It touches the controller every surface and test
  depends on. *Mitigation:* land the result contract and the session model as separate
  incremental drops behind existing behavior; never a big-bang merge.
- **`--json`/`--quiet` coverage is ~14 commands.** *Mitigation:* audit each subcommand's
  output before deleting `detectOutputURL`; keep the path-line fallback for the rest.
- **Multi-turn context growth (WS-2.1).** The CLI is stateless; the app owns history.
  *Mitigation:* explicit token budgeting + visible truncation, not silent.
- **Bundle/dyld fragility (done, but watch on Dev ID).** The relocated, signed bundle must be
  validated on a clean machine — frameworks load via co-located `@executable_path`.
- **Light theme is a design task, not just code.** *Mitigation:* define semantic tokens first;
  a real light palette needs design sign-off before flipping off forced dark.
- **Two parallel UIs drift.** *Mitigation:* the shared option schema (3.5) is the structural
  fix — do it before adding more options.
- **Sandbox/MAS deferred.** If Mac App Store is ever a goal, the `/usr/local/bin` installer,
  raw-string file paths, and localhost HTTP need rearchitecting — an explicit product
  decision, not a late surprise.

---

## 6. Definition of Done (v1.0)

- Every mode produces a usable, inline-previewable result (no icon dead-ends).
- Chat/code maintain conversation context with a real message-history UI.
- Multiple runs execute concurrently with isolated logs; outputs resolve from the CLI
  contract, not filesystem guessing.
- Studio surfaces the high-value CLI families; options have real depth via one schema.
- The app is a first-class Mac citizen: light/dark, Dynamic Type, VoiceOver, standard menus,
  Quick Look, notifications, drag-drop/paste, window restore.
- A Developer ID-signed, hardened-runtime, **notarized** DMG opens with no Gatekeeper prompt
  on a clean machine, built and published by CI.
- `./scripts/check.sh` is green; UI/snapshot/unit coverage protects the critical flows.
