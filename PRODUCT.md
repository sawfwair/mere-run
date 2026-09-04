# Product

## Register

product

## Users

Developers, tinkerers, and creative professionals on Apple Silicon Macs who want to run
AI models locally: image, video, music, sound, speech, chat, code, vision, and
Earth observation — without a cloud account and without their prompts or media leaving
the machine. They range from CLI-fluent engineers (who also use the bundled `mere.run`
terminal command and the Command Console) to prompt-first creators who never open a
terminal. The Studio is the prompt-first face; the CLI is the engine underneath.

Context of use: a desktop creative/utility session on macOS 15+, often long-running
(model pulls are gigabytes, video/music renders take minutes). Light and dark
environments both matter; the app must be a first-class Mac citizen (menus, Quick Look,
drag-and-drop, VoiceOver, Dynamic Type).

## Product Purpose

mere.run Studio (`apps/macos/StudioKit`, `apps/macos/StudioUI`, and the
`apps/macos/MereRunStudio` executable) is the macOS front end for the
open-source mere.run local-inference CLI. Tagline: **"Create anything. Locally."** It exists so that
local-first inference feels as effortless as a hosted tool: pick a domain, type a prompt,
get a previewable result — while the public CLI does the real work and stays the source
of truth. The app is signed, notarized, and shipping; success now is that every capability
the CLI exposes has a place in the app, that the designed surfaces produce a usable inline
result, that state is never silent, and that it feels native rather than like a wrapped
web page or a terminal with buttons.

## Brand Personality

Warm, precise, quietly confident. A crafted instrument, not a console; a studio, not a
dashboard. Local-ness and privacy are stated plainly ("stays on this Mac") rather than
shouted. Copy is short, concrete, and a little literary in display moments ("Make
something visible." / "Score the moment.") but strictly functional in controls and
errors.

## Anti-references

- Generic AI-SaaS styling: neon-on-black, purple-blue gradients, glassmorphism cards,
  gradient text, hero-metric dashboards.
- Electron/web-wrapper feel: giant in-window branding headers, horizontally scrolling
  chip navigation, everything-in-a-card.
- Terminal-wrapper feel: raw log dumps as the primary surface, monospace everywhere,
  CLI flag names leaking into primary UI copy.
- Modal-first flows where inline or popover would do.

## Design Principles

1. **Native first.** Structure and behavior should read as macOS: a `NavigationSplitView`
   with sidebar navigation, real keyboard support and menu commands, Quick Look, drag out
   as well as in. The window toolbar is deliberately empty so the columns run to the top
   of the window; the domain header lives in the content column instead.
2. **One navigation.** Every capability is a peer, reached the same way: a domain in the
   sidebar and a task in its control. Nothing is a modal detour, and nothing lives in a
   second navigation system.
3. **Prompt-first, depth on demand.** The default surface is one field and one button;
   every extra control must earn its place. Full CLI depth lives one deliberate step away
   — the inspector, then the Command view, then the Command Console — and is rendered from
   the capability contract, so it never drifts from the CLI.
4. **State is never silent.** Readiness, queueing, trimming, progress, and failure are
   always visible and always explained in words, never color alone. Work in flight is
   listed in one place, the Activity popover, whatever page you are on.
5. **Results are files.** A run writes a named file to a folder a person can find, and
   the app says where. The library is a view onto those files, not a vault.
6. **Local is a feature.** Surface the on-device story: model store, server status,
   "stays on this Mac" — as calm facts, not marketing.
7. **Warmth over chrome.** The paper-and-bronze palette, serif display moments, and a
   few earned signature details (waveforms, streaming caret) carry the identity; the
   rest stays quiet.

## Accessibility & Inclusion

- VoiceOver labels and values for stateful and icon-only controls; selection carries the
  `isSelected` trait, headers carry `isHeader`, and color is never the sole carrier of
  state — a status dot always has its phrase, a banner always has its severity word.
- Dynamic Type via relative font tokens; layouts tolerate Larger Text.
- Reduce Motion is honored where motion is structural: the shell's column and banner
  transitions, the feed's card entrances, and the Models page's indeterminate progress
  sweep. Hover and press animations are not yet conditioned on it — a known gap.
- Contrast target ≈ WCAG AA in both light and dark themes.
