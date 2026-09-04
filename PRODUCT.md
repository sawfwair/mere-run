# Product

## Register

product

## Users

Developers, tinkerers, and creative professionals on Apple Silicon Macs who want to run
AI models locally: image, video, music, sound, speech, chat, code, and vision — without a
cloud account and without their prompts or media leaving the machine. They range from
CLI-fluent engineers (who also use the bundled `mere.run` terminal command and the
Advanced surface) to prompt-first creators who never open a terminal. The Studio is the
prompt-first face; the CLI is the engine underneath.

Context of use: a desktop creative/utility session on macOS 15+, often long-running
(model pulls are gigabytes, video/music renders take minutes). Light and dark
environments both matter; the app must be a first-class Mac citizen (menus, Quick Look,
drag-and-drop, VoiceOver, Dynamic Type).

## Product Purpose

mere.run Studio (`apps/macos/StudioKit`, `apps/macos/StudioUI`, `apps/macos/MereRunStudio`)
is the macOS front end for the open-source
mere.run local-inference CLI. Tagline: **"Create anything. Locally."** It exists so that
local-first inference feels as effortless as a hosted tool: pick a mode, type a prompt,
get a previewable result — while the public CLI does the real work and stays the source
of truth. Success is a signed, notarized v1.0 where every mode produces a usable inline
result, state is never silent, and the app feels native rather than like a wrapped web
page or a terminal with buttons.

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

1. **Native first.** Structure and behavior should read as macOS: sidebar navigation,
   thin material toolbars, real keyboard support, Quick Look, drag out as well as in.
2. **Prompt-first, depth on demand.** The default surface is one field and one button;
   every extra control must earn its place. Full CLI depth lives one deliberate step
   away (Options, Advanced) and never drifts from the CLI contract.
3. **State is never silent.** Readiness, queueing, trimming, progress, and failure are
   always visible and always explained in words, never color alone.
4. **Local is a feature.** Surface the on-device story: model store, server status,
   "stays on this Mac" — as calm facts, not marketing.
5. **Warmth over chrome.** The paper-and-bronze palette, serif display moments, and a
   few earned signature details (waveforms, streaming caret) carry the identity; the
   rest stays quiet.

## Accessibility & Inclusion

- VoiceOver labels/values for all stateful and icon-only controls (established in
  WS-4.5); color is never the sole carrier of state.
- Dynamic Type via relative font tokens; layouts tolerate Larger Text.
- Respect Reduce Motion (decorative animation only, degrade gracefully) and Reduce
  Transparency (NSVisualEffectView system fallbacks).
- Contrast target ≈ WCAG AA in both light and dark themes.
