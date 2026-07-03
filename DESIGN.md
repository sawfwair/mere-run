# Design

Visual system for the mere.run macOS Studio (`Sources/MereRunApp`). The source of truth
in code is `MereRunTheme.swift`; every color resolves dynamically for light and dark.

## Theme

Dual-appearance, warm-neutral. Light is a warm paper palette; dark is a warm charcoal.
Never pure `#000`/`#fff`; every neutral is tinted toward the bronze brand hue. Theme
choice follows the system appearance — the app never forces one.

## Color

| Token | Light | Dark | Role |
|---|---|---|---|
| background | `FAF8F3` | `171614` | window base |
| surface | `FFFFFF` | `23211D` | panels, fields |
| surfaceRaised | `F1EDE3` | `302D27` | selected/raised fills |
| border | `D8D2C6` | `4E493F` | hairlines (usually at 0.4–0.8 opacity) |
| textPrimary | `211C13` | `F1EDE3` | primary text |
| textSecondary | `5C564A` | `C9C1B3` | supporting text |
| textMuted | `8A8273` | `918A7C` | captions, metadata |
| accent | `9C7A2E` | `C9A65D` | bronze/gold brand accent |
| accentSoft | accent @ ~12% | accent @ ~16% | selection washes, user chat bubble |
| green / yellow / red | tuned per appearance | | success / caution / failure |

Color strategy: **Restrained** — tinted neutrals plus the single bronze accent (≤10% of
any surface). Status colors appear only as status. No gradients as decoration beyond the
existing faint background wash; no gradient text; no glassmorphism (material is used only
where macOS uses it: sidebar and title-bar regions).

## Typography

- UI: SF Pro via Dynamic-Type-relative tokens (`titleFont`, `sectionFont`, `bodyFont`,
  `captionFont`, `monoFont` for command/code).
- Display: New York (system serif) for empty-state headlines, welcome hero, and other
  short display moments only — never for controls or body copy. This serif-on-paper
  pairing is a core identity element.
- Hierarchy through weight + size contrast; body line length capped (~65–75ch → max
  widths ~560–680pt on text blocks).

## Shape & Depth

- Radius scale: 6 / 8 / 9 / 10 / 18 / 20 (`MereRunTheme.Radius`). Pills and the composer
  use the large end; fields and rows the small end.
- Borders are 1pt hairlines of `border` at reduced opacity; selection borders use
  `accent`.
- Shadows via `mereShadow` (warm-neutral in light, deep in dark) on floating elements
  only: composer, overlays, popovers. Flat surfaces stay flat.
- `merePanel()` = surface fill + hairline; the standard container. Avoid nesting panels.

## Spacing

`MereRunTheme.Spacing`: 8 / 10 / 14 / 18 / 22 / 28 / 32. Rhythm varies — canvases
breathe (24–32), rows and chips stay tight (8–12).

## Motion

- Ease-out only; 0.15–0.25s for state fades, `.snappy` / gentle springs for selection
  and entrance. No bounce/elastic overshoot on layout.
- SF Symbol effects (variableColor, bounce) mark live activity sparingly.
- Everything decorative degrades under Reduce Motion.

## Components

- `MereBanner` — the only inline notice component (info/warning/error).
- `MerePrimaryButtonStyle` — accent primary action; `.bordered` for secondary.
- `mereField()` — canonical text-field chrome.
- Status is one voice: a compact status cluster (dot + word) with a popover for detail —
  never rows of always-on pills.
- Navigation: grouped sidebar (Create / Converse / Voice / Vision) with the standard
  macOS translucent sidebar material; no horizontal chip rails.
- Chat: user turns in `accentSoft` bubbles right-aligned; assistant turns as unboxed
  text blocks with a small mode glyph; fenced code in mono panels with copy.
- Media: audio renders as an interactive waveform (accent = played), video via AVKit in
  a rounded, shadowed frame.
