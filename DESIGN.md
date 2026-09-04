# Design

Visual system for the mere.run macOS Studio (`apps/macos/MereRunStudio`). The source of truth
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
| accentSoft | `F0E7D2` | `39331F` | selection washes (Library row, active header toggle), user chat bubble |
| onAccent | `FFFFFF` | `1B160A` | glyphs and labels on a solid `accent` fill (selected sidebar row) |
| segmentedSelection | `FFFFFF` | `3A362E` | the raised, selected segment of a segmented control |
| wordmarkGreen | `2D6A4F` | `2D6A4F` | the wordmark's period only — brand, not status |
| hoverFill | black @ 6% | white @ 7% | transient pointer hover on rows and icon buttons |
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
- Wordmark: `mere` plus a `wordmarkGreen` period in **Caveat Medium** at 27pt
  (`MereRunTheme.Brand.font()`), the only place the face appears. The font ships in
  `Resources/Fonts` (OFL 1.1) and is registered for the process on first use; if that
  fails the wordmark falls back to the system serif at the same size.
- Eyebrows (`MereEyebrow`): 10.5pt semibold, uppercase, 0.06em tracking, `textMuted` —
  sidebar sections, Library days, panel groups.
- Shell sizes are literal: sidebar rows 13pt medium (semibold when selected) with 13pt
  glyphs; header title 15pt semibold over an 11.5pt medium `textMuted` subtitle on one
  line; segments 12pt (semibold when selected); Library titles 12.5pt medium with 11pt
  medium `textMuted` meta; footer 12pt medium `textSecondary`.
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
- `MereSegmentedControl` / `MereSegment` — the segmented control: a 2pt-padded
  `surfaceRaised` pill (radius 7) of 24pt segments with 12pt side padding (radius 5.5);
  the selected segment is `segmentedSelection` with a 1pt shadow. Used for the header's
  task control (`StudioTaskControl`, with a "More" menu segment when a domain has more
  than six tasks) and the Library scope. Never the native `.segmented` picker.
- `MereToolbarIconButton` — 28pt header toggles (radius 6): `accentSoft` tile and
  accent glyph while their panel is shown, `textSecondary` glyph otherwise.
- Status is one voice: the footer pill (32pt, radius 9, an 8pt dot, "Ready · 92 models",
  a chevron; `hoverFill` only while hovered or open) with a popover for detail — never
  rows of always-on pills. Dot colors: green ready/serving, yellow checking, red when the
  status probe never answers ("Server unreachable").
- Navigation: the grouped sidebar (Create / Converse / Understand / System), 212pt
  ideal, on the standard macOS sidebar material; the wordmark at its top (20pt leading),
  eyebrows for sections, 30pt rows (radius 9, 10pt side padding, 13pt glyph + label).
  The selected row is a solid `accent` pill with `onAccent` glyph and label, drawn by the
  row: the native `List` highlight is switched off so it looks the same whether or not
  the window is key. No horizontal chip rails.
- Content header (`StudioContentHeader`): the first row of the content column, beside the
  Library, 52pt on `background` with a hairline below — never the window toolbar, which
  stays empty so the Library column runs to the top of the window. Leading (18pt in):
  14pt accent domain glyph, title, one-line subtitle. Center: the task pill. Trailing
  (16pt in): Library, Inspector, and Command toggles. With the sidebar collapsed the first
  header leaves room for the traffic lights.
- Raw command form (the Command view column and the Command Console's middle pane):
  eyebrow groups from the capability contract, each row a monospaced 12pt medium
  `textSecondary` flag in a 168pt column beside the control the option's kind calls for,
  over a "Will run" block — the shell-quoted command wrapped at the column width on
  `surfaceRaised` (radius 6, 12/10 padding, 11.5pt mono), with Copy, Open in Terminal,
  and the accent Run. The Console adds a 52pt header per pane and a template catalog
  column that reuses the sidebar's row shape.
- Library column: 248pt on `background` with a right hairline, running from the window
  top to the bottom. Header ("Library" 13pt
  semibold, count 11pt muted, scope segments) at 14/14/8 padding; a 28pt capsule search
  field (12pt); day eyebrows; rows of a 40pt thumbnail (radius 6, or a `surfaceRaised`
  glyph tile), one-line title, and meta with an 8pt status dot while queued (yellow),
  running (accent), or failed (red). Rows are 6/8 padded, radius 9, `accentSoft` when
  selected, `hoverFill` on hover, in a 6pt-inset column with 2pt gaps.
- Chat (Converse): a 248pt thread list in place of the Library ("Threads" 13pt semibold, a
  compose icon button, a "Search threads" capsule, Today / Earlier eyebrows, 8/10-padded
  rows of a 12.5pt title over an 11pt muted "Model · time" meta, `accentSoft` when
  selected). The transcript is a 760pt centered column: a header (13.5pt semibold title,
  model and "System: default" chips, hairline below), then turns 20pt apart, bottom-aligned,
  22/24 padded. User turns are 14pt in `accentSoft` bubbles (radius 14/14/4/14, 10/14
  padding, max 520pt) on the right; assistant turns are unboxed 14pt Markdown at 1.55 line
  height behind a 16pt accent chat glyph (max 640pt), with fenced code on `surfaceRaised`
  (radius 8, a header row of the language eyebrow and Copy over a hairline, 12pt mono) and a
  row of 28pt Copy / Retry / Branch icon buttons beside "model · tok/s · time" in 11pt muted.
  A streaming turn ends in a thin accent caret.
- Media: audio renders as an interactive waveform (accent = played), video via AVKit in
  a rounded, shadowed frame.
