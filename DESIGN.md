# Design

Visual system for the mere.run macOS Studio (`apps/macos/StudioUI`). The source of truth
in code is `StudioUI/MereRunTheme.swift`; every color resolves dynamically for light and
dark. The measurements are `StudioKit/StudioLayout.swift`, so a layout rule can be read
without SwiftUI.

## Boards

The Studio is designed as a set of **boards** — one approved 1440×900 mockup per surface, each
named for what it draws: **Main** (Image ▸ Generate: the feed, the composer, the inspector),
**Analyze** (Vision ▸ Find over one picture), **Converse** (Chat: thread list and transcript),
**Command** (the raw command form and its "Will run" block), **Library** (the column in list,
grid, and batch), **Models** (the Manage list and detail), **Realtime**, **Subjects**, and the
**Activity** popover. `StudioSnapshotTests` renders each of them offscreen at the same size, so
a change can be read against its board without launching the app.

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
| border | `D8D2C6` | `4E493F` | hairlines (0.4–0.8 opacity; 0.53 for the shell's rules) |
| textPrimary | `211C13` | `F1EDE3` | primary text |
| textSecondary | `5C564A` | `C9C1B3` | supporting text |
| textMuted | `8A8273` | `918A7C` | captions, metadata |
| accent | `9C7A2E` | `C9A65D` | bronze/gold brand accent |
| accentSoft | `F0E7D2` | `39331F` | selection washes (Library row, active header toggle), user chat bubble |
| onAccent | `FFFFFF` | `1B160A` | glyphs and labels on a solid `accent` fill (selected sidebar row) |
| segmentedSelection | `FFFFFF` | `3A362E` | the raised, selected segment of a segmented control |
| wordmarkGreen | `2D6A4F` | `2D6A4F` | the wordmark's period only — brand, not status; the one token that does not change with appearance |
| hoverFill | black @ 6% | white @ 7% | transient pointer hover on rows and icon buttons |
| green | `5E7A45` | `8EAA74` | ready, installed, success |
| yellow | `9C7520` | `D2A24E` | checking, queued, needs attention |
| red | `C2493B` | `D98072` | unreachable, failed |

Color strategy: **Restrained** — tinted neutrals plus the single bronze accent (≤10% of
any surface). Status colors appear only as status. Decoration is one faint diagonal wash behind
the prompt workspace (`surfaceRaised` 28% → `background` → `surface` 20%); no gradient text, no
glassmorphism. Material appears only where macOS supplies it: the `NavigationSplitView` sidebar
column. Everything else, the region under the hidden title bar included, paints `background`.

## Typography

- UI: SF Pro via Dynamic-Type-relative tokens (`titleFont` `.title` semibold, `sectionFont`
  `.subheadline` semibold, `bodyFont` `.body`, `captionFont` `.caption` medium, `monoFont`
  `.callout` monospaced for command and code).
- Display: New York (system serif) through `displayFont` (`.largeTitle` medium) and
  `displaySmallFont` (`.title2` medium), for empty-state headlines and other short display
  moments only — never for controls or body copy. This serif-on-paper pairing is a core
  identity element.
- Wordmark: `mere` plus a `wordmarkGreen` period in **Caveat Medium** at 27pt
  (`MereRunTheme.Brand.font()`), the only place the face appears.
  `StudioUI/Resources/Fonts/Caveat[wght].ttf` (OFL 1.1) ships as a StudioUI resource — the
  packaged app carries it in `MereRun_StudioUI.bundle` — and is registered with Core Text for
  the process from `MereRunApp.init()`, before the first window draws; if that fails the
  wordmark falls back to the system serif at the same size.
- Eyebrows (`MereEyebrow`): 10.5pt semibold, uppercase, 0.63pt kerning, `textMuted` —
  sidebar sections, Library days, panel groups.
- Shell sizes are literal: sidebar rows 13pt medium (semibold when selected) with 13pt
  glyphs; header title 15pt semibold over an 11.5pt medium `textMuted` subtitle on one
  line; segments 12pt (semibold when selected); Library titles 12.5pt medium with 11pt
  medium `textMuted` meta; the footer pill 12pt medium.
- Hierarchy through weight + size contrast; body line length capped (the feed's measure is
  640pt, a transcript's 760pt).

## Shape & Depth

- Radius scale: 6 / 8 / 9 / 10 / 12 / 18 / 20 (`MereRunTheme.Radius`: `sm`, `base`, `md`, `lg`,
  `popover`, `xl`, `xxl`). Pills and the composer use the large end; fields and rows the small end.
- Borders are 1pt hairlines of `border` at reduced opacity; selection borders use
  `accent`. `mereFocusRing()` is the focused-field treatment: a 1.5pt `accent` ring at 55% over
  a soft 16% glow.
- Shadows via `mereShadow` (warm-neutral in light, deep in dark) on floating elements
  only: composer, overlays, popovers. Flat surfaces stay flat.
- `merePanel()` = surface fill + hairline; the standard container. Avoid nesting panels.

## Spacing

`MereRunTheme.Spacing`: 8 / 10 / 14 / 18 / 22 / 28 / 32. Rhythm varies — canvases
breathe (24–32), rows and chips stay tight (8–12).

## Layout

`StudioLayoutPolicy` holds every shell measurement: window 1280×820 by default and never
below 768×520; sidebar 212pt ideal (200 minimum, 300 maximum, user-resizable); Library column
248pt; inspector 300pt; Command view 520pt; the feed's text measure 640pt; the content column
never narrower than 520pt. The Command Console window keeps its own three-pane split
(catalog 220/268/360, form 420/560/∞, log 320/440/∞).

## Motion

- Ease-out only: `quick` is 0.15s and `standard` 0.22s for state fades; `spring`
  (response 0.32, damping 0.86) and `gentleSpring` (0.5, 0.9) carry selection and entrance.
  No bounce or elastic overshoot on layout.
- The only SF Symbol animation is the audio player's play/pause replace transition. Live
  activity is carried by progress bars and the status dot, not by symbol effects.
- Reduce Motion is honored where motion is structural: the shell's column and banner
  transitions, the feed's card entrances and "New result" pill, and the Models page's
  indeterminate progress sweep, which becomes a static capsule. Pointer-hover and
  button-press animations are not conditioned on it.

## Components

- `MereBanner` — the only inline notice component. Three severities (info `accent`, warning
  `yellow`, error `red`), a tinted 12% fill inside a 35% hairline at radius 8, up to three
  lines of `captionFont` `textSecondary`, an optional dismiss, and a VoiceOver label that
  prefixes "Warning:" or "Error:" so severity is never color alone.
- `MerePrimaryButtonStyle` (`.merePrimary`) — the accent primary action: `bodyFont` semibold in
  `background` on a radius-6 `accent` fill, 28pt minimum height.
- `MereSecondaryButtonStyle` (`.mereSecondary`) — 11.5pt medium `textPrimary` on `surfaceRaised`
  inside a 60% hairline, radius 6, 26pt tall, lifting with `hoverFill`. The v2 surfaces use it
  everywhere; `.bordered` survives only in the tasks that still host their v1 forms.
- `MereIconButtonStyle` (`.mereIcon`) — a bare glyph in `textSecondary` that darkens and takes a
  radius-6 `hoverFill` backing on hover.
- `mereField()`, `mereHoverRow()`, `merePanel()`, `mereShadow()`, `mereFocusRing()` — the shared
  chrome modifiers. `mereMediaFrame()` (radius 10, hairline, deep shadow) frames media on the
  Analyze board.
- `MereEyebrow` — the uppercase section label.
- `MereSegmentedControl` / `MereSegment` — the segmented control: a 2pt-padded
  `surfaceRaised` pill (radius 7) of 24pt segments with 12pt side padding (radius 5.5);
  the selected segment is `segmentedSelection` with a 1pt shadow. Used for the header's
  task control and the Library scope. Never the native `.segmented` picker.
- `StudioTaskControl` — the task pill in the content header. Up to six tasks are all segments;
  past six it shows the first five and a "More" menu segment, which takes the selected treatment
  itself while the current task lives inside it (Vision, with ten tasks, is the only domain that
  reaches this today).
- `MereToolbarIconButton` — 28pt header toggles (radius 6) with a 14pt glyph: `accentSoft` tile
  and accent glyph while their panel is shown, `textSecondary` glyph otherwise.
- Status is one voice: the footer pill (radius 9, 32pt minimum, an 8pt dot, two lines —
  "Ready · 92 models" over "2 running" — and a chevron; `hoverFill` only while hovered or open)
  opening the Activity popover — never rows of always-on pills. Dot colors: green ready or
  serving, yellow checking, red when the status probe never answers ("Server unreachable").
  VoiceOver rejoins the two lines into one value.
- Activity popover — a 340pt panel the shell draws over the window from the bottom-left, one row
  per running or queued job with its progress and a stop control, over the app↔CLI version
  handshake; with nothing running it shows the local server, models root, and resolved CLI path
  in the same shape.
- Navigation: the grouped sidebar (Create / Converse / Understand / System) over fifteen
  domains, 212pt ideal, on the standard macOS sidebar material; the wordmark at its top (20pt
  leading), eyebrows for sections, 30pt rows (radius 9, 10pt side padding, 13pt glyph + label).
  The selected row is a solid `accent` pill with `onAccent` glyph and label, drawn by the
  row: the native `List` highlight is switched off so it looks the same whether or not the
  window is key. No horizontal chip rails.
- Content header (`StudioContentHeader`): the first row of the content column, beside the
  Library, 52pt on `background` with a hairline below — never the window toolbar, which
  stays empty so the Library column runs to the top of the window. Leading (18pt in):
  14pt accent domain glyph, title, one-line subtitle. Center: the task pill. Trailing
  (16pt in): Library, Inspector, and Command toggles. Side blocks are 220pt while the column
  is wide enough and shrink to their content otherwise, so the pill never truncates. With the
  sidebar collapsed the first header adds 112pt of leading inset for the traffic lights.
- Raw command form (the Command view column and the Command Console's middle pane):
  eyebrow groups from the capability contract, each row a monospaced 12pt medium
  `textSecondary` flag in a fixed column (150pt in the 520pt Command view, 168pt in the
  console) beside the control the option's kind calls for, over a "Will run" block — the
  shell-quoted command wrapped at the column width on `surfaceRaised` (radius 8, 12/10 padding,
  11.5pt mono), with Copy, Open in Terminal, and the accent Run. The Command view's header is
  44pt; the console gives each of its three panes a 52pt header and draws its catalog column in
  the sidebar's row shape.
- Library column: 248pt on `background` with a right hairline, running from the window
  top to the bottom. Header ("Library" 13pt semibold, count 11pt muted, scope segments) at
  14/14/8 padding; a 28pt capsule search field (12pt) beside a filter menu (kind, favorites
  only) and a list-or-grid toggle; day eyebrows; rows of a 40pt thumbnail (radius 6, or a
  `surfaceRaised` glyph tile), one-line title, and meta with an 8pt status dot while queued
  (yellow), running (accent), or failed (red). Rows are 6/8 padded, radius 9, `accentSoft` when
  selected, `hoverFill` on hover, in a 6pt-inset column with 2pt gaps, carrying a hover star and
  Quick Look and dragging out to Finder. Grid mode is three tiles across with 6pt gutters and
  the title on hover; selecting more than one row raises a batch bar.
- Chat (Converse): a 248pt thread list in place of the Library ("Threads" 13pt semibold, a
  compose icon button, a "Search threads" capsule, Today / Earlier eyebrows, 8/10-padded
  rows of a 12.5pt title over an 11pt muted "Model · time" meta, `accentSoft` when
  selected). The transcript is a 760pt centered column: a header (13.5pt semibold title,
  model and system-prompt chips, hairline below), then turns 20pt apart, bottom-aligned,
  22/24 padded. User turns are 14pt in `accentSoft` bubbles (radius 14/14/4/14, 10/14
  padding, max 520pt) on the right; assistant turns are unboxed 14pt Markdown at 3pt line
  spacing behind a 14pt accent chat glyph (max 640pt), with fenced code on `surfaceRaised`
  (radius 8, a header row of the language eyebrow and Copy over a hairline, 12pt mono) and a
  row of 28pt Copy / Retry / Branch icon buttons beside "model · tok/s · time" in 11pt muted.
  A streaming turn ends in a thin accent caret.
- Media: audio renders as an interactive waveform (accent = played) over a 46pt play glyph and
  monospaced-digit times; video plays through AVKit, framed with `mereMediaFrame()` on the
  Analyze board and left unframed in the feed.

## Accessibility

Color is never the only carrier of state: a status dot always has its phrase, a banner always
has its severity word, and a selected segment or sidebar row carries the `isSelected` trait.
Icon-only controls carry labels and help text; eyebrows and headers carry `isHeader`; Library
rows expose one combined label with arrow-key and Space navigation; a streaming chat turn is
marked as updating frequently. Type is Dynamic-Type-relative throughout.
