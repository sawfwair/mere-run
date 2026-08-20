# Raycast integration

Raycast is one example client of MereRun's public macOS handoff surface. Read
[macOS deep links](./macos-deep-links.md) for the app-owned route contracts and
security boundaries independent of any launcher.

Generate an image, video, music track, or spoken-audio file from Raycast, then
open the completed local artifact as the selected item in MereRun Library. The
Raycast extension calls the public `mere.run` CLI, writes a small typed receipt,
and hands that receipt to the macOS app through `mererun://library/import`.

The extension is a separate macOS launcher integration. It is not a MereRun
executable plugin and is not installed through the MereRun plugin catalog.

## Requirements

- macOS with [Raycast](https://www.raycast.com/) installed.
- A MereRun.app build containing the `mererun://library/import` route. Version
  0.31.0 supports the optional preview-only route but not Library import.
- The app-bundled `mere.run` helper, or another executable CLI path configured
  in the extension.
- A locally installed model for the generation command you want to run.

The native import and preview deep links are part of the macOS app. Linux
packages contain the headless CLI only and cannot open the MereRun Studio.

## Install the extension

Load the development extension from its public repository:

```bash
git clone https://github.com/sawfwair/mere-run-raycast.git
cd mere-run-raycast
npm ci
npm run dev
```

Keep `npm run dev` running while using the development extension. Raycast shows
a development warning badge beside locally loaded commands; that badge does not
mean local generation failed.

To remove the development copy, stop the process with Control-C and use
Raycast's **Uninstall Extension** action.

## Commands

| Raycast command | Search phrase | CLI operation | Output |
| --- | --- | --- | --- |
| **Generate Image** | `mere image` | `mere.run image generate` | PNG |
| **Generate Video** | `mere video` | `mere.run video generate` | MP4 |
| **Generate Music** | `mere music` | `mere.run music generate` | WAV |
| **Synthesize Speech** | `mere speech` | `mere.run speech synthesize` | WAV |

Choose a command, enter its prompt in Raycast, and press Return. Raycast shows
the CLI's most recent diagnostic while generation is active. After the command exits
successfully and the output is readable, the extension imports the artifact,
opens its owning MereRun workspace, reveals Library, and selects the imported row.

Generated files are durable. By default they are written to
`~/MereRun/Raycast` with a media-specific name and UTC timestamp, for example:

```text
mere-image-2026-08-01T22-42-07-653Z.png
mere-speech-2026-08-01T22-46-28-054Z.wav
```

## Configure the extension

Open Raycast's extension preferences for **Mere.run** to change either setting:

- **MereRun CLI Path:** Optional absolute path to a `mere.run` executable.
- **Output Directory:** Artifact directory, defaulting to
  `~/MereRun/Raycast`. Tilde paths are supported.
- **Open Result In:** **MereRun Library** by default, or **Quick Look Only**
  when the artifact should not be imported.

When no CLI path is configured, the extension selects the first executable in
this order:

1. `/Applications/MereRun.app/Contents/Helpers/mere.run`
2. `/usr/local/bin/mere.run`
3. `/opt/homebrew/bin/mere.run`
4. `~/.local/bin/mere.run`

The app-bundled helper is preferred so the Studio and CLI stay on the same
release.

## Library import contract

Raycast writes receipts beneath its private extension support directory. A
version 1 receipt contains only the completed local run metadata MereRun needs:

```json
{
  "version": 1,
  "id": "5cb7dc90-a9d4-4634-a486-0b8140226b42",
  "source": "raycast",
  "kind": "image",
  "prompt": "a ceramic coffee mug in soft morning light",
  "artifactPath": "/Users/dana/MereRun/Raycast/mere-image-result.png",
  "createdAt": "2026-08-01T22:42:07Z"
}
```

It then asks Launch Services to open one percent-encoded absolute receipt path:

```text
mererun://library/import?receipt=%2FUsers%2Fdana%2FLibrary%2FApplication%20Support%2Fcom.raycast.macos%2Fextensions%2Fmere-run%2Flibrary-imports%2F5cb7dc90-a9d4-4634-a486-0b8140226b42.json
```

MereRun accepts receipt version 1 and the typed `image`, `video`, `music`, and
`speech` kinds. It rejects oversized or malformed receipts, unsupported
versions, empty prompts, missing or unreadable artifacts, and media that does
not match the declared kind. Reopening the same receipt or artifact selects the
existing row instead of duplicating it. MereRun owns `library.json`; the
launcher does not read or write that file.

## Preview an existing artifact

MereRun.app registers `mererun://preview`. A launcher can pass one
percent-encoded absolute path:

```text
mererun://preview?path=%2FUsers%2Fdana%2FDesktop%2Fresult.png
```

Raycast extensions should let the URL API encode the path:

```typescript
import { open } from "@raycast/api";

const deepLink = new URL("mererun://preview");
deepLink.searchParams.set("path", outputPath);
await open(deepLink.toString());
```

The target must already exist and be a readable file. MereRun rejects relative
paths, directories, missing files, duplicate `path` values, and extra query
parameters. Opening a preview does not import, move, or modify the artifact.

## Local and security boundaries

- Prompts and generated artifacts remain local unless another tool moves or
  uploads them.
- The extension launches `mere.run` directly without a shell, so prompt text is
  not interpolated into a shell command.
- Library import accepts one local receipt no larger than 256 KiB and one
  existing readable artifact whose type matches the receipt.
- Receipt IDs cannot replace an existing Library row that points elsewhere.
- A preview link can open only one existing, readable local file.
- The extension does not run inside MereRun and cannot add runtime providers or
  graph nodes.

## Troubleshooting

### The commands do not appear in Raycast

Confirm `npm run dev` is still running in the `mere-run-raycast` checkout. If
the extension was removed, run the command again to reload the development
copy.

### Raycast cannot find the CLI

Install MereRun.app in `/Applications`, install the terminal command from
Studio Settings, or set **MereRun CLI Path** to an executable absolute path in
the extension preferences.

### Generation reports a missing model

Open **Models** in MereRun Studio and download the required model, or inspect
the available managed models with:

```bash
mere.run model list
```

### Generation finishes but the Library item does not open

Confirm that the installed MereRun.app includes `mererun://library/import`, and
that both the receipt and generated artifact still exist. Choose **Quick Look
Only** temporarily to isolate generation from Library import. Exercise the
preview handler independently with a small text file:

```bash
printf 'MereRun preview smoke\n' >/tmp/mere-preview.txt
open 'mererun://preview?path=%2Ftmp%2Fmere-preview.txt'
```

If the direct link works, inspect Raycast's toast for the output path and verify
that the extension's configured output directory is writable.

## Validate an extension checkout

The extension repository owns its TypeScript checks, command mapping tests, and
Raycast build:

```bash
npm ci
npm run check
```

MereRun owns the CLI behavior, both typed deep-link routes, Library persistence,
and the signed macOS release. See [Getting started](./getting-started.md) for
installation and [Model management](./runtime/model-management.md) for managed
downloads.
