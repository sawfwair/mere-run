# Raycast Integration

Generate an image, video, music track, or spoken-audio file from Raycast, then
open the completed local artifact in MereRun's native Quick Look panel. The
Raycast extension calls the public `mere.run` CLI and hands the resulting file
to the macOS app through `mererun://preview`.

The extension is a separate macOS launcher integration. It is not a MereRun
executable plugin and is not installed through the MereRun plugin catalog.

## Requirements

- macOS with [Raycast](https://www.raycast.com/) installed;
- MereRun.app v0.31.0 or later installed on the Mac so the preview deep link is
  available;
- the app-bundled `mere.run` helper, or another executable CLI path configured
  in the extension;
- a locally installed model for the generation command you want to run.

The native preview deep link is part of the macOS app. Linux packages contain
the headless CLI only and cannot open the MereRun Quick Look panel.

## Install the extension

The extension is not in the Raycast Store yet. Load the development extension
from its public repository:

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
the CLI's latest diagnostic while generation is active. After the command exits
successfully and the output is readable, the extension opens the artifact in
MereRun Preview.

Generated files are durable. By default they are written to
`~/MereRun/Raycast` with a media-specific name and UTC timestamp, for example:

```text
mere-image-2026-08-01T22-42-07-653Z.png
mere-speech-2026-08-01T22-46-28-054Z.wav
```

## Configure the extension

Open Raycast's extension preferences for **Mere.run** to change either setting:

- **MereRun CLI Path** — optional absolute path to a `mere.run` executable;
- **Output Directory** — artifact directory, defaulting to
  `~/MereRun/Raycast`. Tilde paths are supported.

When no CLI path is configured, the extension selects the first executable in
this order:

1. `/Applications/MereRun.app/Contents/Helpers/mere.run`
2. `/usr/local/bin/mere.run`
3. `/opt/homebrew/bin/mere.run`
4. `~/.local/bin/mere.run`

The app-bundled helper is preferred so the Studio and CLI stay on the same
release.

## Preview an existing artifact

MereRun.app registers `mererun://preview`. A launcher can pass one
percent-encoded absolute path:

```text
mererun://preview?path=%2FUsers%2Fme%2FDesktop%2Fresult.png
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

### Generation finishes but Preview does not open

Confirm that `/Applications/MereRun.app` is v0.31.0 or later and that the output
file still exists. Exercise the app's deep-link handler independently with a
small text file:

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

MereRun owns the CLI behavior, the `mererun://preview` route, and the signed
macOS release. See [Getting Started](./getting-started.md) for installation and
[Model Management](./runtime/model-management.md) for managed downloads.
