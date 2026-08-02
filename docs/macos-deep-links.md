# macOS Deep Links

MereRun.app exposes two typed local handoff routes through the `mererun://`
scheme. A launcher, automation, agent, or another macOS app can preview an
existing artifact or ask MereRun to validate and record a completed result in
Library. The caller does not need access to MereRun's private Library storage.

These routes belong to the macOS app. Linux releases contain the headless CLI
only and cannot open MereRun Studio.

## Preview an artifact

Open one existing readable file in MereRun's native Quick Look panel:

```text
mererun://preview?path=%2FUsers%2Fme%2FDesktop%2Fresult.png
```

Build the URL with a URL-components API so the absolute path is percent encoded
exactly once. From a shell, an already encoded link can be passed to Launch
Services with `open`:

```bash
open 'mererun://preview?path=%2FUsers%2Fme%2FDesktop%2Fresult.png'
```

The target may be an image, audio, video, text, or Quick Look-compatible 3D
file. MereRun rejects relative paths, directories, missing or unreadable files,
duplicate `path` values, and extra query parameters. Preview does not import,
move, or modify the artifact.

## Import a completed result into Library

`mererun://library/import` accepts one percent-encoded absolute path to a small
JSON receipt:

```text
mererun://library/import?receipt=%2FUsers%2Fme%2FLibrary%2FApplication%20Support%2FMyLauncher%2Fimports%2F5cb7dc90-a9d4-4634-a486-0b8140226b42.json
```

A version 1 receipt contains the completed local run metadata MereRun needs:

```json
{
  "version": 1,
  "id": "5cb7dc90-a9d4-4634-a486-0b8140226b42",
  "source": "raycast",
  "kind": "image",
  "prompt": "a ceramic coffee mug in soft morning light",
  "artifactPath": "/Users/me/MereRun/Artifacts/mere-image-result.png",
  "createdAt": "2026-08-01T22:42:07Z"
}
```

Version 1 currently accepts the `raycast` source and the typed `image`,
`video`, `music`, and `speech` kinds. The source vocabulary is intentionally
strict even though the deep-link transport is not coupled to Raycast. New
maintained clients should add a source case and contract tests before emitting
receipts under a new source name.

MereRun validates the receipt and referenced artifact, deduplicates repeated
handoffs, records the completed result through its own Library store, activates
the matching workspace, reveals Library, and selects the imported row. The
caller must never read or write `library.json` directly.

## Local and security boundaries

- Each route accepts exactly one named query parameter and one local file.
- Receipt files must be regular readable files no larger than 256 KiB.
- Receipt versions, sources, and artifact kinds are typed and fail closed.
- Imported artifacts must be existing readable files whose media type matches
  the declared kind.
- Receipt IDs cannot replace an existing Library row that points elsewhere.
- Prompts and artifacts remain local unless the calling tool separately moves
  or uploads them.

## Example clients

The public [Raycast integration](./raycast.md) demonstrates image, video, music,
and speech generation followed by Library import or preview. It is a separate
launcher extension, not a MereRun executable plugin, and is only one possible
client of the macOS handoff routes.
