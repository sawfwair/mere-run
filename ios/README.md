# mere.run Studio for iOS

The iOS client for mere.run. It can sign in to a hosted relay or pair directly
with a machine, submit and watch work, fetch digest-verified artifacts, and
inspect fleet nodes. Supported image and chat models can also execute entirely
on a physical iPhone. The architecture and current boundaries live in
[`docs/ios-studio.md`](../docs/ios-studio.md).

## Status

Active client. The local Xcode project is generated from `project.yml`; CI
regenerates it, performs a Release simulator build, and verifies the app/widget
version and Debug/Release associated-domain contracts. Simulator builds cover
the UI, portable client, and background-session integration. Actual suspended
multi-gigabyte transfer, MLX inference and memory release, browser sign-in, and
direct-machine pairing still require physical-device validation.

## Building

Requires Xcode 16+ on macOS and [XcodeGen](https://github.com/yonaskolb/XcodeGen):

```bash
brew install xcodegen
cd ios
xcodegen generate                              # simulator builds
MERERUN_IOS_TEAM=<team id> xcodegen generate   # device builds (automatic signing)
open MereRunStudio.xcodeproj
```

The app target links `MereRunRelayKit` for portable relay workflows and
`MereRunCore` for on-device inference. Xcode resolves the root package graph on
first open.

## Layout

```text
MereRunStudio/
  App/       app entry and the RelayStore observable (profiles, auth, client)
  Theme/     paper-and-bronze tokens ported from the macOS Studio theme
  Views/     pairing, fleet, run inbox, and run detail surfaces
```

Storage: the executor profile persists in the app's protected Application
Support directory using the CLI-compatible `executors.json` shape. OAuth and
direct-pairing credentials live in the device-only Keychain. Existing
file-backed OAuth credentials migrate to Keychain once and are removed.

Browser sign-in uses Authorization Code + PKCE and returns through the
`mere.world` universal link. Device-code authorization remains the browserless
fallback.

## Model downloads and runtime residency

On-device model installs use a fixed iOS background URL session. The app stores
pending install intent in `UserDefaults`, rejoins the matching system transfer
after relaunch, moves completed files into durable staging, and only removes
the pending record after the managed snapshot is pinned and validated. Leaving
the app does not cancel an active payload transfer. New transfers default to
unmetered, unconstrained Wi-Fi; Settings can opt a new download into cellular,
expensive, and constrained networks. The selected policy is persisted with that
pending download across relaunch. User force-quit behavior is owned by iOS and
is not a supported continuation path.

The local engine unloads chat and image runtime state when the app backgrounds,
on memory warnings, on model or inference-lane switches, before model removal,
and after two idle minutes. Active inference is allowed to finish before a
requested unload begins, and new inference is rejected during the release
transition. Live Activities use one bounded poller per run with exponential
backoff and stop after persistent transport failures. Artifact refreshes are
staged and validated before replacing an earlier verified local copy.

## Release footprint

The 2026-08-19 clean arm64 Release-simulator audit measured a 75.4 MiB
uncompressed app and a 68.3 MiB main executable after keeping the unused
llama.cpp binary dependency macOS-only. That is not an App Store download-size
estimate. It is a reproducible signal that the current broad `MereRunCore`
import dominates the mobile build. CI keeps the simulator build arm64-only and
fails if the Release executable exceeds 80 MiB; a future mobile runtime target
must actually extract the needed image, chat, and model-storage sources rather
than wrap the existing core dependency.
