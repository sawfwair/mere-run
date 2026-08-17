# mere.run Studio for iOS

The iOS client for a paired mere.run fleet. The phone is a relay client in the
same family as hosted Graph Studio: it signs in to a relay, watches the run
inbox, streams worker events, fetches digest-verified artifacts, and inspects
fleet nodes. Model execution stays on the paired nodes; nothing infers on the
phone in this phase. The architecture and phasing live in
[`docs/ios-studio.md`](../docs/ios-studio.md).

## Status

Scaffold. The sources here compile against `MereRunRelayKit` (the portable
relay client extracted from the CLI) but have not yet been built or run on a
Mac with Xcode — this directory was authored in a Linux environment where the
`MereRunRelayKit` package target and its tests build and pass, and where iOS
toolchains do not exist. Treat the first `xcodegen generate && xcodebuild`
on a Mac as part of landing this change.

## Building

Requires Xcode 16+ on macOS and [XcodeGen](https://github.com/yonaskolb/XcodeGen):

```bash
brew install xcodegen
cd ios
xcodegen generate                              # simulator builds
MERERUN_IOS_TEAM=<team id> xcodegen generate   # device builds (automatic signing)
open MereRunStudio.xcodeproj
```

The app target links only the `MereRunRelayKit` product from the repository's
root package. Xcode will resolve the whole package graph on first open (the
root package also declares the runtime's dependencies); only RelayKit's small
closure is compiled into the app.

## Layout

```text
MereRunStudio/
  App/       app entry and the RelayStore observable (profiles, auth, client)
  Theme/     paper-and-bronze tokens ported from the macOS Studio theme
  Views/     pairing, fleet, run inbox, and run detail surfaces
```

Storage: the executor profile and OAuth token set persist in the app's
Application Support directory with complete file protection, using the same
JSON shapes as the CLI (`executors.json`, `executor-auth/<profile>.json`).
Keychain-backed credential storage and an Authorization Code + PKCE sign-in
(matching hosted Studio) are planned follow-ups tracked in the design doc.
