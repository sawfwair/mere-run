# Apple apps

This directory groups the open-source Apple clients that ship from the public
mere.run source tree:

- `macos/` contains the optional SwiftUI Studio target, its SwiftPM tests, and
  app assets. It launches the public `mere.run` CLI and does not duplicate the
  inference runtime.
- `ios/` contains the XcodeGen-managed iOS Studio app, widget, unit and UI
  tests, declared entitlements, and simulator build metadata. It consumes
  `MereRunRelayKit` and selected on-device paths from `MereRunCore`.

Buildable source, tests, configuration declarations, and simulator CI belong in
this public repository. Developer certificates, provisioning profiles, export
options containing maintainer choices, notarization, signed archives, release
receipts, and store-upload automation belong in the private
`mere-run-release-tools` repository.

Start with [`macos/README.md`](macos/README.md) or
[`ios/README.md`](ios/README.md) for platform-specific setup.
