# MereRunContract

`MereRunContract` is the shared, compile-time boundary between the `mere.run`
CLI and its user-interface shells. It describes stable command identifiers,
supported flags, typed choices, inputs, and outputs without importing a model
runtime into the app.

`CommandCapabilityContract.swift` owns the machine-readable catalog emitted by
`mere.run catalog --json`. The CLI remains the runtime source of truth; shells
use this module to build and validate commands instead of maintaining a second
copy of the command surface.

When extending the contract:

- Add only public CLI capabilities that a shell needs to discover or invoke.
- Prefer typed enums for bounded choices shared by more than one target.
- Preserve existing identifiers and serialized values. Bump the schema version
  when making a breaking catalog change.
- Update the contract, CLI catalog/help, and app command-generation tests
  together so drift fails locally.

Contract tests live in `Tests/MereRunContractTests`, CLI serialization and help
coverage in `Tests/MereRunCLITests`, and shell command-generation coverage in
`Tests/MereRunAppTests`.
