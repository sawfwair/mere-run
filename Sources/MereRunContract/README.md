# MereRunContract

`MereRunContract` is the shared, compile-time boundary between the `mere.run`
CLI and its user-interface shells. It describes stable command identifiers,
supported flags, typed choices, inputs, and outputs without importing a model
runtime into the app.

`CommandCapabilityContract.swift` owns the machine-readable catalog emitted by
`mere.run catalog --json`. The CLI remains the runtime source of truth; shells
use this module to build and validate commands instead of maintaining a second
copy of the command surface.

Each option may carry additive shell metadata: `default_value` (the CLI's
static ArgumentParser default, rendered as the CLI parses it), `group` (one of
`MereRunCapabilityOptionGroup`), `tier` (`essential`, `standard`, `expert`),
`range` (`min`, `max`, `step` for numeric options), and `depends_on` (another
flag on the same capability that must be set for this one to matter). The
prompt-mode capabilities populate all of them; other capabilities may leave
them `nil`. Long-running generation capabilities also declare `--receipt` and,
where the pipeline reports steps, `--progress-json`; the line shapes are
documented on `MereRunCapabilityCatalog.resultReceiptExample` and
`progressEventExample`.

Positional arguments also expose `repeatable`. Earlier v1 documents that omit
this field decode it as `false`. `capabilityPositionalsMatchArgumentParser`
checks argument order, names, cardinality, and required values. Narrow, documented
exceptions retain existing serialized keys and requirements enforced by command
validation rather than by the parser.

Each capability's `output` says what one successful run leaves behind. `kind`
describes the run with no destination named: `text` prints to stdout, `service`
runs until it is stopped, and `file` or `directory` always writes the artifact.
`flag` names the option that carries the destination path, and `optional` marks
the artifact a run writes only when that flag is passed — so a `text`
capability with a `flag` prints its result and additionally saves it on
request. Look `flag` up in `options` to learn whether it names a file or a
directory. `flag` stays `nil` only where the command chooses the location
itself; `capabilityFileOutputsDeclareADestinationFlag` lists those.

When extending the contract:

- Add only public CLI capabilities that a shell needs to discover or invoke.
- Only record a `default_value` when the CLI default is static; machine- or
  model-specific defaults stay `nil`. `capabilityOptionsMatchArgumentParser`
  checks recorded defaults, cardinality, choices, aliases, and Boolean inversions
  against the pinned parser metadata format.
- Prefer typed enums for bounded choices shared by more than one target.
- Preserve existing identifiers and serialized values. Bump the schema version
  when making a breaking catalog change.
- Update the contract, CLI catalog/help, and app command-generation tests
  together so drift fails locally.

Contract tests live in `Tests/MereRunContractTests`, CLI serialization and help
coverage in `Tests/MereRunCLITests`, and shell command-generation coverage in
`apps/macos/StudioKitTests`.

`CapabilityCatalogTests` walks the parser's complete public command tree. Every
leaf is cataloged or has a documented command exemption; every displayed option
group of a cataloged command is represented. Only the parser's built-in help and
version flags are omitted. Long aliases share an option group; positive and
negative Boolean forms each need a catalog entry. The adapter decodes version 0
of ArgumentParser's help metadata and fails if that format changes.

After changing options, regenerate the shell's derived constants with
`./scripts/update-studio-command-flags.sh`. Command documentation inventory checks
continue to read the live command declarations.
