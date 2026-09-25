# MereRunContract

`MereRunContract` is the shared, compile-time boundary between the `mere.run`
CLI and its user-interface shells. It describes stable command identifiers,
supported flags, typed choices, inputs, and outputs without importing a model
runtime into the app.

`CommandCapabilityContract.swift` defines the serialized schema and its decoding
compatibility. `CommandCapabilityCatalog.swift` assembles the ordered catalog
emitted by `mere.run catalog --json`. The CLI remains the runtime source of truth;
shells use this module to build and validate commands instead of maintaining a
second copy of the command surface.

`TextChatTokenBudget` owns chat output/context token bounds shared by Core and
Studio validation. Numeric capability ranges remain display hints; they do not
replace runtime constraints.

To change a capability, open its `CommandCapabilityCatalog+<family>.swift` file.
Model benchmark definitions have their own file. Shared receipt and progress
options belong to `CommandCapabilityCatalog+Options.swift`; typed text and video
choices belong to `CommandCapabilityChoices.swift`. Preserve the assembly order
and serialized fields when reorganizing definitions.

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

Each option also lists `aliases`: every other spelling ArgumentParser accepts
for it, long and short. The other polarity of an inverted Boolean is its own
option. `capabilityOptionsMatchArgumentParser` requires `flag` plus `aliases` to
equal the parser's names, because the invocation reader folds every spelling
into the canonical flag (see Runtime families).

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

## Runtime families

A capability that loads a model declares `routing`
(`CommandCapabilityRouting.swift`): the `families` of code paths that can run
it, each with its managed model ids and any `selectors` (flag values that pick
the family: a value such as `--backend qwen`, `["true"]` or `["false"]` for a
Boolean passed or omitted, or a flag that must be `absent`); the `model_flags`
whose value names the model, highest precedence first; `default_models` rules
for a blank model; and `excluded_models`, managed models a picker might offer
that can't run the command, each with a `reason`. `identified_models` lists
managed models whose family depends on the checkpoint the command finds
installed (an environment override root, or an id that falls back to another
install). The CLI's identifier answers for them first; without an answer, a
listed model keeps its family and an unlisted one is unidentified, so shells
ask `catalog resolve`. A named model normally has to agree with the selectors;
`selectors_override_model: true` says the selectors win instead, so a model
whose family's selectors fail runs the default the rules pick and the named
model draws a warning (speech transcribe swaps a Parakeet id for Qwen3-ASR
under `--task translate`). `routed_by_command: true` says the command's own
router has the last word over the declared rules (speech transcribe's language
routing): the CLI's gate runs it, and shells ask `catalog resolve`.
`listing_flags` are Boolean flags that make the command list something and exit
before it reads a model (`--list-devices`); with one passed nothing is checked,
and a model the command would refuse resolves as `unrouted`.
An excluded model with `severity: warning` is one the command accepts and
replaces with its default (`speech listen` runs Qwen3-ASR whatever `--model`
names): it resolves as the default and draws one warning. An id that is both
excluded and identified is refused unless the identifier finds a root the
command loads before it reads the id (`--checkpoints-root` in place of an
ACE-Step language model). A selector or default-rule condition names a flag's
presence, its `values`, a numeric `minimum`, or its `absent`ce.
Single-runtime commands declare one family. Commands without a model omit
`routing`. Each domain keeps its routing in
`CommandCapabilityCatalog+<Domain>Routing.swift`.

Per option, `families` lists the families that use it (absent: every family),
`ignored_by` the families that accept it without effect, and `family_rules` a
family's narrowed `values`, `default_value`, `range`, `required`, and
`max_count`, with a `severity` for values it accepts and replaces. Numeric
values match by number, so `0.0` meets a rule of `0`. A family in neither list
rejects the option, except that a value the command reads as not passed only
warns: an empty `string` value, a `list_separator` list with no item left
(`--stems ","`), or a blank value of an option marked `blank_reads_as_omitted`
(text chat `--image " "`). Such a value also counts as absent for routing.
`overridden_by` lists conditions under which the command replaces the option's
value with its default for every family (`--flow-edit` runs text-to-music
whatever `--task-type` says); a passed value then only warns. A rule on a family in `ignored_by` lists the only
`values` or `range` that family tolerates: a runtime that runs without the
option but refuses anything except its own value (`--vocal-language en` on
YuE2) warns for those and fails for the rest. Declare families as a
per-capability enum that conforms to `MereRunFamilyID`, and scope options with
its builders:
`.scoped(F.only(.wan, ignoredBy: [.ltx]), .rule(.wan, required: true))`. Keep
arithmetic, cross-option, and file-content checks in Core.

`CommandCapabilityInvocation.swift` reads argv after a command path the way
ArgumentParser does, and `MereRunCapabilityCatalog.capability(forCommandLine:)`
finds the capability. `CommandCapabilityScope.swift` holds the one resolver
(`resolveFamily`), the family's narrowed surface (`options(forFamily:)`), and
its `violations`, each an error or a warning with the sentence the CLI prints.
`resolutionReport` combines them into the `MereRunFamilyResolutionReport` that
the CLI's capability gate enforces and `mere.run catalog resolve --json` prints.
The resolver answers managed ids, defaults, and selectors itself; the CLI
passes an `identify` closure for aliases, local folders, and `identified_models`,
a `chooseDefault` closure for a default whose candidates span families (the
machine's pick for `geo tessera`), and a `routedFamily` closure for commands
whose own router decides by rules the contract only approximates (speech
transcribe's language routing). A routed family wins over the declared rules;
shells without these hooks keep the rules and ask `catalog resolve`, whose
report `MereRunFamilyResolutionReport.resolution(in:)` reads back and
`report(for:_:identify:)` completes for the rest of the command line. Values
compare through `MereRunCapabilityOption.reads(_:asOneOf:)`: numbers by value
and choices by any spelling `choice_spellings` declares (`ignores_case` for a
value the CLI trims and lowercases, `numeric` for one it parses as a number).
A choice the contract compares that ArgumentParser does not enumerate declares
its `choice_spellings`, even when the CLI takes it exactly as written.
`CommandCapabilityRoutingTests` checks every routed capability's structure and
each resolver branch; `CapabilityGateTests` runs generated cases for every
family and option through the CLI gate; and `MainBehaviourRegressionTests` runs
every command line recorded as running on main before the gate
(`Tests/MereRunCLITests/Fixtures/MainBehaviour/main-behaviour.json`) through the
gate, which must not refuse any of them.
