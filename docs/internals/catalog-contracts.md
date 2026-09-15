# Catalog and contract ownership

Use this guide when you change model discovery, shell capability metadata, or
API request and response contracts. Start with the owner of the behavior you
need to change.

| Surface | Owner | Reading order |
| --- | --- | --- |
| Managed model discovery | `MereRunCore` | `ManagedModelCatalog.swift`, the matching `ManagedModelCatalog+*.swift` family file, then `ManagedModelSpec.swift`. |
| Model API capabilities | `MereRunCore` | `ManagedModelAPIProfile.swift`, then the profile assigned by the model's catalog definition. |
| Installed model validation | `MereRunCore` | `ManagedModelSpec+Validation.swift`, then the referenced family resource validator. |
| CLI and shell capability schema | `MereRunContract` | `CommandCapabilityContract.swift` and `CommandCapabilityChoices.swift`. |
| Command capability definitions | `MereRunContract` | `CommandCapabilityCatalog.swift`, then `CommandCapabilityCatalog+<family>.swift`. |
| API transport and validation errors | `MereRunCLI` | `APIServerContract.swift`, `APIMultipartFormData.swift`, and `APIServerContract+Fields.swift`. |
| API discovery | `MereRunCLI` | `APIServerContract+Models.swift`, which projects Core profiles. |
| API request translation and responses | `MereRunCLI` | The matching `APIServerContract+<modality>.swift` file. InstantMesh uses `Commands/InstantMeshAPIContract.swift`. |

## Preserve discovery behavior

`ManagedModelCatalog.allSpecs` explicitly assembles family definitions in public
inventory order. Keep that order: repository aliases resolve to the first
matching spec. Canonical model IDs remain independently addressable when models
share an upstream repository.

Companion definitions participate in model lookup without appearing in the public
inventory. Keep source revisions, artifact patterns, mounted components, usage
terms, and runtime download policy with their definitions. Shared source constants
and usage-term construction belong to `ManagedModelCatalog+Sources.swift`.

To change installed-root behavior, edit the spec validation extension or its
family resource validator. Model metadata and API profiles do not inspect files
or load a runtime.

## Preserve shell contracts

`CommandCapabilityCatalog.swift` declares catalog order and schema version.
Family files declare capabilities; `CommandCapabilityCatalog+Options.swift`
owns the shared receipt and progress options. The schema applies presentation
defaults and preserves decoding compatibility for earlier documents.

Keep IDs, positional arguments, option order, choices, defaults, and output
metadata aligned with the CLI parser. Preserve serialized fields when moving
definitions. A file reorganization does not require a schema version change.

## Keep API policy with its modality

API modality files translate wire fields into shared operation settings and
construct response payloads. They retain model aliases, transport-specific
limits, validation diagnostics, and response schemas. Core and audio operations
own runtime preparation, execution, and cleanup.

`MultipartFormData.validateFields` checks allowed text and file fields, text
encoding, and duplicate text fields. Routes supply their field sets and error
messages. Image reconstruction routes require UTF-8 text; video depth retains
optional UTF-8 field decoding. The validator does not reorder uploads. Each
route checks required files, content types, view counts, and cameras afterward.

HTTP routing, authentication, admission, upload storage, and artifact retention
remain with the server adapters. A contract reorganization must preserve these
boundaries.

## Verify a change

Capture `mere.run catalog --json`, per-capability output, model inventory, and
API metadata or representative responses before changing their owners. Compare
the results afterward, preserving array order and normalizing only measured
durations or generated timestamps when needed.

Run the focused catalog, resolver, capability, Studio command-generation, and API
contract tests for the affected surface. Finish with `./scripts/check.sh` and the
documentation build. Runtime changes also need the behavioral evidence described
by their operation's documentation; schema and fixture checks do not qualify
native inference quality or performance.
