# Catalog and contract ownership

Use this map when you change model discovery, command capability metadata, or
API request and response contracts. Start with the owner of the behavior you
need to change.

## Ownership

| Surface | Owner | Reading order |
| --- | --- | --- |
| Managed model discovery | `MereRunCore` | `ManagedModelCatalog.swift`, the matching `ManagedModelCatalog+*.swift` family file, then `ManagedModelSpec.swift` |
| Model API capabilities | `MereRunCore` | `ManagedModelAPIProfile.swift`, then the profile that the model's catalog definition assigns |
| Installed model validation | `MereRunCore` | `ManagedModelSpec+Validation.swift`, then the referenced family resource validator |
| CLI capability schema | `MereRunContract` | `CommandCapabilityContract.swift` and `CommandCapabilityChoices.swift` |
| Command capability definitions | `MereRunContract` | `CommandCapabilityCatalog.swift`, then `CommandCapabilityCatalog+<family>.swift` |
| Runtime families and option scope | `MereRunContract` | `CommandCapabilityRouting.swift`, `CommandCapabilityScope.swift`, then `CommandCapabilityCatalog+<family>Routing.swift` |
| Model scope check and `catalog resolve` | `MereRunCLI` | `Support/CLICapabilityGate.swift`; `Support/CLIFamilyRouters.swift` for commands whose own router decides; Core's `ModelFamilyIdentification.swift`, with its per-domain probes and default choosers, for aliases, local and install-dependent models, and machine-chosen defaults |
| API transport and validation errors | `MereRunCLI` | `APIServerContract.swift`, `APIMultipartFormData.swift`, and `APIServerContract+Fields.swift` |
| API discovery | `MereRunCLI` | `APIServerContract+Models.swift`, which projects Core profiles |
| API request translation and responses | `MereRunCLI` | The matching `APIServerContract+<modality>.swift` file; InstantMesh uses `Commands/InstantMeshAPIContract.swift` |

## Contracts to preserve

### Model discovery

`ManagedModelCatalog.allSpecs` assembles family definitions explicitly, in
public inventory order. Keep that order: a repository alias resolves to the
first matching spec. Canonical model IDs stay independently addressable when
models share an upstream repository.

Companion definitions take part in model lookup without appearing in the public
inventory. Keep source revisions, artifact patterns, mounted components, usage
terms, and runtime download policy with their definitions. Shared source
constants and usage-term construction belong in
`ManagedModelCatalog+Sources.swift`.

To change installed-root behavior, edit the spec validation extension or its
family resource validator. Model metadata and API profiles never inspect files
or load a runtime.

### Command capabilities

`CommandCapabilityCatalog.swift` declares catalog order and the schema version.
Family files declare capabilities, and `CommandCapabilityCatalog+Options.swift`
owns the shared receipt and progress options. The schema applies presentation
defaults and keeps decoding compatibility with earlier documents.

Keep IDs, positional arguments, option order, choices, defaults, and output
metadata aligned with the CLI parser. Preserve serialized fields when you move
definitions. Reorganizing files doesn't require a schema version change.

### API modality policy

API modality files translate wire fields into shared operation settings and
build response payloads. They keep model aliases, transport-specific limits,
validation diagnostics, and response schemas. Core and audio operations own
runtime preparation, execution, and cleanup.

`MultipartFormData.validateFields` checks allowed text and file fields, text
encoding, and duplicate text fields; routes supply their field sets and error
messages. Image reconstruction routes require UTF-8 text, and video depth keeps
optional UTF-8 field decoding. The validator doesn't reorder uploads. Each
route checks required files, content types, view counts, and cameras afterward.

HTTP routing, authentication, admission, upload storage, and artifact retention
stay with the server adapters.

## Validate a change

Before you change one of these owners, capture `mere.run catalog --json`,
per-capability output, the model inventory, and API metadata or representative
responses. Compare the results afterward. Preserve array order, and normalize
only measured durations or generated timestamps.

Run the focused catalog, resolver, capability, Studio command-generation, and
API contract tests for the affected surface. Finish with `./scripts/check.sh`
and the documentation build. Runtime changes also need the behavioral evidence
that their operation page describes; schema and fixture checks don't qualify
native inference quality or performance.
