---
name: mere-run
description: Develop and validate the public mere.run CLI, runtime libraries, and Studio clients in the OSS source repository. Use for command parsing, model integration, workflow contracts, app integration, tests, or CLI documentation. For operating an installed CLI without source changes, use use-mere-run.
---

# Develop mere.run

Work from the public `mere.run` source checkout. The separate root `mere`
portfolio CLI is outside this skill's scope. For operator help without code
changes, use the `use-mere-run` skill when available.

## Establish the source and runtime

Read `AGENTS.md`, `Package.swift`, `CODEBASE.md`, and the closest module README.
Use `docs/repository-tour.md` and `docs/architecture.md` to find runtime owners.
Inspect Git status before editing and preserve unrelated work.

Use the source CLI when testing changes:

```bash
swift run mere.run --version
swift run mere.run --help
swift run mere.run catalog --json
```

An installed `mere.run` can differ from the checkout. Record which executable
and version produced each result; installed-binary success does not validate
source changes. Use command help for parsing and the catalog for structured
capabilities, options, and output contracts. The catalog is not a replacement
for the full command tree.

## Find the contract that owns the change

| Change | Start here |
| --- | --- |
| Commands, flags, stdout, and diagnostics | `Sources/MereRunCLI/MereRunCLI.swift`, `Sources/MereRunCLI/Commands/`, `Sources/MereRunCLI/Support/` |
| Capability metadata shared with Studio | `Sources/MereRunContract/`, `Tests/MereRunCLITests/CapabilityCatalogTests.swift` |
| Model IDs, availability, and storage | `Sources/MereRunCore/ManagedModelCatalog.swift`, `ManagedModelSupport.swift`, `ModelResolver.swift` |
| Prompting and model-specific workflows | `Sources/MereRunCLI/Guides/`, `GuideCommand.swift` and `ModelGuideRegistry.swift` under `Sources/MereRunCLI/Commands/` |
| Graphs, executors, relay, and durable runs | Relevant CLI command, `Sources/MereRunRelayKit/`, `docs/workflows.md` |
| Evaluation packs and results | `Sources/MereRunEvaluation/`, `docs/evaluation-packs.md` |
| macOS Studio | `apps/macos/StudioKit/`, `apps/macos/StudioUI/`, `apps/macos/StudioKitTests/` |
| iOS Studio | `apps/ios/`, `Sources/MereRunRelayKit/`, `docs/ios-studio.md` |

Discover command additions from `--help` instead of maintaining a second
command inventory in this skill. This also covers plugins, adapters, world
sessions, geospatial inference, audio, and model optimization.

For a model or command change, inspect its cookbook and handbook:

```bash
swift run mere.run guide --list
swift run mere.run guide --list-models --json
swift run mere.run catalog image.generate --json
swift run mere.run guide image generate
swift run mere.run guide --model image-zimage-nano
```

Handbooks explain provider guidance and validation limits. They do not prove
that a checkpoint has passed local inference. Resolve canonical model IDs from
the managed catalog; do not infer support from a runtime type name or alias.

## Implement at the owning boundary

- Keep runtime behavior in the owning library. Studio consumes CLI and shared
  contracts; it must not become a second runtime implementation.
- Use typed decoding at configuration and tokenizer boundaries. Keep stdout
  consistent with the command's declared output; send diagnostics to stderr.
- Update the closest CLI or core tests when changing parsing, model resolution,
  compatibility, API behavior, or output shape.
- When command paths or abstracts change, run
  `./scripts/update-docs-command-reference.sh`. When capabilities change,
  update the shared catalog and its parity tests.
- Keep managed model entries, support metadata, and model handbooks aligned.
  Use `ModelGuideTests` to check handbook coverage.
- Preserve loopback API defaults and required authentication for non-loopback
  binds. Preserve the supported tool-execution approval behavior.
- Follow the repository's OSS boundary and vendor rules. Public macOS and iOS
  clients are in scope; private deployment and store distribution machinery
  belong elsewhere.

## Validate the change

Use focused tests while iterating. Before opening a PR, run the repository gate:

```bash
./scripts/check.sh
```

The gate owns lint, build, tests, CLI help checks, and hygiene checks. Never
remove a rejected pattern from a hygiene check to make the change pass.
Consult the managed catalog, docs, and tests for the supported replacement.

Use runtime smoke when relevant and the machine has the required assets:

```bash
MERERUN_RUN_E2E=core ./scripts/check.sh
MERERUN_RUN_E2E=installed ./scripts/check.sh
```

Follow `AGENTS.md` for asset, GPU, and network validation limits. Report parsing,
local tests, real inference, hosted checks, and deployment as separate evidence.

## Maintain the distributed skills

`skills/mere-run/` owns this developer skill. `skills/use-mere-run/` owns the
operator skill copied into the macOS app by `scripts/build_mere_run_app.sh`.
Installed skills are copies and can drift from the repository.

When updating skills, keep examples explicit about model selection. Run:

```bash
swift test --filter SkillContractTests
```

This test reads the skill examples, parses their commands, resolves catalog and
handbook references, and checks that pull/run recipes select the same model.
It does not download models or run inference. Update installed copies at their
actual discovered locations when requested; preserve unrelated local files.
