# macOS Studio capability review

This review compares the public CLI command tree against the surfaces macOS
Studio owns, and identifies every capability that has not been brought to the
app as a first-class path.

Reviewed at commit `5525608` against `Sources/MereRunCLI`,
`Sources/MereRunContract/CommandCapabilityContract.swift`, and
`apps/macos/MereRunStudio`.

## Summary

Studio is not drifting from the contract it tracks. It matches that contract
exactly, and a test enforces the match in both directions. The gap is one level
up: the contract itself covers 101 of the CLI's 158 leaf commands, and nothing
tests that the contract covers the CLI.

Of the 57 leaf commands outside the contract, 31 are accounted for by
documented product boundaries or by aliases Studio already covers elsewhere.
The remaining **26 commands across seven families are genuine gaps**: they ship
in the CLI, they have no macOS Studio path, and no test reports them.

| Measure | Count |
|---|---|
| CLI leaf commands | 158 |
| Commands in the shared capability contract | 101 |
| Commands with no contract entry | 57 |
| Explained by external boundaries or aliases | 31 |
| Genuine first-class gaps | 26 |

## Why the gaps are invisible

Three artifacts describe the product surface, and each pair is guarded
differently.

1. **CLI to documentation.** `DocumentationContractTests` walks
   `MereRunCLI.configuration.subcommands`, regenerates the tree in
   `docs/cli.md`, and asserts it matches. It also validates every documented
   command path against the live tree. This link is enforced, so `docs/cli.md`
   is an accurate inventory of all 158 leaf commands.

2. **Contract to CLI.** `CapabilityCatalogTests` iterates
   `MereRunCapabilityCatalog.document.commands` and asserts each entry's flags
   appear in that command's ArgumentParser help. This runs contract-first: it
   proves cataloged commands are accurate, not that every CLI command is
   cataloged. Its `helpByID` fixture is a hand-maintained dictionary that
   contains exactly the commands already in the contract.

3. **Contract to app.** `StudioTypesTests.testEverySharedCLICapabilityHasAnAppOwnedSurface`
   asserts set equality between the app's capability IDs and the contract's.
   This is the strongest guard in the chain, and it works.

The unguarded link is **CLI to contract**. Adding a command to the CLI does not
require adding it to the contract, and nothing fails when you skip that step.
Because the Studio gate is keyed to the contract rather than to the CLI, an
omission at that first link propagates silently: the command ships, Studio
never gets a surface, and every test still passes.

`apps/macos/README.md` states the intent precisely:

> The inverse coverage test also requires every command in the shared contract
> to have an App-owned template or utility surface, so a newly cataloged CLI
> command cannot silently ship without a macOS path.

The guarantee holds only for a *cataloged* command. Cataloging is the manual
step, and it is where these 26 commands were lost.

## Genuine gaps

Listed by family, ordered by product impact.

### Geospatial — 4 commands, no surface at all

- `mere.run geo flood`
- `mere.run geo fire`
- `mere.run geo tessera`
- `mere.run geo olmoearth`

This is the largest gap. The repository README lists geospatial analysis as a
headline capability alongside media generation and language models, and the
docs give `geo` its own runtime page. In Studio the family does not exist:
searching `apps/macos` for `geo`, `tessera`, `olmoearth`, `TerraMind`, or
`Sentinel` returns no match outside unrelated `geometry` identifiers. There is
no template, no category, and no contract entry.

`CommandCategory` has no geospatial case, so this family needs a new category
rather than a template added to an existing workspace.

### Model benchmarks — 10 of 12 commands

Studio exposes `model.benchmark.q36-mtp` and `model.benchmark.laguna-dflash`.
Both templates hardcode a single suite, so neither reaches the others:

- `mere.run model benchmark fused`
- `mere.run model benchmark fused-fixture`
- `mere.run model benchmark chat`
- `mere.run model benchmark code`
- `mere.run model benchmark vlm`
- `mere.run model benchmark tool-calls`
- `mere.run model benchmark tool-continuations`
- `mere.run model benchmark gemma4-kv`
- `mere.run model benchmark gemma4-mtp`
- `mere.run model benchmark api-workload`

The two shipped benchmarks are narrow decode micro-benchmarks. The missing set
includes the broad quality suites a user is most likely to want from a GUI —
`fused` is the versioned Mere Lite and Mere Comprehensive suite with its own
documentation page at `docs/benchmark-fused.md`, and `chat`, `code`, and `vlm`
are the per-capability evaluation slices.

### Model store locations — 5 commands

- `mere.run model location list`
- `mere.run model location add`
- `mere.run model location remove`
- `mere.run model location bind`
- `mere.run model location unbind`

Studio owns downloads, storage cleanup, garbage collection, manifest repair,
and runtime policy, but not the location of the store itself. Read-only search
roots and explicit model bindings have no GUI path.

This is a desktop-shaped need. Checkpoints are large, external and secondary
volumes are common on a Mac, and a user who keeps models on another disk
currently has to leave the app to register it.

### Plugin lifecycle — 3 commands

- `mere.run plugin info`
- `mere.run plugin run`
- `mere.run plugin rollback`

Studio covers list, install, and doctor. `rollback` is the notable omission:
it restores a retained signed plugin bundle, so it is the recovery path for a
bad plugin update, and it arrived with the signed-bundle install work in
`6f3f597`. A GUI that can install and update but cannot roll back leaves the
user without the recovery half of that feature.

### Configuration — 2 commands

- `mere.run config list`
- `mere.run config path`

Studio registers `config.set`, `config.get`, and `config.unset` as app-owned
utilities. `list` and `path` are missing from the same family, so the app can
write and read individual keys but cannot show the full masked configuration or
tell the user where it lives.

### Live microphone transcription — 1 command

- `mere.run speech listen`

The abstract is "Transcribe a macOS microphone with live Qwen ASR." A
macOS-specific capability has no macOS GUI path.

Voice Studio looks like it covers this but does not. `StudioVoiceView` records
through `AVAudioRecorder` to a `.wav` file, then submits `.speechTranscribe`
against that path behind a `fileExists` guard. That is record-then-transcribe.
It is not the streaming path, and the streaming path is the one the CLI
command provides.

### Resident vision grounding — 1 command

- `mere.run vision serve`

The Serving destination references only `.apiServe`. `music serve` and
`world serve` have their own templates elsewhere, so `vision serve` is the one
server in the CLI with no Studio surface. It is the resident, binary-frame
grounding endpoint.

## Not gaps

These account for the other 31 uncovered commands. No action is needed.

**External product boundaries — 27 commands.** `graph` (14 commands),
`executor` (12), and `relay serve` are deliberately delegated. Studio ships
`graphStudio` and `nodeConsole` as `externalURL` templates pointing at
`studio.mere.run/app` and `relay.mere.run`, both declared through
`StudioProductBoundary`. `docs/macos-studio-roadmap.md` records the decision
("Graph Studio and Node remain explicit external boundaries"), and
`apps/macos/README.md` restates it: Relay owns node identity, placement,
scheduling policy, and fleet telemetry, and the app links to the Relay console
instead of copying those controls.

**Aliases already covered — 3 commands.** `vision image-to-3d`,
`vision image-to-3d-trellis2`, and `vision image-to-3d-multiview` are VFX
aliases of the `image reconstruct-3d` family, which Studio surfaces through the
3D Creation workspace. `apps/macos/README.md` states the intent: "Image-to-3D
workflows share the Image workspace instead of being duplicated."

**Contract introspection — 1 command.** `mere.run catalog` emits the contract
that Studio compiles against, so the app has no need to shell out for it.

## Recommendations

1. **Close the unguarded link first.** Add a test that walks
   `MereRunCLI.configuration.subcommands` and asserts every leaf command either
   has a contract entry or appears in an explicit, named exemption list. The
   exemption list is where `graph`, `executor`, `relay serve`, the
   `image-to-3d` aliases, and `catalog` are recorded as deliberate, which turns
   today's undocumented absence into a reviewed decision. Without this, any
   surface work done now will drift again.

2. **Then close the gaps by impact.** Geospatial is the only missing product
   pillar and needs a new `CommandCategory`. The benchmark suites, model store
   locations, and plugin rollback are additions to workspaces that already
   exist. Configuration list and path extend an existing utility set.
   `speech listen` and `vision serve` each need a streaming surface rather than
   a form.

3. **Correct the stale count.** `docs/macos-studio-roadmap.md` describes the
   app as consuming an "89-command capability contract." The contract now holds
   101 commands.
