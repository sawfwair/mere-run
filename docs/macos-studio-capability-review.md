# macOS Studio capability coverage

## The policy

Every capability the CLI declares is filed under exactly one **domain** and
reached at a **task** inside it. Coverage is that mapping, not a count of
bespoke views.

Three things follow from it.

1. **A capability's home is never in question.** `CommandTemplateID.studioDomain`
   (`apps/macos/StudioKit/StudioNavigation.swift`) is an exhaustive mapping from
   every command template to one of the fifteen
   domains, so a Library row files where the work was done and a command is
   always somewhere a person can name.
2. **Designed surfaces are a deliberate subset.** Some capabilities have a task
   built for what they actually produce — the Analyze canvas draws detection
   boxes over the picture, Video ▸ Subjects is a three-stage board, Models ▸
   Installed is a list and detail. Most do not, and that is a decision about
   where the effort goes, not a gap in reachability.
3. **The Command Console is the guaranteed home.** It renders any capability
   from `MereRunCapabilityCatalog`: the option's `kind` picks the control, its
   `choices` fill the picker, its `depends_on` gates the row, and the "Will run"
   block is built from the same values. Nothing has to be written for a new
   command. The day the contract declares one, the console can run it.

What that retires is the earlier reading of "first class", recorded in the audit
below: that every command must have a dedicated workspace, which in v1 was built
as a dedicated modal sheet per command. Twenty-two sheets and a second
navigation system in the sidebar footer were the cost.
[macOS Studio v2](./macos-studio-v2.md) keeps the coverage goal and drops that
reading.

## What is enforced today

| Guard | Where | What it proves |
|---|---|---|
| `everyPublicCLICommandIsCatalogedOrExplicitlyExempt` | `Tests/MereRunCLITests/CapabilityCatalogTests.swift` | Every public CLI leaf command is either in the contract or in `contractExemptCommandIDs` with the reason it stays CLI-only. It also rejects a stale exemption, so the list cannot rot. |
| `capabilityFlagsMatchArgumentParserHelp` | same file | Every flag the contract declares is one the command's ArgumentParser help accepts. |
| `testEverySharedCLICapabilityHasAnAppOwnedSurface` | `apps/macos/StudioUITests/StudioTypesTests.swift` | Set equality between the app's capability IDs and the contract's, in both directions. |
| `testEveryCommandTemplateMapsToADomain` | `apps/macos/StudioUITests/NavigationModelTests.swift` | Every command template files into exactly one domain, and every domain owns at least one command. |
| `CommandContractGuardTests` | `apps/macos/StudioKitTests` | Every flag the app can emit, from a sweep of maximal drafts, is one the contract declares, under the subcommand path the contract names. |

The chain is closed in both directions: a CLI command cannot ship without a
contract entry or a recorded exemption, a contract entry cannot ship without an
app surface, and an app surface cannot emit a flag the contract does not
declare.

One thing is deliberately *not* asserted: that each capability maps to a single
named task. The domain mapping is total and tested; the task mapping is a design
decision made surface by surface, and the console is what makes the absence of a
task assertion safe.

## Exemptions

Thirty-one leaf commands stay CLI-only, each named in `contractExemptCommandIDs`
with its reason: the twelve `executor` commands and `relay serve` (the Relay
console owns node identity, placement, scheduling, and fleet telemetry), the
fourteen `graph` commands (Graph Studio owns workflow authoring and execution,
and the worker verbs are a machine-to-machine protocol), the three
`vision image-to-3d` aliases (surfaced through the 3D domain as
`image reconstruct-3d`), and `catalog` (it emits the contract that the shells
compile against).

---

# Historical audit (August 2026)

Everything below is the original review, preserved as the record of how the
coverage gaps were found and closed. It describes v1 as it shipped: the counts
are accurate for that moment, and the "workspace" language is the superseded
reading. Do not read it as a description of the app today — see
[`apps/macos/README.md`](../apps/macos/README.md) for that.

This review compared the public CLI command tree against the surfaces macOS
Studio owned, and identified every capability that had not been brought to the
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

| Measure | Before | After |
|---|---|---|
| CLI leaf commands | 158 | 158 |
| Commands in the shared capability contract | 101 | 127 |
| Commands with no contract entry | 57 | 31 |
| Explained by external boundaries or aliases | 31 | 31 |
| Genuine first-class gaps | 26 | 0 |

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

The unguarded link was **CLI to contract**. Adding a command to the CLI did not
require adding it to the contract, and nothing failed when you skipped that step.
Because the Studio gate is keyed to the contract rather than to the CLI, an
omission at that first link propagates silently: the command ships, Studio
never gets a surface, and every test still passes.

`apps/macos/README.md` stated the intent precisely: the inverse coverage test
requires every command in the shared contract to have an app-owned template or
utility surface, so a newly cataloged CLI command cannot silently ship without a
macOS path.

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
aliases of the `image reconstruct-3d` family, which Studio surfaces once rather
than twice.

**Contract introspection — 1 command.** `mere.run catalog` emits the contract
that Studio compiles against, so the app has no need to shell out for it.

## What changed

1. **The unguarded link is closed.**
   `everyPublicCLICommandIsCatalogedOrExplicitlyExempt` walks
   `MereRunCLI.configuration.subcommands` and requires every public leaf command
   to have a contract entry or an entry in `contractExemptCommandIDs` carrying
   the reason it stays CLI-only. The test also rejects a stale exemption — one
   that no longer matches a CLI command, or one that has since been cataloged —
   so the list cannot rot. `graph`, `executor`, `relay serve`, the
   `image-to-3d` aliases, and `catalog` are recorded there as deliberate
   decisions rather than undocumented absences.

2. **All 26 capabilities are cataloged and surfaced.** The contract grew from
   101 to 127 commands, each with the flags its ArgumentParser command actually
   declares, and `CapabilityCatalogTests` verifies every one against CLI help.
   Studio gained 24 templates and 2 app-owned utilities, so the inverse coverage
   test still holds at set equality.

3. **Geospatial is a first-class category.** `CommandCategory.geospatial` carries
   flood, fire, TESSERA, and OlmoEarth. Benchmarks, model store locations, and
   plugin details/run/rollback extend the Models and Plugins workspaces.
   `config list` and `config path` join the existing configuration utilities as
   `loadConfigurationSummary()` and `loadConfigurationPath()`. `speech listen`
   and `vision serve` are typed resident-service surfaces.

4. **Stale counts corrected.** `docs/macos-studio-roadmap.md` described the app
   as consuming an "89-command capability contract"; it now names the current
   127 and records the external boundaries as named exemptions.

## Where the closed gaps live now

The v1 destinations named in this audit were renamed and re-hosted by v2. Their
current addresses:

| Capability | v1 destination | Today |
|---|---|---|
| `geo` (4) | Geospatial Lab, a sheet | **Earth** ▸ Flood, Fire, TESSERA, OlmoEarth |
| `model benchmark` (12) | Health & Repair, a sheet | **Models ▸ Benchmarks**, beside **Models ▸ Health** |
| `model location` (5) | Models → Locations, a nested sheet | **Models ▸ Locations** |
| `speech listen` (1) | Voice Studio → Listen Live | **Audio ▸ Live** |
| `vision serve` (1) | Serving → Vision Grounding | **Server ▸ Serving**, its Vision Grounding section |
| `plugin` info/run/rollback (3) | Plugins, a sheet | **Plugins ▸ Catalog** |
| `config list` / `path` (2) | Settings | **Settings ▸ Advanced** |

Each was built against a typed capability already proven to match CLI help,
rather than a hand-copied argument list — which is what made the move cheap.
