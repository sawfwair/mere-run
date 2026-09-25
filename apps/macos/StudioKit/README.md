# Studio model layer

This directory is for contributors changing macOS Studio state and CLI adapters.
Keep it independent of SwiftUI and model runtimes. Read the
[macOS source map](../README.md#source-map) before following an individual task.

`StudioPromptTaskController` owns the active prompt draft, conversation identity,
task transitions, and run preparation. Its conversation methods prepare sends
and retries before changing history. `StudioTaskSessions` owns inactive drafts,
selection memory, result focus, command overrides, and persistence. Keep session keys stable.

`StudioCommandAdapter` translates prompt controls into a CLI request.
`StudioTaskSessions.commandForm` and `resolving` share the effective Command
values. `CommandTemplate.validationMessage(for:execution:)` applies the same
validation during preparation and final job admission.

`StudioOptionScope` is the one answer to "which options does this run's model
take". Build it through `StudioScopeSource` from the argv a surface would
launch (`scope(mode:draft:)`, `scope(capability:form:)`), never from a model
name. It resolves the family with the contract, asks `StudioModelIdentifying`
only for a model the contract doesn't list, and hands every surface
`options(forFamily:)`. Drafts keep hidden values; `StudioDraft.scoped(to:mode:)`
and `StudioConsoleDraft.scoped(to:)` reset them in the copy that validates and
launches, and `StudioOptionScopes.filtered` drops what a builder emits anyway.
`StudioModelIdentityStore.shared` caches `catalog resolve` answers per folder;
`MereRunController` supplies its resolver. Tests pass a `StudioScopeSource` with
a routed capability and fixed identities.

`StudioTaskRunner` is the one submission path for every task run that is not a
conversation turn: it names the destination
(`StudioOutputLocation.destination(for:)`), applies Command edits, validates,
records the Library row under the template's own mode, and remembers
`"<task>.requestID"` for Stop. A task on the shared task workspace keeps its
draft as a `StudioTaskDraft` under `"<task>.taskDraft"` (the template plus its
`StudioConsoleDraft`, with the task's other variants parked beside it),
imported once from what a legacy page kept — every command's draft, or the
Vision and 3D pages' keys rebuilt into their commands
(`StudioTaskDraftMigration.swift`); `commandForm` returns that form directly, so the
Command view and the workspace never disagree. `StudioArchetypes` declares each
task's archetype and whether it uses a task draft; `StudioTaskSchema` reads the
well slots, chips, and inspector sections from the template's contract.

`MereRunController` owns CLI configuration and submission. `JobStore` owns jobs,
queues, cancellation, and completion. `StudioLibraryStore` records history and
artifacts independently of window lifetime. Historical replay uses recorded
arguments rather than the active task's command overrides.

The native process runner observes process exit and drains both output streams,
then reports completion after the final decoding callbacks.
Cancellation stops the owned process group and escalates if it ignores termination.
Job results retain outputs confirmed by a success receipt or a resident render
result. An unsuccessful exit clears unverified file probes from the job and
Library row without deleting files. Confirmed outputs survive later process
failure and console-buffer truncation.

`StudioServiceProcess` owns one long-lived server command (`api serve`,
`vision serve`, `music serve`) in the service lane: the job Studio started or
adopted from the Command Console, and start, stop, restart, and preflight.
`StudioLocalServer` owns the local API server for the life of the app: its serve
options, its `StudioServiceProcess`, and a phase derived from that process and
`StudioServingMonitor`'s last answer from the endpoint. The Server page and the
menu bar extra both drive it; neither keeps its own copy of the server's state.
The endpoint and key stay on `MereRunController`, so the key is read from the
Keychain at launch and never stored with the options.

`StudioModelStore` owns the inventory shared by Models and the composer. Refreshes
publish a complete typed snapshot and reject results from older requests or CLI
configurations. Downloads use the same typed inference jobs from either entry
point. Read progress and cancellation from `JobStore`; do not copy them into view
state. After a download, recheck the current readiness requests.

`StudioTaskSessions` reconciles result focus when Library selection changes and
clears references to deleted rows. `StudioFileExport` honors an explicit file
destination and preserves existing names during folder exports. It reports missing
sources and copy failures without modifying original artifacts.

Run `swift test --filter StudioPromptTaskControllerTests` for prompt journeys.
The full repository gate also checks command argument fixtures, persistence,
job behavior, and view contracts. See
[prompt workspace ownership](../../../docs/internals/studio-prompt-workspace.md)
for validation boundaries.
