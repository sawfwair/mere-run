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

`MereRunController` owns CLI configuration and submission. `JobStore` owns jobs,
queues, cancellation, and completion. `StudioLibraryStore` records history and
artifacts independently of window lifetime. Historical replay uses recorded
arguments rather than the active task's command overrides.

The native process runner drains both output streams before reporting completion.
Cancellation stops the owned process group and escalates if it ignores termination.
Job results retain outputs confirmed by a success receipt or a resident render
result. An unsuccessful exit clears unverified file probes from the job and
Library row without deleting files. Confirmed outputs survive later process
failure and console-buffer truncation.

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
