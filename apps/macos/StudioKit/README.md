# Studio model layer

This directory is for contributors changing macOS Studio state and CLI adapters.
Keep it independent of SwiftUI and model runtimes. Read the
[macOS source map](../README.md#source-map) before following an individual task.

`StudioPromptTaskController` owns the active prompt draft, conversation identity,
task transitions, and run preparation. Its conversation methods prepare sends
and retries before changing history. `StudioTaskSessions` owns inactive drafts,
selection memory, command overrides, and persistence. Keep session keys stable.

`StudioCommandAdapter` translates prompt controls into a CLI request.
`StudioTaskSessions.commandForm` and `resolving` share the effective Command
values. `CommandTemplate.validationMessage(for:execution:)` applies the same
validation during preparation and final job admission.

`MereRunController` owns CLI configuration and submission. `JobStore` owns jobs,
queues, cancellation, and completion. `StudioLibraryStore` records history and
artifacts independently of window lifetime. Historical replay uses recorded
arguments rather than the active task's command overrides.

Run `swift test --filter StudioPromptTaskControllerTests` for prompt journeys.
The full repository gate also checks command argument fixtures, persistence,
job behavior, and view contracts. See
[prompt workspace ownership](../../../docs/internals/studio-prompt-workspace.md)
for validation boundaries.
