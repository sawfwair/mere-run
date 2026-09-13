# Studio prompt workspace ownership

This guide is for contributors changing macOS Studio prompt tasks. Use the
following owners to preserve drafts, command edits, and history across task
changes. Studio runs the public CLI; it does not implement model inference.

## State and transitions

| Owner | Responsibility |
| --- | --- |
| `StudioPromptTaskController` | Active prompt draft, conversation identity, task restoration, Analyze handoffs, and request preparation |
| `StudioTaskSessions` | Inactive drafts, per-thread Chat and Code drafts, selection memory, result focus, Command overrides, and versioned persistence |
| `NavigationModel` | Destination, selected Library item, and panel visibility |
| `StudioRootView` | Window composition, navigation wiring, layout, focus, dialogs, and platform interactions |
| `MereRunController` and `JobStore` | CLI launch configuration, admission, queues, cancellation, and completion |
| `StudioModelStore` | Shared model inventory, refresh identity, and download discovery through JobStore |
| `StudioLibraryStore` | Conversation transcripts, run history, receipts, and artifacts |

Bind the composer, inspector, and mapped Command fields to the active draft.
Draft edits update task sessions synchronously. Disk writes use the session
store's existing debounce and flush behavior. Do not add an inactive-draft cache
or depend on SwiftUI change callbacks to save edits.

Keep the persisted task and conversation keys stable. Import all recognized
legacy `studio.drafts` entries when the workspace appears. Full session drafts
take precedence, including intentionally empty prompts. Preserve the legacy
scene value and the session store's handling of unreadable files.

Chat and Code retain separate next-turn settings for the same conversation.
An explicit new thread has no Library row until its first send. A task detour
restores that choice and its unsent draft. Branching uses the preset, model,
and system instruction recorded at the selected turn.

## Prepare and submit requests

Use `StudioCommandAdapter` for prompt-to-command translation, then
`StudioTaskSessions.resolving` for Command overrides. The Command panel reads
the same effective form through `StudioTaskSessions.commandForm`.

Validate the resolved request with
`CommandTemplate.validationMessage(for:execution:)` before changing history or
preparing output directories. Final job admission repeats the same validation.
The command contract and template retain their respective validation policies.
Chat uses `TextChatTokenBudget` from `MereRunContract` for the same output/context
bounds as Core. Numeric display ranges remain hints; they do not reject valid
requests outside a preferred slider range.

Conversation send and retry share transcript rendering, context budgeting, and
command resolution. Retry builds a candidate transcript without changing the
stored thread. Only a validated replacement removes the preceding assistant
reply. A rejected request leaves the transcript and unsent composer draft intact.

Historical Library replay uses its recorded argument vector and allocates its
output identity through `StudioLibraryReplay`. Current Command overrides do not
change historical replay. Stop selects the current task's job or the open
conversation's turn, even when another task submitted a newer job.

## Model setup and result recovery

Both the composer and Models read `MereRunController.modelStore`. The store
publishes inventory and catalog metadata together, retains the last readable
snapshot after a failed refresh in the same configuration, and rejects responses
from superseded requests or another CLI/model location. A location change clears
the previous location's inventory before loading its replacement.

Submit downloads through `StudioModelStore.startPull`. It reuses an active pull
for the same model and launch configuration, and submits new pulls through the
existing inference queue. A pull for another model location remains owned by
JobStore and does not satisfy the new location's download request.
Queued and running downloads expose their actual job progress, logs, and cancel
action on the Models page. Model pulls stay in the job store for the current session and do not create
media Library entries. The Models page keeps the latest download log available
after failure or cancellation. Completion refreshes inventory and the currently requested model
readiness, without restoring the draft that initiated the download.

When you select another Library row, `StudioTaskSessions.rememberSelection`
clears a different focused result. Task detours retain focus for the same row.
Deletion clears saved focus and selection references. Closing the result returns
keyboard focus to the composer. Historical retry and continuation preserve the
source artifact and use their existing lineage and output-allocation rules.

Use `StudioFileExport.copy(_:to:)` after the user chooses a single destination.
It copies before replacing a confirmed destination. Use `copy(_:into:)` for
folder exports: it deduplicates shared sources, keeps existing destination files,
and returns every missing or failed source for the UI to report.

## Validate a change

Run the prompt controller journey tests:

```bash
swift test --filter 'StudioPromptTaskControllerTests|StudioModelStoreTests|StudioResultWorkflowTests'
```

Also run command argument/default fixtures, session and Library persistence
tests, navigation tests, and the full repository gate. Render affected prompt,
conversation, Analyze, and compact layouts with `StudioSnapshotTests`.

Model tests establish state, command, and fixture-backed job behavior. Rendered
screens establish layout. Neither establishes keyboard operation, VoiceOver
announcements, real-model output quality, or GPU cancellation recovery. Record
those checks separately when validating the corresponding workflow.
