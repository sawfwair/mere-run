# Studio prompt workspace ownership

Use this map when you change macOS Studio prompt tasks, so drafts, Command
edits, and history survive task changes. Studio runs the public CLI; it
doesn't implement model inference. The types live in `apps/macos/StudioKit`
and `apps/macos/StudioUI`.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `StudioPromptTaskController` | Active prompt draft, conversation identity, task restoration, Analyze handoffs, and request preparation |
| `StudioTaskRunner` | The single submission path for every task run that isn't a conversation turn (servers, model pulls, and Realtime keep their own): destination naming, Command edits, validation, Library attribution, and the job that Stop acts on |
| `StudioTaskSessions` | Inactive drafts, the shared task workspace's `StudioTaskDraft` per task, per-thread Chat and Code drafts, selection memory, result focus, Command overrides, and versioned persistence |
| `NavigationModel` | Destination, selected Library item, and panel visibility |
| `StudioRootView` | Window composition, navigation wiring, layout, focus, dialogs, and platform interactions |
| `MereRunController` and `JobStore` | CLI launch configuration, admission, queues, cancellation, and completion |
| `StudioModelStore` | Shared model inventory, refresh identity, and download discovery through `JobStore` |
| `StudioLibraryStore` | Conversation transcripts, run history, receipts, and artifacts |

## Drafts and task state

Bind the composer, inspector, and mapped Command fields to the active draft.
Draft edits update task sessions synchronously, and disk writes use the session
store's debounce and flush behavior. Don't add an inactive-draft cache or rely
on SwiftUI change callbacks to save edits.

Keep the persisted task and conversation keys stable. Import every recognized
legacy `studio.drafts` entry when the workspace appears. Full session drafts
take precedence, including intentionally empty prompts. Preserve the legacy
scene value and the session store's handling of unreadable files.

Chat and Code keep separate next-turn settings for the same conversation. An
explicit new thread has no Library row until its first send, and a task detour
restores that choice and its unsent draft. Branching uses the preset, model,
and system instruction recorded at the selected turn.

## Prepare and submit requests

Translate a prompt into a command with `StudioCommandAdapter`, then apply
Command overrides with `StudioTaskSessions.resolving`. The Command panel reads
the same effective form through `StudioTaskSessions.commandForm`.

Validate the resolved request with
`CommandTemplate.validationMessage(for:execution:source:)` before you change
history or prepare output directories; final job admission repeats the same
validation. The command contract and the template keep their own validation
policies. Every command whose contract declares `--max-tokens` (Chat, Code, and
the vision prompts) uses `TextChatTokenBudget` from `MereRunContract` for the
same output and context bounds as Core. Numeric display ranges are hints; they
don't reject valid requests outside a preferred slider range.

Conversation send and retry share transcript rendering, context budgeting, and
command resolution. Retry clears the same model and readiness gates as send,
then builds a candidate transcript without changing the stored thread. Only a
validated replacement removes the preceding assistant reply. A rejected request
leaves the transcript and the unsent composer draft intact.

Historical Library replay uses its recorded argument vector and allocates its
output identity through `StudioLibraryReplay`; current Command overrides don't
change it. Stop selects the current task's job or the open conversation's turn,
even when another task submitted a newer job.

## Model setup and result recovery

The composer and the Models page both read `MereRunController.modelStore`. The
store publishes inventory and catalog metadata together. After a failed
refresh in the same configuration, it keeps the last readable snapshot. It
rejects responses from superseded requests or from another CLI or model
location, and a location change clears the previous location's inventory
before it loads the replacement.

Submit downloads through `StudioModelStore.startPull`. It reuses an active pull
for the same model and launch configuration, and submits new pulls through the
inference queue. A pull for another model location stays with `JobStore` and
doesn't satisfy the new location's download request.

Queued and running downloads show their job progress, logs, and cancel action
on the Models page. Model pulls stay in the job store for the current session
and don't create media Library entries. The Models page keeps the latest
download log after failure or cancellation. Completion refreshes inventory and
the readiness of the currently requested model, without restoring the draft
that started the download.

When you select another Library row, `StudioTaskSessions.rememberSelection`
clears a different focused result; task detours keep focus for the same row.
Deletion clears saved focus and selection references. Closing the result
returns keyboard focus to the composer. Historical retry and continuation keep
the source artifact and use their lineage and output-allocation rules.

Call `StudioFileExport.copy(_:to:)` after the user chooses a single
destination; it copies before it replaces a confirmed destination. Use
`copy(_:into:)` for folder exports: it deduplicates shared sources, keeps
existing destination files, and returns every missing or failed source for the
UI to report.

## VoiceOver announcements

The active Studio window announces prompt validation errors, including repeated
submissions with the same error. Chat and Code announce when a reply starts,
finishes, fails, or stops. These announcements leave keyboard focus in place;
you can move to the transcript to read the reply. Streaming tokens don't
trigger individual announcements.

Completion announcements use the job's terminal state from the completion
stream, so a cancelled job announces that the reply stopped even if its process
exits with code 0. Background threads don't interrupt the active conversation.

## Validate a change

Run the prompt controller journey tests:

```bash
swift test --filter 'StudioPromptTaskControllerTests|StudioModelStoreTests|StudioResultWorkflowTests'
```

Also run the command argument and default fixtures, session and Library
persistence tests, navigation tests, and the full repository gate. Render
affected prompt, conversation, Analyze, and compact layouts with
`StudioSnapshotTests`.

Model tests establish state, command, and fixture-backed job behavior; rendered
screens establish layout. Neither establishes keyboard operation, VoiceOver
announcements, real-model output quality, or GPU cancellation recovery. Record
those checks separately when you validate the workflow.
