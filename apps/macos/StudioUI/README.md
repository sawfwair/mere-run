# Studio views

This directory is for contributors changing the macOS Studio interface.
Read the [macOS source map](../README.md#source-map) for the complete view and
model inventory.

`StudioRootView` composes the window's services and prompt task controller.
The workspace view owns navigation wiring, layout, focus, dialogs, file panels,
and clipboard actions. Bind prompt controls to `StudioPromptTaskController.draft`
and delegate task transitions and run preparation to that controller. Do not
mirror inactive drafts or reconstruct CLI execution in a view.

Keep selection and panel behavior in `NavigationModel`. Schemas and pure
presentation rules belong in StudioKit. Job progress comes from `JobStore`;
history and artifacts come from `StudioLibraryStore`.

When changing prompt workflows, run the Studio tests and render the affected
screens with `MERERUN_STUDIO_SNAPSHOT_DIR` and `StudioSnapshotTests`. Check
keyboard navigation, focus, error recovery, compact layouts, and accessibility
separately from rendering. See
[prompt workspace ownership](../../../docs/internals/studio-prompt-workspace.md).
