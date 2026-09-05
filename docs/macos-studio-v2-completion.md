# Studio v2 completion

This guide describes the Studio workflows and the checks that protect them.
The app continues to run the public CLI; it does not load model runtimes itself.

## Inspect and compare results

1. In an image result card, select **Focus** or click the image.
2. Use the zoom controls, pinch, or drag to inspect it. **Fit** restores the viewport.
3. Select **Compare** and choose another image. Both views share zoom and pan.
4. Review **Changed settings**. Output paths and credentials do not appear as
   generation differences. The comparison includes positional prompts, different
   commands, and additional options entered in Command. Older results without
   settings are identified as such.
5. Select **Results** or press Escape to return to the feed and its existing draft.

The feed keeps earlier work visible while new runs finish. Image focus stays on
the selected result, and another finished image becomes a comparison candidate.
Panning stays within the image surface when you zoom out or resize the window.

## Continue from a result

**Continue with…** creates a draft for editing, reference guidance, video,
image understanding, or segmentation. It carries the source file and relevant
prompt into the destination task. Review the destination's model and settings,
then run it. The resulting Library item records its source run.

**Save copy…** preserves the original. Rerun and Vary preserve recorded command
options and allocate distinct output destinations before execution, including
known sidecar destinations. Subjects preview and tracking operations also use
distinct directories.

## Edit the command for a task

Use **Command** or Option-Command-C from any task. The panel edits the current
task's contract form and shows the command that will run. Mapped prompt controls
synchronize with the form. Specialist forms retain Command edits and merge later
changes from their simple controls into the edited command. A notice identifies
active Command overrides; **Use form settings** clears them.

The standalone **Command Console** has its own run identity, output, and Stop
control. Command-Return follows the same Run or Queue behavior as its button.
Recorded arguments preserve repeatable options and quoted values. API keys and
passwords supported by the launch environment stay out of process arguments;
credentials are removed or masked in saved task and history state.

## Resume work

Task sessions retain full prompt drafts, specialist input and settings values,
and selected runs. Each chat thread retains its own unsent text, attachment,
model, and settings, including separate Chat and Code drafts. **New thread**
returns to an unsent new thread when one exists. Switching threads or tasks
does not discard unsent text. Relaunch restores
saved state; an earlier queued or running row becomes **Interrupted** when no
process owns it. Malformed session files are preserved instead of overwritten.

History recording and server monitoring belong to the app session, so closing
or replacing a view does not stop them. Stop in a chat targets its active turn.
Branch uses the model, system instruction, and preset recorded at that turn,
including an intentionally empty system instruction.

Analyze checks the selected input against the recorded source. For newly
recorded files, it also checks size and modification time. Replacing an input
removes stale overlays, result text, and processed-video playback. When a run
provides boxes without mask pixels, the overlay labels that limitation.

## Window layout and reading

The sidebar toggle, task switcher, and panel controls share one 52-point header.
There is no separate toolbar row. Drag the header background to move the window;
Control-Command-S toggles the sidebar. Native window controls remain available.

The default window is 1440 by 820 points. At smaller widths, task navigation
compacts and auxiliary panels overlay the workspace. Analysis controls and
results stack vertically when they cannot fit beside one another. Each pane
scrolls independently. Selecting a result or thread dismisses the Library
overlay so you can continue working. The result feed loads cards lazily.

Primary reading surfaces use semantic text styles. Secondary text and selected
controls use stronger contrast in light and dark appearances. Icons retain
accessible action labels; image focus supports Escape and Command-plus,
Command-minus, and Command-zero.

## Model inventory

`mere.run model list --json` returns an `inventory` object with `rows`, inventory
mode, completeness, and elapsed time, plus usage-term descriptions. Each row may
include `contextWindow` from an installed model's `config.json`. Studio uses it
when estimating conversation history room; an explicit context size takes
precedence. Missing or unfamiliar configurations report no context capacity.
This value is declared capacity, not a tokenizer measurement or a runtime
qualification result.

## Validation contract

Run the repository gate before opening a PR:

```sh
./scripts/check.sh
```

`StudioContinuityTests` and `StudioResultWorkflowTests` cover execution metadata,
exact replay, cancellation isolation, app-owned history, interruption recovery,
branch settings, secret exclusion, source preservation, task ownership, and
responsive layout policy. CLI tests cover typed context configuration and the
additive inventory option. Existing contract guards check declared options and
generated flag constants.

Render the Studio surfaces without launching model processes:

```sh
MERERUN_STUDIO_SNAPSHOT_DIR=/tmp/studio-v2-snapshots \
  swift test --filter StudioSnapshotTests
```

The renderer supplies a reference date and isolates saved voice profiles.
Comparison, focus, compact specialists, and the Command overlay supplement the
existing board renders. Inspect the images as well as their nonblank assertions.
Offscreen captures cannot prove native glass compositing, keyboard focus in a
live window, VoiceOver navigation, or installed-model inference. Those remain
separate interactive and hardware acceptance checks.
