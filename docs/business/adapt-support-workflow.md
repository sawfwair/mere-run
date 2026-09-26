# Adapt the support workflow to your ticketing app

This guide is for a developer or support operator with a ticketing website
open on an Apple Silicon Mac. You can run GLiNER2.5 Decide, Ornith, and
Computer Use without changing the website's code. You still need a small
controller that understands how your app presents tickets and how to check the
result of each action.

Follow [Triage tickets in your support app](./customer-support.md) to set
up the tools yourself. The coding-agent prompt in this guide can also do that
setup. Keep the website open in one macOS window. Select a ticket that your
team authorizes for a draft-only run. Confirm that you can inspect its draft,
tags, team, and priority afterward.

## Map one ticket

1. Run `mere-computer-use windows` and record the process ID and window ID for
   the ticketing website. Keep the controller limited to that window.
2. Select the approved ticket. Record where the window shows its stable ID,
   status, customer message, and context needed for classification.
3. Record the exact team, priority, and tag options in the app. Locate the
   reply field, **Save draft**, and any **Send** control. Use the app's actual
   names in prompts and code.
4. Record how the app lists open tickets and how you return to the list. Check
   whether a fresh observation identifies the selected ticket after each
   navigation step.
5. Choose a read-only way to verify the saved result. Prefer a ticket
   API or audit log when you have access. Otherwise, read the ticket again
   through a fresh window observation and record that this is a weaker check.

The selected window can expose customer information in observations and run
records. Keep those files in a private local directory. Do not commit them to
the controller's source repository.

## Build the controller around the app

Keep app-specific reading and navigation in an adapter. The rest of the
controller can follow the same sequence for each ticket:

1. **Discover tickets.** List open tickets and identify each one by a stable
   ID. Stop if the selected window or list changes unexpectedly.
2. **Observe a ticket.** Read its message, context, status, and current values.
   Save the observation before inference.
3. **Classify the text.** Pass the observed text and your app's team and
   priority labels to `mere.run text classify`. Keep the selected labels and
   scores.
4. **Decide whether to act.** Skip resolved tickets, `No action` results, and
   requests that need review. GLiNER scores are not calibrated probabilities.
5. **Act in the app.** Recheck the ticket ID and content. Give accepted labels
   and draft-only limits to `mere-computer-use plan`, then run the plan with
   Ornith.
6. **Verify the result.** Read the ticket again. Check its team, priority,
   tags, saved draft, unchanged status, and sent-reply count. Stop at the first
   mismatch.
7. **Resume carefully.** Compare the current ticket with the saved receipt
   before retrying. Skip a completed ticket only after verifying its final
   state.

Set an action limit and timeout for each ticket. Save the initial observation,
classification, plan, action count, final observation, verification result,
and wall time. Test navigation, resolved tickets, uncertain classifications,
interrupted runs, and a changed ticket before processing an inbox.

The controller uses the website's visible controls for changes. A read-only
API can strengthen verification, but the workflow does not require adding an
API or model integration to the ticketing app. Use labels, navigation, and
ticket IDs from your own app rather than another app's controller.

## Ask a coding agent to set it up

Run Codex or Claude on the same Mac with terminal access. Open the approved
ticket in the website, then paste this prompt into the coding agent. The agent
can check the CLI, models, plugin, and driver. If `CuaDriver.app` needs
Accessibility or Screen Recording permissions, grant them in macOS System
Settings before the agent operates the window.

If your coding agent uses a hosted model, its provider may receive ticket
content that the agent reads during setup. Use an agent configuration that your
organization permits for that data.

```text
I have my organization's customer support website open in a macOS window.
Create a separate support controller directory in this workspace using
mere.run. Do not change the website's source code or require a new integration
in that app.

Use GLiNER2.5 Decide through `mere.run text classify` to suggest team and
priority. Use Ornith through `mere-computer-use` and Cua Driver to read and
operate only the selected window. Use local model inference. First check the
installed `mere.run` commands, model support, Computer Use plugin, Pi, and Cua
Driver. Install missing public components using their documented commands;
build mere.run from source if the installed CLI lacks GLiNER classification or
Ornith image input. Run `mere-computer-use doctor` for Ornith before acting.
Do not bypass hardware checks, accept model terms on my behalf, or infer that
macOS permissions are granted. Tell me the exact setting to change if needed.

Start with the approved ticket I have selected. Identify the correct window with
`mere-computer-use windows`; ask me to choose if several windows match. Inspect
its visible ticket ID, status, message, context, team and priority options,
tags, reply field, Save draft, and Send controls. Map how to find the open
ticket list. If a required control or stable ticket ID is unclear, ask me
before acting. Do not open other tickets during the first run.

Build an app adapter for ticket discovery, selection, observation, and final
state verification. Put the classification labels and action policy in an
editable local configuration file. Skip resolved tickets, No action outcomes,
and classifications that need human review. Recheck the ticket ID and content
after classification and before any change. Limit the number and duration of
actions per ticket. Write receipts with model scores, actions, elapsed time,
and before and after state. Make resume verify the current state so it cannot
duplicate a saved draft. Keep customer text, screenshots, credentials, and
receipts out of version control.

For the first run, operate only the selected approved ticket. Set the team
and priority only if its visible content supports the suggestion. Add
relevant tags and save a helpful reply as a draft. Never click Send, change
status, or process the rest of the inbox in this setup run. Verify the final
state independently through a read-only API or audit log if available.
Otherwise, use a fresh window observation and state that limit.
Stop on any mismatch.

Give me the controller source, setup commands, tests for skip, changed-state,
and resume behavior, and a one-ticket receipt. Report what worked, what needs
my input, and what remains unverified before I use it on more tickets.
```

After the first run, review the receipt and the ticket in the website. For a
larger pilot, use several labeled tickets and measure classification errors,
action success, and total wall time. Each additional app needs its own adapter
and verification rules.

Before expanding the pilot, confirm that the controller stops when a ticket
changes after classification, a second run does not add another draft, and the
ticket has no new sent reply. Check the final state in the app rather than
relying only on the agent's report.
