# Route tickets with GLiNER and Cua Driver

This guide sets the team and priority on support tickets in your existing
ticketing website. GLiNER2.5 Decide suggests a team and priority from the
ticket text. A small controller then uses Cua Driver to read the selected
window and apply the labels that your team approves. This path doesn't use
Ornith, Pi, or the `mere-computer-use run` agent loop, and GLiNER doesn't write
replies.

To prepare reply drafts with a language model, follow
[Triage tickets and save drafts](./customer-support.md) instead.

## Before you begin

You need:

- An Apple Silicon Mac with macOS 15 or later.
- Your ticketing website open in one window, and one ticket that you're
  allowed to update.
- [pipx](https://pipx.pypa.io/), which installs the plugin. If you use
  Homebrew, run `brew install pipx`.

## Install the tools

1. Check that your `mere.run` command includes classification, then install the
   GLiNER model:

   ```bash
   mere.run text classify --help
   mere.run model capabilities
   mere.run model pull text-classify-gliner25-decide
   ```

   If `text classify` isn't available, build `mere.run` from source as
   described in
   [Triage tickets and save drafts](./customer-support.md#install-the-tools).

2. Install the Computer Use plugin to get its pinned Cua Driver app. This
   workflow calls the driver directly, so you don't need to pull Ornith or
   install Pi.

   ```bash
   mere.run plugin install mere-computer-use --yes
   cua-driver permissions grant
   cua-driver permissions status --json
   cua-driver call list_windows '{}'
   ```

   If `cua-driver` isn't available after the plugin installs, run
   `mere-computer-use setup --yes`. When macOS prompts you, grant
   `CuaDriver.app` Accessibility and Screen Recording access in System
   Settings.

For driver setup and the license boundary, see the
[Computer Use plugin guide](https://github.com/sawfwair/mere-run-plugins/blob/main/docs/plugins/computer-use.md).

## Map one ticket

1. In the `list_windows` result, find the process ID (`pid`) and window ID
   (`window_id`) of the ticketing website. If more than one window matches,
   stop and choose one.
2. Select the approved ticket in that window. Record its stable ID, status,
   message, current team, priority, and tags.
3. Record the exact team, priority, and tag choices that your app offers. Map
   each GLiNER label to one app choice, and leave unsupported labels unmapped.
4. Choose how to verify changes. Use a read-only ticket API or audit log if
   your organization has one. Otherwise, take a fresh driver observation, and
   record that verification covers only the visible state.
5. Inspect the window's accessibility elements. Enter the IDs from
   `list_windows` when prompted:

   ```bash
   printf 'Process ID: '; read -r WINDOW_PID
   printf 'Window ID: '; read -r WINDOW_ID
   cua-driver call get_window_state \
     "{\"pid\":$WINDOW_PID,\"window_id\":$WINDOW_ID,\"max_elements\":300,\"include_screenshot\":false}"
   ```

   The response lists element labels, values, roles, and handles for the
   window.

If the ticket ID or message is missing or ambiguous, stop and map the app with
an operator. A screenshot can help an operator inspect a custom control, but
GLiNER can't classify pixels. Don't infer ticket text from a control that the
driver didn't read.

## Classify the ticket

1. Create a private work directory outside your source repository:

   ```bash
   umask 077
   mkdir -p "$HOME/mere-run-support"
   chmod 700 "$HOME/mere-run-support"
   cd "$HOME/mere-run-support"
   ```

2. Create `ticket-classification.json` with the observed message and relevant
   context. Replace every uppercase placeholder with text or an exact label
   from your app, and include every team and priority that the controller
   might choose:

   ```json
   {
     "text": "REPLACE_WITH_TICKET_MESSAGE_AND_RELEVANT_CONTEXT",
     "tasks": [
       {
         "name": "team",
         "labels": ["TEAM_LABEL_A", "TEAM_LABEL_B", "No action"]
       },
       {
         "name": "priority",
         "labels": ["PRIORITY_LABEL_A", "PRIORITY_LABEL_B"]
       }
     ]
   }
   ```

3. Check the request, then run the model:

   ```bash
   mere.run text classify --input ticket-classification.json --preflight --pretty
   mere.run text classify --input ticket-classification.json --pretty
   ```

4. Approve or reject each suggested label before you change the app. GLiNER
   scores aren't calibrated probabilities. Skip resolved tickets, `No action`
   results, and requests that need human review. `No action` tells the
   controller to skip a ticket; it isn't a team to set in the app.

For request limits, see [GLiNER2.5 Decide](../runtime/gliner25-decide.md).

## Apply and verify approved labels

Build the controller around the app's visible fields. Before you write it,
inspect the exact input schemas of the driver tools you'll use:

```bash
cua-driver describe get_window_state
cua-driver describe set_value
cua-driver describe click
```

For each approved ticket, the controller follows this sequence:

1. Read the selected window again. Stop if the ticket ID, message, status, or
   window differs from the observation used for classification.
2. Find the exact team or priority control in the fresh accessibility result.
   For a supported select control, use Cua Driver's `set_value` with its
   current element handle and the app's exact option text.
3. If the control needs clicks, click its current element handle, then read
   the window again before you choose an option from the open menu. Never
   reuse an element handle or screenshot coordinate after a new observation.
4. Read the ticket after each change. Verify the selected team, priority, and
   tags before you move to another ticket. Stop on an unknown or mismatched
   result.
5. If a read-only API or audit log is available, check the ticket there too.
   Record the initial state, GLiNER output, approved labels, driver actions,
   final state, verification result, and wall time.

Cua Driver changes the website through its visible controls; it doesn't decide
which GLiNER labels are safe to apply. Keep an action limit for each ticket,
and never click **Send** or change ticket status.

For an inbox loop, discover open ticket IDs, classify one ticket at a time, and
verify each final state before you open the next ticket. On restart, compare
the saved receipt with the current ticket, and skip a ticket only after you
verify its labels. Before you increase the number of tickets, test changed
tickets, missing controls, menu failures, and interrupted runs.

## Ask Codex or Claude to build the controller

Open an approved ticket in your website, then paste the following prompt into
Codex or Claude on the same Mac. A hosted coding model might receive any ticket
content that it reads during setup, so use a configuration that your
organization permits for that data.

```text
I have my organization's ticketing website open in one macOS window. Build a
separate local support controller for that existing website. Use GLiNER2.5
Decide through `mere.run text classify` for team and priority suggestions.
Use Cua Driver directly to observe and operate the selected window. Do not
use Ornith, Pi, `mere-computer-use plan/run`, a browser API, or changes to the
website's source code. Use a read-only ticket API or audit log for independent
verification only if one is already available to me.

First check `mere.run text classify --help`, model support, the installed
GLiNER model, Cua Driver, and its macOS permissions. Install missing public
components using their documented commands. If the installed mere.run lacks
classification, build it from source in release mode. Do not bypass hardware
checks or infer that macOS permissions are granted. Tell me which setting to
change if a permission is missing.

Start with only the approved ticket I have selected. Use
`cua-driver call list_windows '{}'` to identify its exact process and window.
Ask me to choose if several windows match. Use `get_window_state` to read the
ticket ID, message, status, existing labels, and the app's team, priority,
and tag options. Inspect the Cua Driver tool schemas with `cua-driver describe`
before using an action. Stop if the ticket text or a required control cannot
be identified from the selected window.

Create an editable mapping from GLiNER labels to the app's exact choices and
a policy for labels that need human review. Do not treat model scores as
calibrated probabilities or invent a universal score threshold. For the
first ticket, show me the classification and proposed changes before acting.
Re-read the ticket ID, message, status, and window before each change. Use
fresh Cua Driver element handles or screenshot coordinates for actions. Use
`set_value` for a supported select control; otherwise open its menu, observe
again, and choose the exact option. Verify each change from a fresh window
observation. Never click Send, write a reply, or change ticket status.

After the one-ticket run, build a bounded loop over open tickets. Process
one ticket at a time. Stop on a changed ticket, missing control, uncertain
classification, or failed verification. Save a private receipt for each
ticket with its initial state, model output, approved labels, driver actions,
final state, verification result, and elapsed wall time. On restart, verify
the current ticket before skipping or retrying it. Keep customer text,
screenshots, credentials, and receipts out of version control.

Give me the controller source, setup commands, tests for skip and changed-state
behavior, and a receipt for the first ticket. State what you verified in the
app and what still needs my review before the inbox loop runs.
```
