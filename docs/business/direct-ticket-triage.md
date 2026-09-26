# Route tickets with GLiNER and Cua Driver

This guide is for a support team that wants to route tickets in its existing
website. GLiNER2.5 Decide suggests a team and priority from ticket text. A
small controller uses Cua Driver to read the selected window and apply labels
that your team approves. This path does not use Ornith, Pi, or the
`mere-computer-use run` agent loop.

Start with one ticket that you may update. GLiNER does not write replies. To
prepare reply drafts with a language model, follow
[Triage tickets and save drafts](./customer-support.md).

## Prepare the tools

Use an Apple Silicon Mac with macOS 15 or later. Open your ticketing website in
one window. Check that your `mere.run` command includes classification:

```bash
mere.run text classify --help
mere.run model capabilities
mere.run model pull text-classify-gliner25-decide
```

If `text classify` is unavailable, follow the source-build instructions in
[Triage tickets and save drafts](./customer-support.md#prepare-your-mac-and-ticket).

Install Computer Use to obtain its pinned Cua Driver app. The direct workflow
calls the driver itself, so you do not need to pull Ornith or install Pi:

```bash
mere.run plugin install mere-computer-use --yes
cua-driver permissions grant
cua-driver permissions status --json
cua-driver call list_windows '{}'
```

If `cua-driver` is unavailable after installing the plugin, run
`mere-computer-use setup --yes`. Grant `CuaDriver.app` Accessibility and Screen
Recording access in macOS System Settings when prompted. For the driver setup
and license boundary, see the
[Computer Use plugin guide](https://github.com/sawfwair/mere-run-plugins/blob/main/docs/plugins/computer-use.md).

## Map one ticket

1. In the `list_windows` result, identify the process ID (`pid`) and window ID
   (`window_id`) of the ticketing website. Stop if more than one window matches.
2. Select the approved ticket in that window. Record its stable ID, status,
   message, current team, priority, and tags.
3. Record the exact team, priority, and tag choices that your app offers. Map
   each GLiNER label to one app choice. Leave unsupported labels unmapped.
4. Choose a way to verify changes. Prefer a read-only ticket API or audit log
   when your organization provides one. Otherwise, take a fresh driver
   observation and record that verification is limited to the visible state.

To inspect the selected window's accessibility elements, enter its IDs from
`list_windows` when prompted:

```bash
printf 'Process ID: '; read -r WINDOW_PID
printf 'Window ID: '; read -r WINDOW_ID
cua-driver call get_window_state \
  "{\"pid\":$WINDOW_PID,\"window_id\":$WINDOW_ID,\"max_elements\":300,\"include_screenshot\":false}"
```

The response includes element labels, values, roles, and handles for the
selected window. If the ticket ID or message is missing or ambiguous, stop and
map the app with an operator. A screenshot can help an operator inspect a
custom control, but GLiNER cannot classify pixels. Do not infer ticket text
from a control that the driver did not read.

## Classify the ticket

Create a private work directory outside your source repository:

```bash
umask 077
mkdir -p "$HOME/mere-run-support"
chmod 700 "$HOME/mere-run-support"
cd "$HOME/mere-run-support"
```

Create `ticket-classification.json` with the observed message and relevant
context. Replace every uppercase placeholder with text or an exact label from
your app. Include every team and priority that the controller may choose:

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

Run the model and review its suggestions:

```bash
mere.run text classify --input ticket-classification.json --preflight --pretty
mere.run text classify --input ticket-classification.json --pretty
```

GLiNER scores are not calibrated probabilities. For the first ticket, approve
or reject each suggested label before changing the app. Skip resolved tickets,
`No action` results, and requests that need human review. `No action` tells the
controller to skip a ticket; it is not a team to set in the app. For request limits,
see [GLiNER2.5 Decide](../runtime/gliner25-decide.md).

## Apply and verify approved labels

Build the controller around the app's visible fields. For each approved
ticket, use this sequence:

1. Read the selected window again. Stop if the ticket ID, message, status, or
   window differs from the observation used for classification.
2. Find the exact team or priority control in the fresh accessibility result.
   For a supported select control, use Cua Driver's `set_value` with its current
   element handle and the app's exact option text.
3. If the control needs clicks, click its current element handle. Read the
   window again before selecting an option from the opened menu. Never reuse
   an element handle or screenshot coordinate after a new observation.
4. Read the ticket after each change. Verify the selected team, priority, and
   tags before moving to another ticket. Stop on an unknown or mismatched
   result.
5. Check the ticket through the read-only API or audit log if one is available.
   Record the initial state, GLiNER output, approved labels, driver actions,
   final state, verification result, and wall time.

Use `cua-driver describe get_window_state`, `cua-driver describe set_value`,
and `cua-driver describe click` to inspect the exact input schemas before
writing the controller. Cua Driver changes the website through its visible
controls; it does not decide which GLiNER labels are safe to apply. Keep a
per-ticket action limit, and do not click **Send** or change ticket status.

For an inbox loop, discover open ticket IDs, classify one ticket at a time,
and verify each final state before opening the next. On restart, compare the
saved receipt with the current ticket. Skip an already completed ticket only
after verifying its labels. Test changed tickets, missing controls, menu
failures, and interrupted runs before increasing the number of tickets.

## Ask Codex or Claude to build the controller

Open an approved ticket in your website, then paste this prompt into Codex or
Claude on the same Mac. A hosted coding model might receive any ticket content
that it reads during setup. Use a configuration your organization permits for
that data.

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
