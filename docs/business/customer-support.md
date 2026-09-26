# Triage tickets and save drafts

This guide is for a support team that wants to classify a ticket and save a
reply draft in its ticketing website. GLiNER2.5 Decide suggests the team and
priority. Ornith uses Computer Use to operate the selected macOS window. A
person checks the ticket before sending the reply.

To route tickets by using GLiNER and Cua Driver without Ornith, follow
[Route tickets with GLiNER and Cua Driver](./direct-ticket-triage.md).

Start with one ticket that your team authorizes for a draft-only run. You can
use the workflow without adding model code or an API to the ticketing app.
A separate controller connects the steps to an inbox.

## Prepare your Mac and ticket

Use an Apple Silicon Mac with macOS 15 or later. Open the ticketing website in
one window and select the ticket that you will process. Record its ticket ID,
status, customer message, and any order or account context needed for triage.
List the actual team, priority, and tag choices in your app. Decide which
outcomes require human review before an app action.

Ornith's 4-bit model needs at least 32 GB of unified memory; 48 GB is the
conservative recommendation. Check that your `mere.run` command supports
classification:

```bash
mere.run text classify --help
```

For source build prerequisites, see
[Getting started](../getting-started.md#build-the-package).

If classification is unavailable, build `mere.run` from source and use that
binary in the terminal where you run the plugin:

```bash
git clone https://github.com/sawfwair/mere-run.git
cd mere-run
swift build -c release
export PATH="$PWD/.build/release:$PATH"
mere.run text classify --help
```

To install the models and Pi, the local agent runner, run:

```bash
mere.run model capabilities
mere.run model pull text-classify-gliner25-decide
mere.run model pull text-agent-ornith-35b-mlx-4bit
mere.run agent onboard --install-pi
```

To install Computer Use and check the selected-window setup, run:

```bash
mere.run plugin install mere-computer-use --yes
cua-driver permissions grant
mere-computer-use doctor --model text-agent-ornith-35b-mlx-4bit
mere-computer-use windows
```

Grant `CuaDriver.app` Accessibility and Screen Recording permissions in macOS
System Settings if `doctor` reports that they are missing. The plugin can start
the local Ornith API for a run. It does not install the models. For driver
setup and supported models, see the
[Computer Use plugin guide](https://github.com/sawfwair/mere-run-plugins/blob/main/docs/plugins/computer-use.md).
If `doctor` reports that the installed CLI cannot process Ornith screenshots,
use the source build described in this guide.

## Classify the selected ticket

Create a private work directory outside a source repository:

```bash
umask 077
mkdir -p "$HOME/mere-run-support"
chmod 700 "$HOME/mere-run-support"
cd "$HOME/mere-run-support"
```

Create the `ticket-classification.json` file in that directory.
Use the ticket's actual message and relevant context as `text`. Replace each
uppercase label with a team or priority that your app accepts. Include every
team and priority that the controller may select. `No action` tells your
controller to skip the ticket; it does not need to be a team in the app.
The following file shows the request shape. It contains placeholders, so edit
it before running the command:

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

To check that the request fits the model and then get suggestions, run:

```bash
mere.run text classify --input ticket-classification.json --preflight --pretty
mere.run text classify --input ticket-classification.json --pretty
```

Review both selected labels and scores. The scores are not calibrated
probabilities. Skip a resolved ticket or a `No action` result. If the message
does not support the suggested team or priority, leave the app unchanged and
send the ticket for human review. For task limits and more request options, see
[GLiNER2.5 Decide](../runtime/gliner25-decide.md).

## Save a draft in the selected window

In the `mere-computer-use windows` output, find the process ID and window ID
for the window that contains your selected ticket. Use the team and priority
that you accepted after reviewing GLiNER's output. In the following command,
replace `WINDOW_PID`, `WINDOW_ID`, `TEAM_LABEL`, and `PRIORITY_LABEL` with those
values:

```bash
SUPPORT_TASK="Work only on the selected ticket.
Suggested team: TEAM_LABEL; priority: PRIORITY_LABEL.
Recheck the ticket ID and message before changing anything.
Set those fields only if the visible request supports them.
Add relevant tags. Write a reply and save it as a draft.
Do not send a reply or change ticket status. Observe the final state."
mere-computer-use plan \
  --pid WINDOW_PID \
  --window-id WINDOW_ID \
  --model text-agent-ornith-35b-mlx-4bit \
  --max-actions 12 \
  --task "$SUPPORT_TASK" \
  --output ./support-ticket-run
```

The `plan` command creates `./support-ticket-run/run.json` without changing
the app. Review its window and task. To let Ornith operate that window, run:

```bash
mere-computer-use run ./support-ticket-run/run.json
```

The task text instructs Ornith to save a draft, but the prompt is not a
technical block on the app's **Send** control. Use an account or app setting
that prevents sending during the first run when your ticketing system provides
one. Keep an operator present.

After the run, open the same ticket in the app. Check the team, priority,
tags, draft text, unchanged status, and sent-reply count. Use a
read-only API or audit log for an independent check when you have access. A
model report or action count alone does not verify the final ticket state.

If the run stops, inspect the ticket before doing more work. The
`mere-computer-use resume ./support-ticket-run/run.json` command displays the
saved record; it does not replay the actions. If work remains, create a new
plan only after checking whether a draft was already saved.

## Process more than one ticket

The commands in this guide require you to review GLiNER's result and pass the
accepted labels to Computer Use. The plugin does not discover or classify an
entire inbox by itself. To connect the steps for your website, follow
[Adapt the support workflow](./adapt-support-workflow.md). That guide covers
ticket discovery, changes between observation and action, verification,
resume, and a prompt for a coding agent.

Measure classification errors, completed actions, and total wall time on a
set of tickets from your own workflow before increasing the scope.
