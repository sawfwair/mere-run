# Triage tickets and save drafts

This guide routes one support ticket and saves a reply draft in your existing
ticketing website, without adding model code or an API to the app. GLiNER2.5
Decide suggests the team and priority. Ornith, a local vision-language model,
then uses the Computer Use plugin to set those fields, add tags, and save a
draft in the selected macOS window. A person reviews the ticket before anyone
sends the reply.

To route tickets without Ornith or reply drafts, follow
[Route tickets with GLiNER and Cua Driver](./direct-ticket-triage.md). To
connect these steps to a whole inbox, follow
[Adapt the support workflow](./adapt-support-workflow.md) after your first run.

## Before you begin

You need:

- An Apple Silicon Mac with macOS 15 or later and at least 32 GB of unified
  memory. Ornith's 4-bit model needs 32 GB; 48 GB is the conservative
  recommendation.
- One ticket that your team authorizes for a draft-only run.
- The team, priority, and tag choices that your app offers, and a decision
  about which outcomes need human review before anyone changes the app.
- [pipx](https://pipx.pypa.io/), which installs the plugin. If you use
  Homebrew, run `brew install pipx`.

## Install the tools

1. Check that your `mere.run` command supports classification:

   ```bash
   mere.run text classify --help
   ```

   If the command isn't available, build `mere.run` from source and use that
   binary in the terminal where you run the plugin. For build prerequisites,
   see [Getting started](../getting-started.md#build-the-package).

   ```bash
   git clone https://github.com/sawfwair/mere-run.git
   cd mere-run
   swift build -c release
   export PATH="$PWD/.build/release:$PATH"
   mere.run text classify --help
   ```

2. Check what your Mac supports, then install both models and Pi, the local
   agent runner:

   ```bash
   mere.run model capabilities
   mere.run model pull text-classify-gliner25-decide
   mere.run model pull text-agent-ornith-35b-mlx-4bit
   mere.run agent onboard --install-pi
   ```

3. Install the Computer Use plugin and check the window setup:

   ```bash
   mere.run plugin install mere-computer-use --yes
   cua-driver permissions grant
   mere-computer-use doctor --model text-agent-ornith-35b-mlx-4bit
   mere-computer-use windows
   ```

   The plugin install also installs the `CuaDriver.app` helper. If `doctor`
   reports missing permissions, grant `CuaDriver.app` Accessibility and Screen
   Recording access in macOS System Settings. If `doctor` reports that the
   installed CLI can't process Ornith screenshots, use the source build from
   step 1.

The plugin starts the local Ornith API for a run, but it doesn't install
models. For driver setup and supported models, see the
[Computer Use plugin guide](https://github.com/sawfwair/mere-run-plugins/blob/main/docs/plugins/computer-use.md).

## Classify the selected ticket

1. Open the ticketing website in one window and select the ticket. Record its
   ticket ID, status, customer message, and any order or account context that
   triage needs.

2. Create a private work directory outside any source repository:

   ```bash
   umask 077
   mkdir -p "$HOME/mere-run-support"
   chmod 700 "$HOME/mere-run-support"
   cd "$HOME/mere-run-support"
   ```

3. In that directory, create `ticket-classification.json`. Replace the `text`
   placeholder with the ticket's message and relevant context. Replace each
   uppercase label with a team or priority that your app accepts, and include
   every team and priority that the controller might choose. `No action` tells
   your controller to skip the ticket; it doesn't need to be a team in the app.

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

4. Check that the request fits the model, then get suggestions:

   ```bash
   mere.run text classify --input ticket-classification.json --preflight --pretty
   mere.run text classify --input ticket-classification.json --pretty
   ```

   The `--preflight` run validates the request and reports token use without
   loading the model.

5. Review both selected labels and their scores. The scores aren't calibrated
   probabilities. Skip a resolved ticket or a `No action` result. If the
   message doesn't support the suggested team or priority, leave the app
   unchanged and send the ticket for human review.

For task limits and more request options, see
[GLiNER2.5 Decide](../runtime/gliner25-decide.md).

## Save a draft in the selected window

1. In the `mere-computer-use windows` output, find the process ID and window ID
   of the window that shows your ticket.

2. Create a plan. Replace `WINDOW_PID`, `WINDOW_ID`, `TEAM_LABEL`, and
   `PRIORITY_LABEL` with the window IDs and the labels you accepted:

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

   The `plan` command writes `./support-ticket-run/run.json` without changing
   the app.

3. Review the window and task in `run.json`, then let Ornith operate the window:

   ```bash
   mere-computer-use run ./support-ticket-run/run.json
   ```

The task text tells Ornith to save a draft, but the prompt doesn't technically
block the app's **Send** control. For the first run, use an account or app
setting that prevents sending if your ticketing system has one, and keep an
operator present.

## Check the result

Open the same ticket in the app and check its team, priority, tags, draft text,
unchanged status, and sent-reply count. If you have a read-only API or audit
log, use it for an independent check. A model report or action count alone
doesn't verify the final ticket state.

If the run stops, inspect the ticket before you do anything else. The following
command displays the saved record; it doesn't replay the actions:

```bash
mere-computer-use resume ./support-ticket-run/run.json
```

If work remains, check whether a draft was already saved before you create a
new plan.

## Process more than one ticket

In this guide, you review GLiNER's result and pass the accepted labels to
Computer Use yourself. The plugin doesn't discover or classify an inbox on its
own. To connect the steps for your website, follow
[Adapt the support workflow](./adapt-support-workflow.md). That guide covers
ticket discovery, tickets that change between observation and action,
verification, resume, and a prompt for a coding agent.

Before you increase the scope, measure classification errors, completed
actions, and total wall time on a set of tickets from your own workflow.
