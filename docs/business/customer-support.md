# Triage support tickets in an existing app

Use this guide to test one support ticket without adding a model call to the
ticketing app. A local classifier suggests the team and priority. A vision
language model reads the selected app window and operates its controls through
Computer Use. A person checks the saved draft before sending anything.

## See the Northline Care example

Northline Care is a fictional ticketing app. Its controls contain no GLiNER or
Ornith integration. In the recorded pilot, a customer reported a duplicate
charge and asked for help before the weekend. The workflow used:

| Component | Job in the pilot |
| --- | --- |
| GLiNER2.5 Decide | Classified the request as `Billing` with `High` priority |
| Ornith 35B | Read the page and selected ticket actions using the classification as input |
| Computer Use with Cua Driver | Changed the visible controls, added tags, and saved a reply as a draft |

One GLiNER classification request took 682 ms of wall time. That figure does
not include operating the app. A completed Ornith pass recorded 10 actions,
12 observations, and about nine minutes of wall time. The ticket was assigned
to Billing, marked High, tagged, and left with one saved draft and no sent
reply. These observations describe a synthetic pilot, not production accuracy
or throughput. The demo run records are not bundled with `mere.run`; the steps
that follow let you collect results for your own ticket.

## Prepare a one-ticket pilot

Use an Apple Silicon Mac with macOS 15 or later, a ticketing app you can open
in a macOS window, and a `mere.run` release that supports GLiNER classification
and Ornith image input. Ornith's recommended 4-bit model needs at least 32 GB
of unified memory; 48 GB is the conservative recommendation. Install Pi for
the Computer Use agent.

Check that your `mere.run` binary has the required command:

```bash
mere.run text classify --help
```

If the command is unavailable, build the source checkout and use its release
executable in the same terminal where you run the plugin:

```bash
git clone https://github.com/sawfwair/mere-run.git
cd mere-run
swift build -c release
export PATH="$PWD/.build/release:$PATH"
mere.run text classify --help
```

An older installed release might lack GLiNER or the Ornith image API support
that Computer Use needs. For build prerequisites, see
[Getting started](../getting-started.md#build-the-package).

Check your machine and pull the two models before you start the ticket:

```bash
mere.run model capabilities
mere.run model pull text-classify-gliner25-decide
mere.run model pull text-agent-ornith-35b-mlx-4bit
mere.run agent onboard --install-pi
```

Install Computer Use. The plugin installs the separately signed Cua Driver
app. Grant that app Accessibility and Screen Recording in macOS System
Settings, then check the installed model and driver:

```bash
mere.run plugin install mere-computer-use --yes
cua-driver permissions grant
mere-computer-use doctor --model text-agent-ornith-35b-mlx-4bit
mere-computer-use windows
```

The plugin installs neither Ornith nor GLiNER. It can start the local Ornith
API when a run begins. `doctor` checks readiness without changing a ticket.
For installation details and model limits, see the
[Computer Use plugin guide](https://github.com/sawfwair/mere-run-plugins/blob/main/docs/plugins/computer-use.md).
The plugin uses the MIT-licensed Cua Driver and excludes Cua's optional
perception components with AGPL dependencies. Review the separate model terms
before using the workflow with business data.

## Classify a sample ticket

Save the following fictional request as a `ticket-classification.json` file.
The labels are the actions your team wants to consider, not universal support
categories:

```json
{
  "text": "I placed one order for a jacket, but my card shows two charges. Please check before the weekend.",
  "tasks": [
    {
      "name": "team",
      "labels": ["Billing", "Fulfillment", "Care Team", "No action"]
    },
    {
      "name": "priority",
      "labels": ["High", "Normal", "Low"]
    }
  ]
}
```

Check that the request fits the checkpoint, then inspect its selected labels
and scores:

```bash
mere.run text classify --input ticket-classification.json --preflight --pretty
mere.run text classify --input ticket-classification.json --pretty
```

The model scores depend on your labels and examples. Review the result before
using it to change a ticket. Add a `No action` rule for resolved requests and
cases below your chosen confidence threshold. For classification options and
limits, see [GLiNER2.5 Decide](../runtime/gliner25-decide.md).

## Plan and run the app action

Open one sample ticket and use `mere-computer-use windows` to get that window's
process ID and window ID. The following task uses `Billing` and `High` as
reviewed example labels. Replace them with the labels you accepted for your
ticket:

```bash
mere-computer-use plan \
  --pid WINDOW_PID \
  --window-id WINDOW_ID \
  --model text-agent-ornith-35b-mlx-4bit \
  --task "Inspect this ticket. Suggestion: team Billing, priority High. Set those fields only if the visible request supports them. Add relevant tags. Write a helpful reply and save it as a draft. Do not send a reply or change the ticket status. Observe the final state." \
  --output ./support-pilot-run
```

Replace `WINDOW_PID` and `WINDOW_ID` with the numeric values from `windows`.
`plan` creates `./support-pilot-run/run.json` without operating the app. Review
the selected window and task in that file. Then run the plan:

```bash
mere-computer-use run ./support-pilot-run/run.json
```

The plugin limits actions to the chosen window and records the action count
and model report. It cannot certify that the ticket has the right values. In
the app, check the assigned team, priority, tags, draft content, and sent
reply count. Use the app's own API or audit log for an independent check when
one exists. If the run stops early, inspect the app before using
`mere-computer-use resume ./support-pilot-run/run.json`.

## Extend the pilot to an inbox

The one-ticket pilot above is reproducible with the public CLI and plugin. It
requires you to review the GLiNER result and pass accepted labels to Computer
Use. The Northline multi-ticket runner is a separate, unpublished demo harness;
the public plugin does not yet provide an inbox controller. To build one for
your app, use the following order:

1. Observe the selected ticket and read its message, order context, and status.
2. Classify the observed text with `mere.run text classify` or its loopback API.
3. Skip resolved tickets, `No action` results, and uncertain results that
   need a person to decide.
4. Give Ornith the accepted labels, the current page, and explicit limits on
   what it may change. Require **Save draft** rather than **Send**.
5. Verify the final state in the app before advancing to another ticket.
6. Save the initial observation, classification scores, actions, final state,
   and wall time. Resume only after checking what the previous run changed.

The page can change between classification and action. The controller must
recheck the selected ticket before every change. Use a small labeled set of
your own requests to measure classification errors and the full workflow's
time and success rate. For a pilot of label selection before any app action,
see [GLiNER2.5 Decide](../runtime/gliner25-decide.md). For app mapping, controller
requirements, and a prompt you can paste into Codex or Claude, see
[Adapt the support workflow](./adapt-support-workflow.md).

## Troubleshoot the first run

- If `doctor` reports missing permissions, grant Accessibility and Screen
  Recording to `CuaDriver.app`, then run `doctor` again.
- If `doctor` cannot start Ornith or the agent cannot process screenshots,
  confirm that your `mere.run` binary includes Qwen-family image API support
  and that `text-agent-ornith-35b-mlx-4bit` is installed.
- If `run.json` reports `verification: observation-only`, no action was
  recorded. Check the selected window and the app's ticket state before
  retrying. Do not infer success from the model's final text alone.
