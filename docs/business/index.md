# mere.run for business

Start with work your team already does in an app or a folder. `mere.run` runs
models on your machine. Companion plugins connect those models to a selected
window or a local collection of files. You can test one task before you build a
larger workflow.

These guides use fictional data and show the checks needed to decide whether a
result is useful. They do not require an inference API key.

## Choose a first workflow

| Task | Start with | What the pilot verifies |
| --- | --- | --- |
| Route a support request and prepare a reply | [Triage support tickets](./customer-support.md) | Classification, visible app actions, saved draft, and no sent reply |
| Find evidence across years of mixed files | [Search a shared archive](./shared-archive.md) | Index coverage, source-linked search results, and unresolved claims |

The support guide uses a fictional Northline Care ticket to explain how
GLiNER2.5 Decide, Ornith, and Cua Driver work together. The archive guide uses
Archive Tools and its locally generated Harbourline fixture. Both guides
include commands you can run with the public CLI and plugins.

## Pilot with your own work

1. Pick one task with a result a person can check, such as a saved draft or a
   source document that answers a specific question.
2. Define the allowed actions and the stop condition before starting an agent.
   Keep sending, deleting, and other irreversible actions outside the first
   pilot.
3. Run the workflow on fictional or approved sample data. Record model output,
   action history, elapsed time, and the final state in the source app or files.
4. Compare the result with your own expected answer. Check the underlying
   ticket or cited documents; a model report alone is not verification.
5. Expand to more cases only after you know which errors need human review.

Inference runs locally, but installation and model pulls use the network.
Computer Use can enter information into the selected app, so that app's own
network and access rules still apply. Archive Tools keeps its index locally;
your organization remains responsible for source permissions, index access,
backup, and retention.

For installation and model sizing, see [Getting started](../getting-started.md)
and [Model management](../runtime/model-management.md). For the plugin
installation boundary, see [Companion plugins](../plugins.md).
