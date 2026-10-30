# Use mere.run in your business

These guides show a team how to route support tickets in an existing ticketing
website, or how to search a shared document archive, with models that run on
your own Mac. `mere.run` doesn't need an inference API key.

## Choose a guide

Each guide starts with one ticket or one folder that you can check by hand.
The following table lists the guides and the first result to check:

| Your task | Guide | First result to check |
| --- | --- | --- |
| Route support requests with GLiNER and Cua Driver | [Route tickets with GLiNER and Cua Driver](./direct-ticket-triage.md) | The selected ticket has the intended team, priority, and tags. |
| Route requests and prepare reply drafts with Ornith | [Triage tickets and save drafts](./customer-support.md) | The selected ticket has the intended team, priority, tags, and saved draft. No reply was sent. |
| Connect ticket triage to your own app | [Adapt the support workflow](./adapt-support-workflow.md) | A one-ticket receipt matches the ticket in the app. |
| Find information across a shared drive | [Search your document archive](./shared-archive.md) | Search returns the expected source files, and cited claims match those files. |

For support, GLiNER2.5 Decide suggests a team and priority from the ticket
text. A small controller that drives Cua Driver directly can apply approved
labels without a chat model. To also prepare reply drafts, use the Ornith model
through the Computer Use plugin. Both support guides include a prompt that you
can paste into Codex or Claude while your website is open.

For documents, Archive Tools indexes an approved folder without changing its
source files. You can search the index or ask a question that needs evidence
from several files. Your team controls who can read the source folder and the
derived index.

## Plan your first run

1. Choose an operator and a small set of work that the operator is allowed to
   process.
2. Define the expected result and how you'll check it in the source app or
   files.
3. Install the models and plugin that your guide lists.
4. Run one ticket or one folder, inspect the result, and record the elapsed
   time.
5. Expand the scope only after you've checked errors, access, and recovery.

Model inference runs on your Mac. Installing software and pulling models use
the network, and a ticketing website can still send data through its own
service. An archive index holds information derived from your files, even with
reduced retention. Apply your organization's access and retention rules to the
source, the index, and the run records.

For platform setup, see [Getting started](../getting-started.md) and
[Model management](../runtime/model-management.md).
