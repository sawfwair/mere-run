# Use mere.run in your business

These guides are for teams that want to use local models with a ticketing
website or a shared document archive. Start with work that your team can
verify in the source app or files. `mere.run` does not require an inference
API key.

## Choose a workflow

| Your task | Guide | First result to check |
| --- | --- | --- |
| Route support requests with GLiNER and Cua Driver | [Route tickets with GLiNER and Cua Driver](./direct-ticket-triage.md) | The selected ticket has the intended team, priority, and tags. |
| Route requests and prepare reply drafts with Ornith | [Triage tickets and save drafts](./customer-support.md) | The selected ticket has the intended team, priority, tags, and saved draft. No reply was sent. |
| Find information across a shared drive | [Search your document archive](./shared-archive.md) | Search returns the expected source files, and cited claims match those files. |

For support, GLiNER2.5 Decide suggests a team and priority. A direct Cua
Driver controller can apply approved labels without Ornith. To prepare reply
drafts, use Ornith through the Computer Use plugin. Both support guides include
or link to a prompt that you can paste into Codex or Claude while your website
is open.

For documents, Archive Tools indexes an approved folder without changing its
source files. You can search the index or ask a question that requires several
files. Your team controls who can read the source folder and the derived index.

## Prepare your first run

1. Choose an operator and a small set of work that the operator may process.
2. Define the expected result and how you will check it in the source app or
   files.
3. Install the models and plugin required by the guide you chose.
4. Run one ticket or one folder, inspect the result, and record elapsed time.
5. Expand the scope after you have checked errors, access, and recovery.

`mere.run` model inference runs on your Mac. Installing software and pulling
models use the network. A ticketing website can still send data through its
own service.
An archive index contains derived information from your files, even when you
choose reduced retention. Apply your organization's access and retention rules
to both the source and the run records.

For platform setup, see [Getting started](../getting-started.md) and
[Model management](../runtime/model-management.md).
