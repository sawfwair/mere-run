# Use mere.run in your business

Use local models and companion plugins to search records, prepare production
media, or evaluate a task-specific adapter. These guides explain the inputs,
setup, outputs, and checks for a small first run. Your team keeps control of
source files, access, and the decision to use a result.

Start with [Set up companion workflows](./plugin-workflows.md). Installing a
plugin doesn't install every model or service its workflows can use.

## Research and records

Choose a guide by the result you need:

| Your task | Guide | First result to check |
| --- | --- | --- |
| Collect public web evidence | [Collect research with traceable sources](./cited-research.md) | Saved snapshots match citation IDs and original URLs. |
| Find information across a shared drive | [Search your document archive](./shared-archive.md) | Search returns expected files, and cited claims match those sources. |
| Convert documents or review named fields | [Review documents and extract fields](./document-review.md) | Markdown or CSV values match the source document. |
| Prepare recording transcripts and subtitles | [Prepare transcripts and timed captions](./meeting-records.md) | Words, cue timing, speaker labels, and redaction survive review. |

## Media production

Keep generated and processed files usable in the next stage of your edit:

| Your task | Guide | First result to check |
| --- | --- | --- |
| Retake a shot region and continue editing | [Export an editable timeline](./editable-media.md) | The chosen interval and separate tracks import into your editor. |
| Extract, clean, or separate audio | [Prepare audio for an edit](./audio-post.md) | Reviewed WAVs match the intended content and delivery levels. |
| Package World clips or a single-image splat | [Hand off World captures and 3D assets](./visual-handoffs.md) | Media opens on the receiving workstation with its recorded conventions. |
| Cover sensitive regions before sharing images | [Review and export redacted images](./approved-media.md) | Every approved exported image covers the intended pixels. |

## Training and geospatial analysis

Use explicit comparisons and source lineage when adapting or analyzing data:

| Your task | Guide | First result to check |
| --- | --- | --- |
| Adapt a repeatable text task | [Train and evaluate a local text adapter](./local-text-adapters.md) | Paired behavior checks and held-out loss show what improved or regressed. |
| Run pinned satellite examples | [Review pinned satellite observations](./geospatial-review.md) | Source observations and grid match the candidate output's manifest. |

## Customer support

These guides work through an existing ticketing website. GLiNER2.5 Decide
suggests labels. A controller applies them through Cua Driver, or uses Ornith
and Computer Use when the task also needs a reply draft:

| Your task | Guide | First result to check |
| --- | --- | --- |
| Apply team, priority, and tags | [Route tickets with GLiNER and Cua Driver](./direct-ticket-triage.md) | The selected ticket has the intended labels. |
| Route a request and prepare a reply | [Triage tickets and save drafts](./customer-support.md) | The intended draft is saved and no reply is sent. |
| Connect the workflow to your app | [Adapt the support workflow](./adapt-support-workflow.md) | A one-ticket receipt matches the state in your app. |

## Plan a first run

1. Choose one approved input and an operator authorized to process it.
2. Define the expected result and how to compare it with the source.
3. Check the guide's plugin, model, service, and hardware prerequisites.
4. Run the small example, inspect artifacts, and record elapsed time and disk use.
5. Test interruption and failure before expanding the scope.

Local inference doesn't make every surrounding operation offline. Downloads,
web research, ticketing sites, and configured external services use the
network. Raw transcripts, document conversions, snapshots, and review previews
can contain unredacted data. Apply access, backup, and retention rules to
sources, derivatives, and run records. Check model terms for the intended use.

For platform setup, see [Getting started](../getting-started.md). For exact
plugin contracts and options, see the [companion plugin documentation](https://plugins-docs.mere.run/).
