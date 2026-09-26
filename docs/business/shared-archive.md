# Search a shared archive

Use Archive Tools to make a local, searchable index of mixed documents and
images. The plugin reads source files without changing them. Start with its
fictional Harbourline Operations fixture, then apply the same steps to a
shared drive your team is allowed to index.

## See the Harbourline example

Harbourline is a fictional regional cold-storage company. Its sample archive
has 58 files across maintenance, safety, procurement, compliance, finance, and
capital planning. One question asks whether a Freezer 3 repair was covered by
warranty and when the warranty expires. A useful answer needs a repair record,
an invoice, and a vendor agreement. If the records do not establish whether
the charge was reimbursed, the answer must say that this part is unresolved.

Archive Tools converts files locally, reduces personally identifiable
information (PII) before embedding text, and stores a SQLite index. Its
bounded investigator can search that index and return source-linked claims.
A citation shows which file was retrieved; a person still checks whether the
file supports the claim.

## Install and check readiness

Install `mere.run` on an Apple Silicon Mac, then inspect your model capacity.
Archive Tools needs the local anonymization, embedding, captioning, and OCR
commands. Indexing images can download the default caption model on first use,
so complete a connected first run before trying to index offline.

```bash
mere.run model capabilities
mere.run model pull text-anonymize-privacy-filter
mere.run model pull vision-embed-qwen3-vl-2b
mere.run model pull vision-ocr-lighton
mere.run plugin install mere-archive-tools --yes
mere-archive-tools doctor
```

`doctor` checks the installed command surface, document conversion, and
SQLite full-text search. It does not index your files. The first image caption
can download `mlx-community/Qwen3-VL-2B-Instruct-4bit` unless that checkpoint
is already cached. For model installation and storage, see
[Model management](../runtime/model-management.md).

## Generate and index fictional files

Create the Harbourline fixture on your machine. The command writes a source
folder and `benchmark.json` manifest under `./harbourline-demo`:

```bash
mere-archive-tools benchmark prepare \
  --dataset harbourline-operations-archive \
  --output-dir ./harbourline-demo
```

Create a reduced-content index. Keep the database and run directory outside
the generated source folder:

```bash
mere-archive-tools index \
  --source ./harbourline-demo/source \
  --database ./harbourline.sqlite3 \
  --output-dir ./harbourline-index-run \
  --storage-tier safe-content
```

The `safe-content` tier retains short reduced summaries, keywords, and
embeddings. It does not retain the complete extracted body. The index command
records per-file errors and a durable `run.json` file. Review coverage before
you treat search results as complete:

```bash
mere-archive-tools stats --database ./harbourline.sqlite3
mere-archive-tools benchmark evaluate \
  ./harbourline-demo/benchmark.json \
  --database ./harbourline.sqlite3 \
  --output ./harbourline-evaluation.json
```

The evaluation checks source custody, retention, PII canaries, and retrieval
against 30 fixture questions. Review the report's file errors and retrieval
metrics. Its canary test checks exact fictional values; it does not prove that
all sensitive information is absent.

## Search and investigate

Search for the repair before using an agent. The result identifies source
paths and whether each file is still available:

```bash
mere-archive-tools search \
  --database ./harbourline.sqlite3 \
  --query "Freezer 3 repair warranty and vendor agreement"
```

For a question that requires several searches, install Pi and the default
local investigation model. The model is separate from the plugin:

```bash
mere.run agent onboard --install-pi
mere.run model pull text-chat-bonsai-27b-2bit
mere-archive-tools investigate \
  --database ./harbourline.sqlite3 \
  --question "Was the Freezer 3 repair covered by warranty, and when does that warranty expire?" \
  --diagnostics ./harbourline-investigation-metrics.json
```

The investigator can make up to four searches with five results per search by
default. It can cite only paths returned by those searches. Read its supported
and unresolved claims, then open the cited source files. The plugin checks
citation membership, not whether a passage proves the claim. An installed
model passed this fictional case on a 128 GB test Mac; that result does not
qualify other archives or smaller machines.

## Apply the workflow to your files

Choose a mounted source folder that the operator may read. Keep generated
artifacts outside that folder and start with `safe-content`:

```bash
mere-archive-tools index \
  --source /Volumes/Shared \
  --database ./shared-archive.sqlite3 \
  --output-dir ./shared-archive-run \
  --storage-tier safe-content
```

For an interrupted run, inspect `./shared-archive-run/run.json` and use
`mere-archive-tools resume ./shared-archive-run/run.json`. Later index runs
compare file metadata and hashes, reuse identical content, and remove records
for source paths that disappeared. They do not delete source files.

The plugin does not reproduce the drive's access-control list. Protect the
SQLite database and search interface under your organization's access,
encryption, backup, and retention rules. Its default image mode embeds reduced
captions and OCR; `--image-index visual` also embeds source pixels and requires
a separate privacy review. For storage tiers, limits, and source formats, see
the [Archive Tools plugin guide](https://github.com/sawfwair/mere-run-plugins/blob/main/docs/plugins/archive-tools.md).

## Troubleshoot the first index

- If `doctor` reports a missing command, update `mere.run` before indexing.
  `doctor` checks command availability, not every model checkpoint.
- If the first image takes longer than a text file, check whether `mere.run`
  is downloading its default caption model. Complete that step while the Mac
  has network access.
- If a scanned PDF appears in the file-error list, render its pages to images
  and index those images. Archive Tools rejects hosted OCR during document
  conversion.
- If search returns a path marked unavailable, mount the source drive again.
  The index does not contain a copy of the original file.
