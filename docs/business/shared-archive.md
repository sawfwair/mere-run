# Search your document archive

This guide builds a searchable local index of a shared document folder, then
uses it to find files and answer questions that need evidence from several
files. Archive Tools creates a SQLite index of documents and images without
changing the source files.

Start with a small, approved part of the archive that your team needs to
search. Keep the database and run records outside the source folder. The index
doesn't inherit the source drive's access rules, so protect it under your
organization's access and retention policy.

## Before you begin

You need:

- An Apple Silicon Mac with `mere.run` installed.
- A mounted folder that the operator is allowed to read and index.
- A network connection for the first index run. Image captioning downloads its
  default checkpoint the first time it runs.
- [pipx](https://pipx.pypa.io/), which installs the plugin. If you use
  Homebrew, run `brew install pipx`.

## Install Archive Tools

Archive Tools uses local text anonymization, image embedding, optical
character recognition (OCR), and image captioning. To check what your Mac
supports, install the models, and install the plugin, run:

```bash
mere.run model capabilities
mere.run model pull text-anonymize-privacy-filter
mere.run model pull vision-embed-qwen3-vl-2b
mere.run model pull vision-ocr-lighton
mere.run plugin install mere-archive-tools --yes
mere-archive-tools doctor
```

The `doctor` command checks command availability, document conversion, and
SQLite full-text search. It doesn't confirm that every model is installed, and
it doesn't index a file. For model storage, see
[Model management](../runtime/model-management.md).

## Index an approved folder

1. Set the source and work directories. Replace
   `/Volumes/Shared/APPROVED_FOLDER` with the mounted folder that the operator
   is allowed to index. The work directory must be outside that folder.

   ```bash
   ARCHIVE_SOURCE="/Volumes/Shared/APPROVED_FOLDER"
   ARCHIVE_WORK_DIR="$HOME/mere-run-archive"
   ```

2. Create a private work directory and build the index. The `umask` setting
   limits access to newly created local files.

   ```bash
   umask 077
   mkdir -p "$ARCHIVE_WORK_DIR"
   chmod 700 "$ARCHIVE_WORK_DIR"
   mere-archive-tools index \
     --source "$ARCHIVE_SOURCE" \
     --database "$ARCHIVE_WORK_DIR/archive.sqlite3" \
     --output-dir "$ARCHIVE_WORK_DIR/first-run" \
     --storage-tier safe-content
   ```

3. Check coverage and file errors in the index statistics and run record:

   ```bash
   mere-archive-tools stats --database "$ARCHIVE_WORK_DIR/archive.sqlite3"
   cat "$ARCHIVE_WORK_DIR/first-run/run.json"
   ```

The `safe-content` tier stores short reduced summaries, keywords, and
embeddings, which are numeric representations used for similarity search. It
doesn't store the complete extracted document body, but it still keeps file
paths and derived information that might be sensitive. The plugin reduces
personally identifiable information (PII) before it embeds text, but that
reduction can't guarantee that every sensitive value is removed.

Resolve file errors before you treat search results as complete. The plugin
supports common text, Office, PDF, and image files. It rejects hosted OCR
during document conversion, so a scanned PDF might appear as a file error. For
source formats and storage tiers, see the
[Archive Tools plugin guide](https://github.com/sawfwair/mere-run-plugins/blob/main/docs/plugins/archive-tools.md).

## Search and check the source

1. Set `ARCHIVE_QUESTION` to a question that your team needs answered from this
   folder, then search:

   ```bash
   ARCHIVE_QUESTION="REPLACE_WITH_A_QUESTION_ABOUT_YOUR_ARCHIVE"
   mere-archive-tools search \
     --database "$ARCHIVE_WORK_DIR/archive.sqlite3" \
     --query "$ARCHIVE_QUESTION"
   ```

2. Open the files that search returns and check that they cover the question.
   Results show each source path and whether the file is available. If a
   result points to an unavailable path, mount the source drive again; the
   index doesn't contain a copy of the original file.

## Investigate a question across files

If a question needs several searches, use the investigator. It needs Pi, the
local agent runner, and its default model. Check the support result from
`mere.run model capabilities` first, and don't bypass it to fit the model on a
Mac with too little memory.

```bash
mere.run agent onboard --install-pi
mere.run model pull text-chat-bonsai-27b-2bit
mere-archive-tools investigate \
  --database "$ARCHIVE_WORK_DIR/archive.sqlite3" \
  --question "$ARCHIVE_QUESTION" \
  --diagnostics "$ARCHIVE_WORK_DIR/investigation-metrics.json"
```

By default, the investigator makes up to four searches with five results per
search. It returns source-linked claims and unresolved points. Check each claim
in the cited file before you use the answer: a citation shows that search
returned the path, not that the file supports the claim. If the investigation
model doesn't fit your Mac, use search without the investigator.

## Maintain and evaluate the index

To continue an interrupted index run, pass its manifest to `resume`:

```bash
mere-archive-tools resume "$ARCHIVE_WORK_DIR/first-run/run.json"
```

To refresh the index after source files change, run `index` again with the same
database and a new output directory. The plugin reuses identical content and
removes index records for paths that disappeared. It never deletes source
files.

Check retrieval with questions whose answers and source files your team already
knows. Record missing files, incorrect matches, claim errors, and elapsed time.
Before you let more people query the index, review its access, backup, and
retention settings.

By default, image indexing embeds reduced captions and OCR text. The
`--image-index visual` option also embeds the source pixels, so it needs a
separate privacy review.
