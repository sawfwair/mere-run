# Search your document archive

This guide is for a team with a mounted document folder that its operator may
read and index. Archive Tools creates a local SQLite index of documents and
images without changing the source files. You can search that index and
investigate questions that need evidence from several files.

Choose a folder that contains work your team needs to search. Start with a
small, approved part of the archive. Keep the database and run records outside
the source folder. The plugin does not copy the source drive's access rules to
the index, so protect the index under your organization's access and retention
policy.

## Install Archive Tools

On an Apple Silicon Mac, install `mere.run` and check model capacity. Archive
Tools uses local text anonymization, image embedding, optical character
recognition (OCR), and image captioning. The first caption can download its
default checkpoint, so complete the first index while connected to the
network.

```bash
mere.run model capabilities
mere.run model pull text-anonymize-privacy-filter
mere.run model pull vision-embed-qwen3-vl-2b
mere.run model pull vision-ocr-lighton
mere.run plugin install mere-archive-tools --yes
mere-archive-tools doctor
```

The `doctor` command checks command availability, document conversion, and
SQLite full-text search. It does not confirm that every model is installed or
index a file. For model storage, see
[Model management](../runtime/model-management.md).

## Index an approved folder

In the following commands, replace `/Volumes/Shared/APPROVED_FOLDER` with the
mounted folder that your operator may index. Choose a work directory outside
that folder. The `umask` setting limits access to newly created local files:

```bash
ARCHIVE_SOURCE="/Volumes/Shared/APPROVED_FOLDER"
ARCHIVE_WORK_DIR="$HOME/mere-run-archive"
umask 077
mkdir -p "$ARCHIVE_WORK_DIR"
chmod 700 "$ARCHIVE_WORK_DIR"
mere-archive-tools index \
  --source "$ARCHIVE_SOURCE" \
  --database "$ARCHIVE_WORK_DIR/archive.sqlite3" \
  --output-dir "$ARCHIVE_WORK_DIR/first-run" \
  --storage-tier safe-content
```

The `safe-content` tier stores short, reduced summaries, keywords, and
embeddings—numeric representations used for similarity search. It does not
store the complete extracted document body. It still
retains file paths and derived information that may be sensitive. The plugin
reduces personally identifiable information (PII) before embedding text, but
that reduction cannot guarantee that every sensitive value is removed.

To check coverage and file errors, inspect the index statistics and run
record:

```bash
mere-archive-tools stats --database "$ARCHIVE_WORK_DIR/archive.sqlite3"
cat "$ARCHIVE_WORK_DIR/first-run/run.json"
```

Resolve file errors before treating search results as complete. The plugin
supports common text, Office, PDF, and image files. It rejects hosted OCR
during document conversion, so a scanned PDF may appear as a file error. For
source formats and storage tiers, see the
[Archive Tools plugin guide](https://github.com/sawfwair/mere-run-plugins/blob/main/docs/plugins/archive-tools.md).

## Search and check the source

Set `ARCHIVE_QUESTION` to a question your team needs answered from this folder.
Replace the uppercase placeholder before running the command:

```bash
ARCHIVE_QUESTION="REPLACE_WITH_A_QUESTION_ABOUT_YOUR_ARCHIVE"
mere-archive-tools search \
  --database "$ARCHIVE_WORK_DIR/archive.sqlite3" \
  --query "$ARCHIVE_QUESTION"
```

Search results identify source paths and whether each file is available.
Open the retrieved files and check that they cover the question. If a result
points to an unavailable path, mount the source drive again. The index does
not contain a copy of the original file.

If the question requires several searches, install Pi, the local agent runner,
and the default investigation model. Follow the support check from
`mere.run model capabilities`;
do not bypass it to fit a model on an undersized Mac:

```bash
mere.run agent onboard --install-pi
mere.run model pull text-chat-bonsai-27b-2bit
mere-archive-tools investigate \
  --database "$ARCHIVE_WORK_DIR/archive.sqlite3" \
  --question "$ARCHIVE_QUESTION" \
  --diagnostics "$ARCHIVE_WORK_DIR/investigation-metrics.json"
```

The investigator can make up to four searches with five results per search by
default. It returns source-linked claims and unresolved points. Check each
claim in the cited file before using the answer. A citation confirms that a
path was returned by search; it does not prove that the file supports the
claim. If the investigation model does not fit your Mac, use search without
the investigator.

## Maintain and evaluate the index

For an interrupted index run, inspect the manifest before resuming it:

```bash
mere-archive-tools resume "$ARCHIVE_WORK_DIR/first-run/run.json"
```

To refresh the archive after source files change, run `index` again with the
same database and a new output directory. The plugin reuses identical content
and removes index records for paths that disappeared. It does not delete
source files.

Check retrieval against questions whose answers and source files your team
already knows. Record missing files, incorrect matches, claim errors, and
elapsed time. Review the index's access, backup, and retention settings before
you let more people query it. Image mode embeds reduced captions and OCR by
default; `--image-index visual` also embeds source pixels and needs a separate
privacy review.
