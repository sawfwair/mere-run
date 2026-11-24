# Collect research with traceable sources

Use Archive Tools to collect public evidence for a supplier comparison, policy
review, or research brief. The workflow searches through SearXNG, reads selected
pages through Firecrawl Local, and saves Markdown snapshots with stable citation
IDs. By default, it also builds a searchable local index.

This workflow produces a source collection. It doesn't write or verify a
finished report. Open the saved sources and check any claims you use.

## Before you begin

Follow [Set up companion workflows](./plugin-workflows.md). Install Archive
Tools, SearXNG Search, and Firecrawl Local:

```bash
mere.run plugin install mere-archive-tools --yes
mere.run plugin install mere-searxng-search --yes
mere.run plugin install mere-firecrawl-local --yes
mere.run plugin doctor mere-archive-tools
```

Start a SearXNG instance with JSON results enabled and a compatible Firecrawl
Local service. Docker is required if you provision the documented local SearXNG
instance. Follow [SearXNG setup](https://plugins-docs.mere.run/plugins/searxng-search)
and [Firecrawl setup](https://plugins-docs.mere.run/plugins/firecrawl-local);
installing an adapter alone doesn't prove that either service is reachable.

For indexing, install the local reduction and embedding models:

```bash
mere.run model pull text-anonymize-privacy-filter
mere.run model pull vision-embed-qwen3-vl-2b
```

## Collect a small source set

Use a public, nonsensitive query and a fresh private output directory. Preview
without contacting the services, then run:

```bash
umask 077
mere.run plugin run mere-archive-tools -- research   --query "municipal flood map methodology"   --output-dir ./flood-research --limit 8 --dry-run
mere.run plugin run mere-archive-tools -- research   --query "municipal flood map methodology"   --output-dir ./flood-research --limit 8
```

Use `--collection-only` to save sources without building the index. The plugin
deduplicates URLs and identical page text. Search results can be skipped,
unreadable, or truncated; inspect the collection record before treating it as
complete.

## Review the collection

The output directory contains these artifacts:

| Artifact | What to check |
| --- | --- |
| `collection/sources/` | Saved Markdown for each collected page |
| `collection/sources.json` | Citation IDs, original URLs, content hashes, skipped results, and truncation |
| `collection/citations.md` | Source links using the same citation IDs |
| `index/archive.sqlite3` | Local reduced-content index, unless collection-only mode is selected |
| `run.json` | Settings, status, and collection and index paths |

For each statement in a brief, check the saved page, original publisher, date,
and scope. A stable citation ID identifies a source; it doesn't prove that the
source supports your interpretation or remains up to date.

## Search or continue

Search the index, or resume an interrupted run using its existing collection:

```bash
mere.run plugin run mere-archive-tools -- search   --database ./flood-research/index/archive.sqlite3 --query "methodology"
mere.run plugin run mere-archive-tools -- resume ./flood-research/run.json
```

Match a search result's file name to its ID in `sources.json`. To search your
own files instead, follow [Search your document archive](./shared-archive.md).

## Understand what leaves your machine

SearXNG sends the query to its configured engines. Firecrawl fetches public
URLs. Collection requires network access. Indexing and model inference run
locally, but saved snapshots can contain unredacted page text. Review the
collection before sharing it; PII reduction in the derived index doesn't redact
the saved source snapshots.
