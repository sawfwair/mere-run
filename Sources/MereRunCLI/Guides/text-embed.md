# Text Embed

## Purpose

Generate JSON embeddings for semantic search, clustering, retrieval, or similarity experiments.

## Required Models

Default managed id: `text-embed-qwen3-0.6b`.

## Install And Check

```bash
mere.run model pull text-embed-qwen3-0.6b
mere.run text embed --help
```

## Parameters

- positional text arguments: one or more strings to embed.
- `--model`, `-m`: managed id or local model path.
- `--max-tokens`: clamp input length.
- `--output`, `-o`: JSON output path.
- `--pretty`: pretty-print JSON.

## Usage Patterns

- Embed queries and documents with the same preprocessing.
- Keep chunk boundaries meaningful: title plus paragraph is usually better than arbitrary fixed text.
- For retrieval, store the original text and metadata alongside each vector.
- Use `--max-tokens` to enforce consistent latency and avoid accidentally embedding whole files.

## Examples

```bash
mere.run text embed "local inference on Apple Silicon" "Swift package layout" --pretty
```

```bash
mere.run text embed \
  "query: how do I pull a model?" \
  "document: use mere.run model pull <id>" \
  --output ./embeddings.json
```

## Iteration Tips

- Test with a tiny pair set before batch embedding a corpus.
- Add task-specific wording like `query:` and `document:` when it matches your downstream scoring style.
- Rebuild the index when chunking or normalization changes.

## Troubleshooting

- Empty input: pass one or more text arguments.
- Slow batch: embed fewer, larger chunks or run in smaller batches.
- Poor retrieval: inspect chunk text first; embeddings cannot recover missing context.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/TextEmbedCommand.swift
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
