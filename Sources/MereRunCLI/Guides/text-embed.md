# Text Embed

For native EmbeddingGemma 2 text, code, image, audio, and video vectors, select
`--model text-embed-embeddinggemma2`. Use `--task query` for retrieval queries,
`--task document` for corpus text, or `--task code-retrieval` for code queries.
`--dimensions` accepts 128, 256, 512, or 768 and normalizes after truncation.
Read `mere.run guide --model text-embed-embeddinggemma2` for full controls and
ordered mixed-input examples and media limits.

For PPLX v2, select `text-embed-pplx-v2-late-0.6b` or
`text-embed-pplx-v2-late-9b` for normalized token vectors scored with sum-MaxSim.
The late models accept text queries and text or image documents.
`text-embed-pplx-v2-context-9b-preview` accepts document chunks via
`--chunks-json` and returns int8-valued chunk vectors. Read
`mere.run guide --model text-embed-pplx-v2-late-0.6b` for the distinct
multi-vector output schema and contextual controls.

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

- positional text arguments: independent strings to embed.
- `--image`: independent local images for EmbeddingGemma 2 or PPLX late documents.
- `--audio`, `--video`: independent local media inputs for EmbeddingGemma 2.
- `--input-json`: ordered mixed records; exclusive with direct inputs. Relative paths resolve beside the JSON file.
- `--chunks-json`: PPLX contextual documents shaped as `{"documents":[["chunk", "chunk"]]}`; exclusive with direct inputs.
- `--normalize`: normalize PPLX contextual vectors after dimension truncation.
- `--model`, `-m`: managed id or local model path.
- `--max-tokens`: lower the input limit; contextual inputs exceeding it are rejected.
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
