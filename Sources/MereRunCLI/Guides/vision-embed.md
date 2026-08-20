# Vision Embed

## Purpose

Generate L2-normalized embeddings in one shared space for text, images, or
mixed text-and-image records. Use them for semantic retrieval, clustering, and
candidate ranking. A high similarity score is not identity proof.

## Required Models

Default managed id: `vision-embed-qwen3-vl-2b` (official Qwen BF16 checkpoint,
2,048 native dimensions).

## Install And Check

```bash
mere.run model pull vision-embed-qwen3-vl-2b
mere.run vision embed --help
```

## Parameters

- `--text`: one or more independent text inputs.
- `--image`: one or more independent local image inputs.
- `--input-json`: a JSON batch path, or `-` for stdin. Use this for mixed
  text-and-image records and stable IDs.
- `--instruction`: retrieval-task instruction applied to direct inputs.
- `--model`, `-m`: managed id or local compatible checkpoint path.
- `--dimensions`: truncate the 2,048-dimensional vector and renormalize it.
- `--max-tokens`, `--min-pixels`, `--max-pixels`: input budgets.
- `--output`, `-o`: optional JSON output path. JSON is always printed to stdout.
- `--pretty`: pretty-print JSON.

## Usage Patterns

Embed an image and text query as independent items:

```bash
mere.run vision embed \
  --image ./vehicle.jpg \
  --text "a white SUV" \
  --instruction "Retrieve visually similar vehicles" \
  --pretty
```

Associate candidates and mixed queries with stable IDs:

```json
{
  "inputs": [
    {"id": "query", "text": "a maroon pickup truck"},
    {"id": "candidate-1", "image": "./crop-001.jpg"},
    {"id": "candidate-2", "text": "rear view", "image": "./crop-002.jpg"}
  ]
}
```

```bash
mere.run vision embed --input-json ./retrieval-inputs.json --output ./vectors.json
```

Compute cosine similarity with a dot product because the returned vectors are
already normalized. For video search, detect and crop candidates first, embed
the crops, rank them against the query, and hand accepted candidates to a
tracker. Keep a no-match outcome and application-specific review threshold.

## Troubleshooting

- Empty input: pass `--text`, `--image`, or a non-empty `--input-json` batch.
- Image not found: JSON-relative image paths resolve beside the JSON file.
- Memory pressure: reduce `--max-pixels` or process a smaller batch.
- Weak matches: use a task-specific instruction, candidate crops rather than
  full scenes, and a reranker or human review for consequential decisions.

## Sources

- https://github.com/QwenLM/Qwen3-VL-Embedding
- https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B
