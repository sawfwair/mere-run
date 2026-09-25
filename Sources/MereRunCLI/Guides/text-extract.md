# Text Extract

## Purpose

Extract entities, relation pairs, and structured records with native
GLiNER2.5 Decide. You can include classification tasks in the same request.

## Usage

Save a JSON request with `text` and at least one of `entities`, `relations`,
`structures`, or `classifications`. See the model handbook for a complete
example.

```bash
mere.run model pull text-classify-gliner25-decide
mere.run text extract --input extraction.json --preflight --pretty
mere.run text extract --input extraction.json --pretty
```

Use `--batch` for a JSON array of requests. Use `--long` for overlapping
document chunks and document-level span offsets. Each normal request must
fit within 512 subword tokens. The checkpoint specializes in classification;
evaluate extraction quality on your own data.

## Sources

- [Pinned GLiNER2.5 Decide checkpoint](https://huggingface.co/fastino/GLiNER2.5-Decide/tree/7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6)
- [GLiNER2 source](https://github.com/fastino-ai/GLiNER2/tree/55656fbfa01d3d4a77485e1a1eeeaf682990ccdf)
