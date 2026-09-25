# Text Classify

## Purpose

Classify text against caller supplied labels with native GLiNER2.5 Decide.

## Usage

Save a JSON request with `text` and an ordered `tasks` array. Each task needs
`name` and `labels`. See the model handbook for a complete example.

```bash
mere.run model pull text-classify-gliner25-decide
mere.run text classify --input request.json --preflight --pretty
mere.run text classify --input request.json --pretty
mere.run guide --model text-classify-gliner25-decide
```

Each task has `name` and `labels` fields. Optional fields include `prompt`,
`descriptions`, `multi_label`, and `threshold`. The complete schema and text
must fit within 512 subword tokens.

## Sources

- [Pinned GLiNER2.5 Decide checkpoint](https://huggingface.co/fastino/GLiNER2.5-Decide/tree/7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6)
- [GLiNER2 classification source](https://github.com/fastino-ai/GLiNER2/tree/55656fbfa01d3d4a77485e1a1eeeaf682990ccdf)
