# Classify and extract text with GLiNER2.5 Decide

GLiNER2.5 Decide classifies English text against labels you supply and extracts
entities, relations, and structured records that match a schema. Use it when
you need fast, local, non-generative labeling or extraction. The managed
checkpoint runs its DeBERTa encoder and its classification, count, and span
heads natively in Swift/MLX.

## Install

Pull the pinned checkpoint before inference. It's about 1.9 GB, and the catalog
lists 8 GB of unified memory as the minimum and 16 GB as recommended.

```bash
mere.run model pull text-classify-gliner25-decide
```

Both `text classify` and `text extract` use this model by default. To use a
local copy, pass its checkpoint directory to `--model`.

## Classify text

Save a request as `classification.json`:

```json
{
  "text": "The package arrived late, and I need a refund.",
  "tasks": [
    {"name": "intent", "labels": ["order_status", "refund_request", "cancel_order"]},
    {"name": "topics", "labels": ["shipping", "billing", "account"],
     "multi_label": true, "threshold": 0.4}
  ]
}
```

Check the token count, then classify the text:

```bash
mere.run text classify --input classification.json --preflight --pretty
mere.run text classify --input classification.json --pretty
```

Each task returns its selected `labels` and a `probabilities` map. Single-label
tasks use softmax. Multi-label tasks use sigmoid and the task's `threshold`; if
no score reaches the threshold, the highest-scoring label is returned. Optional
`prompt` and `descriptions` fields add task instructions and label definitions.

## Extract entities, relations, and structures

Save a request as `extraction.json`:

```json
{
  "text": "Dana Example joined Northwind Labs in Paris in 2024.",
  "entities": [{"name": "person"}, {"name": "organization"}, {"name": "location"}],
  "relations": [{"name": "works_for"}, {"name": "located_in"}],
  "structures": [{
    "name": "employment",
    "fields": [
      {"name": "person", "multiple": false},
      {"name": "organization", "multiple": false},
      {"name": "location", "multiple": false}
    ]
  }],
  "threshold": 0.5
}
```

Run extraction:

```bash
mere.run text extract --input extraction.json --preflight --pretty
mere.run text extract --input extraction.json --pretty
```

The result groups entity spans, relation pairs, and structure records. Each span
includes its text, confidence, and character offsets.

- Entity and relation terms can include a `description`.
- Structure fields can include a `description` and `multiple`. The default,
  `true`, keeps several spans; `false` limits the field to one.
- To run a joint schema in one encoder pass, add `classifications` with the
  same task objects that `text classify` uses.

## Process batches and long documents

Both `text classify` and `text extract` accept two processing flags:

- `--batch` reads a JSON array of independent requests and returns an array of
  results in the same order.
- `--long` splits each document into overlapping word chunks and merges the
  results.

```bash
mere.run text classify --input classifications.json --batch --long
mere.run text extract --input extractions.json --batch --long
```

Chunks default to 384 words with 64 words of overlap. Each encoded chunk must
still fit the checkpoint's 512-token limit, including the schema, so the runtime
shortens a chunk when the schema needs more tokens. Long extraction returns
offsets into the original document. When overlapping chunks report the same
span, the runtime keeps one copy.

## Use the local API and Studio

The loopback API accepts the same single request at
`POST /v1/text/classifications` or `POST /v1/text/extractions`, with an added
`model` field. Set `long: true` for a long document. For a batch, send a
`requests` array with `model` and an optional `long`; the API returns an array.

In macOS Studio, open **Text > Classify** to edit label tasks, or
**Text > Extract** to edit entities, relations, structures, and joint
classification tasks. Both pages support **Check fit**, structured results, and
JSON import and export. For a long document, turn on
**Process long text in overlapping chunks**.

## Limits

- A request without `--long` must fit within 512 subword tokens.
- Classification accepts up to 16 tasks and 64 labels per task.
- Scores depend on the schema you supply and aren't calibrated decision
  probabilities.
- The model doesn't generate text or explanations.
- The checkpoint specializes in English classification. Its count and span heads
  produce entity, relation, and structure predictions, but the published model
  card doesn't qualify their accuracy. Evaluate extraction quality on your own
  data.
- The separate 1B and multilingual Decide checkpoints aren't part of this
  managed model.

The upstream Python library also offers schema metadata such as validators,
choice fields, span attributes, and record formation policies. This native
request implements the classification, entity, relation, and structure heads
with basic field controls; it doesn't expose those Python metadata policies.

## Validation

The native checkpoint tests compare classification scores and representative
extraction predictions with the Python reference. No separate full-checkpoint
qualification report exists for this model.

## Sources and license

- [Pinned GLiNER2.5 Decide model snapshot](https://huggingface.co/fastino/GLiNER2.5-Decide/tree/7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6)
- [GLiNER2 source](https://github.com/fastino-ai/GLiNER2/tree/55656fbfa01d3d4a77485e1a1eeeaf682990ccdf)
- [DeBERTa implementation](https://github.com/huggingface/transformers/blob/v4.48.1/src/transformers/models/deberta_v2/modeling_deberta_v2.py)

The model card declares Apache 2.0. The managed catalog pins the model snapshot
revision.

For a ticket-routing example, see
[Route tickets with GLiNER and Cua Driver](../business/direct-ticket-triage.md).
For other text tools, see the [text runtime](./text.md).
