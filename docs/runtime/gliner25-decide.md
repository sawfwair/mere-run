# Classify and extract text with GLiNER2.5 Decide

This guide is for developers who need local classification and schema driven
extraction. The managed English checkpoint runs its DeBERTa encoder and
classification, count, and span heads in Swift/MLX.

Pull the pinned checkpoint:

```bash
mere.run model pull text-classify-gliner25-decide
```

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

Each task returns selected `labels` and a `probabilities` map. Single label
tasks use softmax; multi label tasks use sigmoid and the task's `threshold`.
If no multi label score reaches the threshold, the highest scoring label is
returned. Optional `prompt` and `descriptions` fields add task instructions
and label definitions.

## Extract entities, relations, and structures

Save a request as `extraction.json`:

```json
{
  "text": "Alice Smith joined Acme in Paris in 2024.",
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

The result groups entity spans, relation pairs, and structure records.
Each span includes text, confidence, and character offsets. Entity and
relation terms can include a `description`. Structure fields can include a
`description` and `multiple` (`true` by default) to limit a field to one span
or keep several. Add `classifications` with
the same task objects used by `text classify` to run a joint schema in one
encoder pass.

## Process batches and long documents

Pass `--batch` to read a JSON array of independent requests and return an
array in the same order. Pass `--long` to split each document into overlapping
word chunks and merge its results. Both flags work with `text classify` and
`text extract`:

```bash
mere.run text classify --input classifications.json --batch --long
mere.run text extract --input extractions.json --batch --long
```

Long extraction returns offsets into the original document. Overlapping
chunks can report the same span; the runtime keeps one copy. The default
chunk size is 384 words with 64 words of overlap. Each encoded chunk still
must fit the checkpoint's 512 token limit, including the schema. The runtime
shortens a chunk when the schema needs more tokens.

The loopback API accepts the same single request at
`POST /v1/text/classifications` or `POST /v1/text/extractions` with a `model`
field. Set `long: true` for long documents. For a batch, send a `requests`
array with `model` and optional `long`; the API returns an array.

In macOS Studio, open **Text > Classify** for editable label tasks or
**Text > Extract** for editable entities, relations, structures, and joint
classification tasks. Both pages support **Check fit**, structured results,
and JSON import and export.
Turn on **Process long text in overlapping chunks** for a long document.

## Check the limits

The checkpoint specializes in English classification. Its retained count
and span heads produce entity, relation, and structure predictions, but the
published model card does not qualify their accuracy. The native checkpoint
tests compare classification scores and representative extraction predictions
with the Python reference. Evaluate extraction quality on your own data.

The normal request must fit within 512 subword tokens. Classification accepts
up to 16 tasks and 64 labels per task. Scores depend on the supplied schema
and are not calibrated decision probabilities. The model does not generate
text or explanations. The separate 1B and multilingual Decide checkpoints
are outside this managed model entry.

The upstream Python library also offers schema metadata such as validators,
choice fields, span attributes, and record formation policies. This native
request implements the checkpoint's classification, entity, relation, and
structure heads with basic field controls; it does not expose those advanced
Python metadata policies.

## Sources and license

- [Pinned GLiNER2.5 Decide model snapshot](https://huggingface.co/fastino/GLiNER2.5-Decide/tree/7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6)
- [GLiNER2 source](https://github.com/fastino-ai/GLiNER2/tree/55656fbfa01d3d4a77485e1a1eeeaf682990ccdf)
- [DeBERTa implementation](https://github.com/huggingface/transformers/blob/v4.48.1/src/transformers/models/deberta_v2/modeling_deberta_v2.py)

The model card declares Apache 2.0. The managed catalog pins the model
snapshot revision.
