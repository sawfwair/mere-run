# GLiNER2.5 Decide

GLiNER2.5 Decide classifies English text against supplied labels and extracts
entities, relations, and structured fields. The managed checkpoint runs a
DeBERTa v3 large encoder and native classification, count, and span heads.
Pull it as `text-classify-gliner25-decide`. It does not generate text.

## Example to adapt

Save this request as `request.json`:

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

Pull the model, check the token count, and classify the text:

```bash
mere.run model pull text-classify-gliner25-decide
mere.run text classify --input request.json --preflight --pretty
mere.run text classify --input request.json --output classifications.json --pretty
```

The command accepts `-` or omitted `--input` for piped JSON. It writes a JSON
result to stdout; `--output` also saves that result to a file. In macOS Studio,
open **Text > Classify** to edit the text and label tasks directly. The page
can also import or export the same JSON request.

To extract spans and records, save this request as `extraction.json`:

```json
{
  "text": "Alice Smith joined Acme in Paris in 2024.",
  "entities": [{"name": "person"}, {"name": "organization"}],
  "relations": [{"name": "works_for"}],
  "structures": [{"name": "employment", "fields": [
    {"name": "person", "multiple": false},
    {"name": "organization", "multiple": false}
  ]}]
}
```

```bash
mere.run text extract --input extraction.json --preflight --pretty
mere.run text extract --input extraction.json --pretty
```

The result includes text, confidence, and character offsets for each span.
In Studio, open **Text > Extract** to edit the schema and inspect results.

## Controls and variants

Each task needs a unique `name` and an ordered array of unique `labels`.
Optional `prompt` adds instructions after the task name. Optional
`descriptions` maps labels to short definitions. A single-label task uses
softmax and selects the highest-scoring label. Set `multi_label: true` to use
sigmoid scores and select every label meeting `threshold` (default `0.5`).
If none meets the threshold, the result includes the highest-scoring label.

The complete schema and text must fit within 512 subword tokens for a normal
request. Pass `--long` to split a document into overlapping chunks. Pass
`--batch` to read an array of requests and return an array of results. The
flags work with both `text classify` and `text extract`.

Classification accepts at most 16 tasks and 64 labels per task. `threshold`
corresponds to `cls_threshold` in the Python API. The extraction request
accepts entity and relation names, structures with scalar or list fields,
and optional classification tasks for a joint schema.

Scores change with the supplied label set and are not calibrated decision
probabilities. The checkpoint specializes in English classification and does
not generate text or explanations. Its extraction heads are functional but
their quality has not been qualified beyond representative Python parity cases.
See the [runtime guide](https://mere.run/runtime/gliner25-decide) for advanced
schema metadata that is not exposed by this native request.

## Sources and validation

- [Pinned model snapshot](https://huggingface.co/fastino/GLiNER2.5-Decide/tree/7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6) (Apache 2.0)
- [GLiNER2 classifier source](https://github.com/fastino-ai/GLiNER2/tree/55656fbfa01d3d4a77485e1a1eeeaf682990ccdf)
- [Runtime guide](https://mere.run/runtime/gliner25-decide)

Pinned checkpoint tests compare classification scores and entity, relation,
structure, and joint-schema predictions with the Python reference.
