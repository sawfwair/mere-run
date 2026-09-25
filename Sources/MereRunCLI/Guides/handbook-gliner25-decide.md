# GLiNER2.5 Decide

GLiNER2.5 Decide classifies English text against labels you supply with each
request. The managed checkpoint uses a DeBERTa v3 large encoder and a
classification head. Pull it as `text-classify-gliner25-decide`. It does not
generate text.

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
open **Text > Classify** and attach the JSON request.

## Controls and variants

Each task needs a unique `name` and an ordered array of unique `labels`.
Optional `prompt` adds instructions after the task name. Optional
`descriptions` maps labels to short definitions. A single-label task uses
softmax and selects the highest-scoring label. Set `multi_label: true` to use
sigmoid scores and select every label meeting `threshold` (default `0.5`).
If none meets the threshold, the result includes the highest-scoring label.

The complete schema and text must fit within 512 subword tokens. Longer
requests fail validation so every requested label remains in the model input.
Scores change with the supplied label set and are not calibrated decision
probabilities. The checkpoint is intended for English classification and does
not generate text or explanations.

## Sources and validation

- [Pinned model snapshot](https://huggingface.co/fastino/GLiNER2.5-Decide/tree/7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6) (Apache 2.0)
- [GLiNER2 classifier source](https://github.com/fastino-ai/GLiNER2/tree/55656fbfa01d3d4a77485e1a1eeeaf682990ccdf)
- [Runtime guide](https://mere.run/runtime/gliner25-decide)

The pinned checkpoint test compares schema token IDs and probabilities with
the Python reference in evaluation mode.
