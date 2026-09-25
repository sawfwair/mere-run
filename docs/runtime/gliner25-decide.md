# Classify text with GLiNER2.5 Decide

This guide is for developers who need local classification with labels supplied
at request time. The managed model runs a DeBERTa v3 large encoder and the
GLiNER2 classification head in Swift/MLX. It returns labels and scores without
generating text.

Install the pinned English checkpoint:

```bash
mere.run model pull text-classify-gliner25-decide
```

Save this request as `request.json`:

```json
{
  "text": "The package arrived late, and I need a refund.",
  "tasks": [
    {
      "name": "intent",
      "labels": ["order_status", "refund_request", "cancel_order"]
    },
    {
      "name": "topics",
      "labels": ["shipping", "billing", "account"],
      "multi_label": true,
      "threshold": 0.4
    }
  ]
}
```

Check the token count, then classify the text:

```bash
mere.run text classify --input request.json --preflight --pretty
mere.run text classify --input request.json --pretty
```

Each task returns selected `labels` and a `probabilities` map. Single-label
tasks use softmax and select one label. Multi-label tasks use sigmoid and select
each label at or above the task's threshold. If no label reaches the threshold,
the operation returns the highest-scoring label, matching the upstream
`classify_text` behavior. These scores depend on the supplied label set and
are not calibrated decision probabilities.

The `tasks` array preserves task order. Each task can include `prompt` and
`descriptions`, where `descriptions` maps label names to short definitions.
The encoder accepts at most 512 subword tokens for the complete schema and
text. The command rejects longer requests so labels remain intact.

The model specializes in English operational classification. It does not
answer open questions or provide explanations. For multilingual classification,
use a model trained for those languages.

## Sources and license

- [GLiNER2.5 Decide model card](https://huggingface.co/fastino/GLiNER2.5-Decide/tree/7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6)
- [GLiNER2 source](https://github.com/fastino-ai/GLiNER2/tree/55656fbfa01d3d4a77485e1a1eeeaf682990ccdf)
- [DeBERTa v2 implementation used by the checkpoint](https://github.com/huggingface/transformers/blob/v4.48.1/src/transformers/models/deberta_v2/modeling_deberta_v2.py)

The model card declares Apache 2.0. The model snapshot revision is pinned in
the managed catalog. The source revision identifies the GLiNER2 classifier and
schema formatting used for this implementation.
