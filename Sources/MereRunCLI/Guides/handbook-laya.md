# Laya decisions

Use this handbook to evaluate typed questions over local text with Laya.
The native Swift/MLX implementation supports the three checkpoints published
in `convaiinnovations/laya`:

| Model | Encoder | Default context / head tokens |
| --- | --- | --- |
| `text-decide-laya` | English ModernBERT-large, 28 layers | 512 / 192 |
| `text-decide-laya-multilingual` | mmBERT-base, 22 layers | 1024 / 256 |
| `text-decide-laya-typed-decisions` | English ModernBERT-large, 28 layers; typed-decision fine-tune | 1024 / 256 |

Select the model explicitly. The command does not detect language or route
between checkpoints. The pinned typed-decisions encoder configuration identifies
ModernBERT-large, even though parts of the upstream model card describe mmBERT.

## Example to adapt

Save this request as `request.json`:

```json
{
  "state": "I was charged twice for my subscription. Please refund the duplicate charge.",
  "questions": [
    {
      "id": "department",
      "type": "choice",
      "instructions": "Which department should handle this request?",
      "criteria": ["billing", "technical support", "sales"]
    },
    {
      "id": "urgency",
      "type": "score",
      "instructions": "Rate how urgently this request needs attention.",
      "criteria": ["routine", "soon", "immediate"]
    },
    {
      "id": "refund",
      "type": "noul",
      "instructions": "The customer requests a refund."
    }
  ]
}
```

Pull the model, inspect token budgets, and evaluate the request:

```bash
mere.run model pull text-decide-laya
mere.run text decide --model text-decide-laya --input request.json --preflight --pretty
mere.run text decide --model text-decide-laya --input request.json --output decisions.json --pretty
```

You can pipe the request to stdin instead of passing `--input`. Output is always
JSON. `--output` also writes the same result to a file. A local `--model` path
must name the checkpoint directory containing `rl_agent_config.json`.

In macOS Studio, open **Text > Decisions**, select the model and request file,
and select **Evaluate questions**. **Inspect token budgets** runs preflight.
Results remain in the Studio Library with the JSON artifact.

## Controls and variants

Requests contain a `state` string and an ordered `questions` array. To use
structured state, serialize the object or conversation into the `state` string.
Each question needs a unique `id`, a `type`, and `instructions`.

- `choice`: `criteria` is an ordered array of unique labels. An entry can be a
  string or an object such as `{"label":"billing","description":"Invoices and payments"}`.
- `score`: `criteria` lists descriptions in ascending ordinal order, starting
  at zero. The answer is the probability-weighted expected score.
- `noul`: the answer is the probability that the statement holds. Optional
  criteria use `false` and `true` labels with descriptions.

Arrays preserve option order explicitly. The request shape differs from the
upstream Python SDK's question and criterion dictionaries. The native command
uses the same prompt construction after translating these ordered inputs.

Optional request fields `max_tokens` and `head_max_tokens` override the
checkpoint budgets. The context limit is 8192; the head budget must be smaller
than the context budget. Larger budgets increase attention memory and latency.
The command supports at most 256 questions and 255 criteria per question,
within a 2 MiB request and 1 MiB state text limit.

The result includes per-question probabilities, confidence, `actProbability`,
and raw/applied calibration temperatures. It also includes token counts and
truncation details in `plan.questions`. `outputTokens` is zero because the
model scores options without generating tokens.

For `choice` and `score`, confidence is one minus normalized entropy. For
`noul`, confidence is the larger of the false/true probabilities. The action
head reports a probability; it does not execute an action or escalation.

Calibration temperatures are clamped to [0.5, 5.0], matching the pinned SDK.
The English checkpoints ship a `choice:11+` temperature below this interval.
The result marks affected answers with `temperatureClamped: true`.
Calibration on your task's held-out data remains necessary before confidence
gating. Upstream benchmark accuracy is not a guarantee for your domain.

The tokenizer replaces literal mask tokens in user text, reserves one marker
per option, and truncates the state from the end. Results report dropped state,
instruction, and option tokens. Requests that cannot retain every option and
the terminal separator fail before weight loading. This rejection is stricter
than the upstream SDK's marker-count-only overflow check.

## Sources and validation

- [Laya model repository](https://huggingface.co/convaiinnovations/laya/tree/1c5edc17a7acd8701df6fc341c0d179f1c62c982)
  pins the model files, tokenizer, and configurations.
- [Laya SDK source](https://github.com/NandhaKishorM/laya/tree/573e5b62696ba441230cd6be71d593331b5d23af)
  defines sequence construction, decision heads, and calibration.
- [ModernBERT reference](https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/modernbert/modeling_modernbert.py)
  defines the encoder implementation used for reference tests.

The model publisher and SDK identify Apache-2.0 licensing. Managed pulls
select only the requested checkpoint. Weights are downloaded separately from
the repository; inference runs in Swift/MLX without Python, ONNX, or remote code.

The native loader checks all tensor names, shapes, and dtypes and uses float32
computation. Tests include a synthetic full-graph reference, padding, a
single-option question, typed answers, calibration, invalid configurations,
and malformed weights. Real-checkpoint qualification results are recorded in
the repository's Laya qualification report. These checks establish the tested
implementation behavior; they do not establish task accuracy or production
calibration.
