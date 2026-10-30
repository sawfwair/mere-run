# Laya decisions

Laya answers typed questions about a piece of text: pick one of several
choices, rate on an ordinal scale, or estimate whether a statement holds. It
scores the allowed answers instead of generating text, so results are
probabilities you can threshold. Laya is the default model for
`mere.run text decide`. Use it for routing and triage decisions over plain text. For JSON state, images, or audio, see [Clef](./clef.md) or
[D1](./d1.md).

## Choose a checkpoint

The native Swift/MLX implementation supports the three checkpoints published in
`convaiinnovations/laya`:

| Model | Encoder | Default context / head tokens | Download |
| --- | --- | --- | --- |
| `text-decide-laya` | English ModernBERT-large, 28 layers | 512 / 192 | About 846 MB |
| `text-decide-laya-multilingual` | mmBERT-base, 22 layers | 1,024 / 256 | About 678 MB |
| `text-decide-laya-typed-decisions` | English ModernBERT-large, 28 layers; typed-decision fine-tune | 1,024 / 256 | About 846 MB |

Each checkpoint needs at least 8 GB of unified memory; 16 GB is recommended.
Select the model explicitly. The command doesn't detect language or route
between checkpoints. The pinned typed-decisions encoder configuration identifies
ModernBERT-large, although parts of the upstream model card describe mmBERT.

## Decide

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

Pull the model, check token budgets, and evaluate the request:

```bash
mere.run model pull text-decide-laya
mere.run text decide --model text-decide-laya --input request.json --preflight --pretty
mere.run text decide --model text-decide-laya --input request.json --output decisions.json --pretty
```

To read the request from stdin, omit `--input` or pass `--input -`. Output is
always JSON on stdout; `--output` also writes it to a file. A local `--model`
path must name the checkpoint directory that contains `rl_agent_config.json`.

## Use the local API and Studio

The loopback API accepts the same request at `POST /v1/text/decisions`, with an
added `model` field that names an installed managed Laya model. Clef and D1 are
available only through the CLI.

In macOS Studio, open **Text > Decisions**, choose the model, enter the text, and
add choice, score, or yes-or-no questions. **Load the example** fills in the
preceding request. **Check fit** runs preflight, and **Decide** shows each answer
with its probabilities. Studio writes `request.json` and `decisions.json` into
the run's folder and can import and export request files.

## Request format

A request contains a `state` string and an ordered `questions` array. To use
structured state, serialize the object or conversation into the `state` string.
Each question needs a unique `id`, a `type`, and `instructions`:

- `choice`: `criteria` is an ordered array of unique labels. An entry can be a
  string or an object such as `{"label":"billing","description":"Invoices and payments"}`.
- `score`: `criteria` lists descriptions in ascending ordinal order, starting
  at zero. The answer is the probability-weighted expected score.
- `noul`: the answer is the probability that the statement holds. Optional
  criteria use `false` and `true` labels with descriptions.

Arrays keep option order explicit. This request shape differs from the upstream
Python SDK's question and criterion dictionaries, but the native command builds
the same prompt after translating these ordered inputs.

The optional request fields `max_tokens` and `head_max_tokens` override the
checkpoint budgets. `max_tokens` accepts 16 to 8,192, and the head budget must be
smaller than the context budget. Larger budgets increase attention memory and
latency.

## Read the result

The result includes per-question probabilities, confidence, `actProbability`,
and the raw and applied calibration temperatures. `plan.questions` reports token
counts and truncation details. `outputTokens` is zero because the model scores
options without generating tokens.

- For `choice` and `score`, confidence is one minus normalized entropy.
- For `noul`, confidence is the larger of the false and true probabilities.
- The action head reports a probability. It doesn't run an action or an
  escalation.

Calibration temperatures are clamped to [0.5, 5.0], matching the pinned SDK. The
English checkpoints ship a `choice:11+` temperature below this interval, and the
result marks affected answers with `temperatureClamped: true`. Before you gate
on confidence, calibrate on held-out data from your task. Upstream benchmark
accuracy isn't a guarantee for your domain.

## Limits

- A request can contain up to 256 questions and 255 criteria per question.
- The request body is limited to 2 MiB, and the state text to 1 MiB.
- The tokenizer replaces literal mask tokens in user text, reserves one marker
  per option, and truncates the state from the end. Results report dropped
  state, instruction, and option tokens.
- A request that can't keep every option and the terminal separator fails
  before weights load. This check is stricter than the upstream SDK's
  marker-count-only overflow check.

## Validation

The native loader checks all tensor names, shapes, and dtypes and computes in
float32. Tests include a synthetic full-graph reference, padding, a
single-option question, typed answers, calibration, invalid configurations, and
malformed weights. The [Laya qualification report](../architecture/laya-native-qualification-2026-09-22.md)
records real-checkpoint results. These checks establish the tested
implementation behavior; they don't establish task accuracy or production
calibration.

## Sources and license

- [Laya model repository](https://huggingface.co/convaiinnovations/laya/tree/1c5edc17a7acd8701df6fc341c0d179f1c62c982)
  pins the model files, tokenizer, and configurations.
- [Laya SDK source](https://github.com/NandhaKishorM/laya/tree/573e5b62696ba441230cd6be71d593331b5d23af)
  defines sequence construction, decision heads, and calibration.
- [ModernBERT reference](https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/modernbert/modeling_modernbert.py)
  defines the encoder implementation used for reference tests.

The model publisher and SDK identify Apache-2.0 licensing. A managed pull
downloads only the requested checkpoint, separately from the repository.
Inference runs in Swift/MLX without Python, ONNX, or remote code.

For other text tools, see the [text runtime](./text.md).
