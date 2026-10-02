# Clef structured decisions

Clef evaluates a state plus a schema of typed questions and returns probabilities
for their allowed answers. `text decide` runs the native Swift/MLX Qwen3.5
backbone and joint schema head. It accepts text, structured JSON, local images,
and video frame arrays. Select `text-decide-clef-4bit` explicitly; Laya remains
the default for this command and uses its own request format.

The managed checkpoint is
[`mlx-community/clef-4bit`](https://huggingface.co/mlx-community/clef-4bit/tree/e0a23bd4406c15075b7473616429c46f3fd130a9),
pinned at `e0a23bd4406c15075b7473616429c46f3fd130a9`. The backbone is affine
4-bit/group-64; the vision tower and joint head retain BF16 weights. Weights
occupy about 16.3 GB before runtime memory. The checkpoint is CLI-only.

```sh
mere.run model pull text-decide-clef-4bit
mere.run text decide --model text-decide-clef-4bit --input request.json --preflight --pretty
mere.run text decide --model text-decide-clef-4bit --input request.json --output decisions.json --pretty
```

The model can also be selected by its repository ID or a local checkpoint
directory containing `joint_head_config.json`. Pull it before inference;
runtime execution does not download it automatically. Omit `--input` or use
`--input -` to read a piped request. Results are always JSON.

Save this request as `request.json`:

```json
{
  "state": "Our checkout started returning errors and orders are blocked.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle the message?",
      "criteria": {
        "billing": "Payments or invoices",
        "technical": "Bugs or outages"
      }
    },
    "urgency": {
      "type": "score",
      "criteria": ["Can wait", "This week", "Today"]
    },
    "outage": {
      "type": "noul",
      "instructions": "Is a service down?"
    }
  }
}
```

`state` may be any JSON value. Questions retain their source order. Each question
has a `type` and optional `instructions`; an omitted or empty instruction uses
the question ID. Choice criteria are a nonempty object of option IDs to
descriptions. Score criteria are a nonempty array in ascending ordinal order.
Noul criteria optionally describe `true` and `false`.

Answers use the upstream SystemOne format:

- `choice`: the chosen option ID, confidence, and per-option probabilities.
- `score`: the expected zero-based ordinal score, confidence, legend, and probabilities.
- `noul`: the probability that the proposition is true.

Numeric answers round to four decimal places. Usage reports input tokens and
zero output tokens. The schema head evaluates fields jointly in one backbone
pass; no text tokens are generated.

Set `max_tokens` to bound the complete request (default and maximum 16,384).
`max_state_tokens` optionally limits state independently. Truncation removes
state tokens only. Schema and media that exceed the budget cause an error.
Preflight reports retained and dropped state tokens, option IDs, and the
half-open question/option token spans without loading neural weights.

For images, add an `images` array of local file paths. For video, add a `videos`
array in which each entry is an ordered array of local frame paths, representing
a 24-fps source. The reference's default 2-fps uniform sampling and timestamped
temporal patch pairs apply. Use images or videos within one record. Limits are
16 images, four videos with up to 768 source frames each, 128 questions, 256
options per question, and 2 MiB of request JSON. Media preprocessing uses the
checkpoint settings; `media_kwargs` overrides are rejected.

The checked-in tests compare the complete tiny FP32 joint head and schema token
layout against the pinned upstream MLX loader. They also cover typed answer
decoding, state truncation, malformed requests, model discovery, and CLI parsing.
The real checkpoint tokenizer and CLI preflight have also been checked against
the reference token IDs and spans. Set `MERERUN_CLEF_TOKENIZER` to its tokenizer
directory to rerun the optional tokenizer parity test.
Independent image/video fixtures match Pillow bicubic RGB pixels, exact FP32
normalization, odd-frame resize geometry, and BF16 learned-position interpolation.
The full pinned checkpoint has also completed four short text/image/video
probes on Apple Silicon, with matching choices and probability differences
within `0.0031` of the checkpoint-configured reference. See the
[full-checkpoint report](../benchmarks/clef-native-qualification-2026-10-01.md)
for results, memory measurements, and limits. These short probes do not establish
exact full-model parity, general accuracy, or maximum-context capacity.

Implementation: `Sources/MereRunCore/Clef` and
`Sources/MereRunQwenModel/ClefJointHead.swift`. Reference source:
[`clef_mlx.py`](https://huggingface.co/mlx-community/clef-4bit/blob/e0a23bd4406c15075b7473616429c46f3fd130a9/clef_mlx.py).
