# Clef structured decisions

Clef evaluates a state plus a schema of typed questions and returns probabilities
for their allowed answers. `text decide` runs the native Swift/MLX Qwen3.5
backbone and joint schema head. It accepts text, structured JSON, local images,
and video frame arrays. Select `text-decide-clef-4bit` or
`text-decide-clef-flash-4bit` explicitly; Laya remains
the default for this command and uses its own request format.

The 27B managed checkpoint is
[`mlx-community/clef-4bit`](https://huggingface.co/mlx-community/clef-4bit/tree/e0a23bd4406c15075b7473616429c46f3fd130a9),
pinned at `e0a23bd4406c15075b7473616429c46f3fd130a9`. The backbone is affine
4-bit/group-64; the vision tower and joint head retain BF16 weights. Weights
occupy about 16.3 GB before runtime memory.

The smaller 9B `text-decide-clef-flash-4bit` checkpoint pins
[`mlx-community/clef-flash-4bit`](https://huggingface.co/mlx-community/clef-flash-4bit/tree/6822f0f244ee9e19df76908ba3302f7fe40ceea6)
at `6822f0f244ee9e19df76908ba3302f7fe40ceea6` and occupies about 6.2 GB.
It uses the same native runtime, question format, and media preprocessing;
backbone and head dimensions come from the checkpoint configuration.
Both checkpoints are CLI-only. Model discovery recommends at least 16 GB of
unified memory for Flash and 32 GB for Clef; actual demand depends on the request.
The CLI also requires 16 GiB of available admission headroom for text inference,
so a busy machine can refuse a run even when it meets the model's physical-memory minimum.

```sh
mere.run model pull text-decide-clef-4bit
mere.run text decide --model text-decide-clef-4bit --input request.json --preflight --pretty
mere.run text decide --model text-decide-clef-4bit --input request.json --output decisions.json --pretty
# Use the same request with the smaller Flash checkpoint:
mere.run model pull text-decide-clef-flash-4bit
mere.run text decide --model text-decide-clef-flash-4bit --input request.json --pretty
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
The full pinned 27B checkpoint has also completed four short text/image/video
probes on Apple Silicon, with matching choices and probability differences
within `0.0031` of the checkpoint-configured reference. See the
[full-checkpoint report](../benchmarks/clef-native-qualification-2026-10-01.md)
for results, memory measurements, and limits. These short probes do not establish
exact full-model parity, general accuracy, or maximum-context capacity.

Flash also completed these four probes in an opt-in native GPU runtime test,
with exact token IDs and spans, matching choices, and rounded probability
differences within `0.0011`. Normal Flash CLI inference was refused by available
memory admission on the test machine. See the
[Flash checkpoint report](../benchmarks/clef-flash-native-qualification-2026-10-02.md)
for the runtime results, reproduction steps, and CLI validation gap.

Implementation: `Sources/MereRunCore/Clef` and
`Sources/MereRunQwenModel/ClefJointHead.swift`. Reference source:
[`clef_mlx.py`](https://huggingface.co/mlx-community/clef-4bit/blob/e0a23bd4406c15075b7473616429c46f3fd130a9/clef_mlx.py).


## Clef Omni

`text-decide-clef-omni` runs the native Qwen3-Omni 30B-A3B MoE thinker and
shared Clef joint head. It pins [`Cloudflare/clef-omni`](https://huggingface.co/Cloudflare/clef-omni/tree/0db1cd2607d76a7bdb2a382f659e7b313079f84b)
at `0db1cd2607d76a7bdb2a382f659e7b313079f84b`. This is the original BF16
checkpoint: about 70.5 GB of sharded weights, including unused upstream speech
output weights. The native runtime loads only thinker components and the joint
head. Discovery recommends at least 96 GB unified memory, preferably 128 GB.
There is no Python inference subprocess and no free-form generation.

```sh
mere.run model pull text-decide-clef-omni
mere.run text decide --model text-decide-clef-omni --input request.json --preflight --pretty
mere.run text decide --model text-decide-clef-omni --input request.json --pretty
```

Omni uses the same `state` and `questions` object. It accepts mixed local
`images`, `audio`, and `videos`. Audio is decoded to mono 16 kHz and Whisper
128-bin log-mel features. `videos` may contain local video paths or arrays of
frames already sampled at 2 fps. File videos hear their soundtracks only when
all videos have an audio track, matching the reference's policy. Video and
audio tokens interleave by timestamp; spatial/temporal positions and all
vision deep-stack features are preserved. Remote URLs, data URLs, and
`media_kwargs` overrides are rejected. Limits are 16 images, 16 audio clips,
four videos with at most 768 sampled frames each, and 384 seconds per audio
clip or heard soundtrack. Oversized inputs fail explicitly. Omni defaults to
64,000 context tokens; `max_state_tokens` truncates only the state.

```json
{
  "state": {"task": "Review the call and dashcam clip"},
  "audio": ["call.wav"],
  "videos": ["dashcam.mp4"],
  "questions": {
    "collision": {"type": "noul", "instructions": "Does the video show a collision?"},
    "glass": {"type": "noul", "instructions": "Is glass breaking audible?"}
  }
}
```

Tiny independently exported PyTorch fixtures cover causal MoE states,
interleaved positions, deep-stack injection, vision, chunked audio, and the
joint head's untied output-embedding prior. These are component tests with
untrained weights. Full 30B checkpoint probabilities and cross-platform GPU
qualification remain unverified; do not treat component agreement as a
published checkpoint qualification.
