# Clef structured decisions

Clef evaluates a state and a schema of typed questions in one forward pass and
returns probabilities for each question's allowed answers. The state can be
text or any JSON value, and requests can include local images and video. Use
Clef when a decision depends on structured data or media. For plain-text
decisions, [Laya](./laya.md) is smaller and is the default for `text decide`.

`text decide` runs Clef natively in Swift/MLX: a Qwen3.5 backbone, or a
Qwen3-Omni thinker for Clef Omni, plus a joint schema head. It doesn't generate
text.

## Choose a checkpoint

All three checkpoints are explicit installs and are available only through the
CLI. The memory columns are the catalog's minimum and recommended unified
memory.

| Model ID | Backbone | Inputs | Download | Memory (minimum / recommended) |
| --- | --- | --- | --- | --- |
| `text-decide-clef-flash-4bit` | 9B Qwen3.5, 4-bit | Text, JSON, images, video frames | About 6.2 GB | 16 GB / 32 GB |
| `text-decide-clef-4bit` | 27B Qwen3.5, 4-bit | Text, JSON, images, video frames | About 16.3 GB | 32 GB / 64 GB |
| `text-decide-clef-omni` | Qwen3-Omni 30B-A3B MoE, BF16 | Text, JSON, images, audio, videos | About 70.5 GB | 96 GB / 128 GB |

The CLI also requires 16 GiB of available admission headroom for text
inference, so a busy machine can refuse a run even when it meets the model's
minimum. Actual demand depends on the request.

## Decide

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

Pull a checkpoint, check the token budget, and evaluate the request:

```bash
mere.run model pull text-decide-clef-4bit
mere.run text decide --model text-decide-clef-4bit --input request.json --preflight --pretty
mere.run text decide --model text-decide-clef-4bit --input request.json --output decisions.json --pretty
```

The same request works with the smaller Flash checkpoint:

```bash
mere.run model pull text-decide-clef-flash-4bit
mere.run text decide --model text-decide-clef-flash-4bit --input request.json --pretty
```

You can also select a model by its repository ID or by a local checkpoint
directory that contains `joint_head_config.json`. Inference doesn't download
the model. To read a piped request, omit `--input` or pass `--input -`. Results
are always JSON.

## Request format

`state` can be any JSON value. Questions keep their source order. Each question
has a `type` and optional `instructions`; an omitted or empty instruction uses
the question ID.

- `choice`: criteria are a nonempty object that maps option IDs to
  descriptions. The answer is the chosen option ID, a confidence, and
  per-option probabilities.
- `score`: criteria are a nonempty array in ascending ordinal order. The answer
  is the expected zero-based ordinal score, a confidence, a legend, and
  probabilities.
- `noul`: criteria optionally describe `true` and `false`. The answer is the
  probability that the proposition is true.

Answers use the upstream SystemOne format, and numeric values round to four
decimal places. Usage reports input tokens and zero output tokens. The schema
head evaluates all fields jointly in one backbone pass.

`max_tokens` bounds the complete request; its default and maximum is 16,384 for
the 4-bit checkpoints. `max_state_tokens` optionally limits the state on its
own. Truncation removes state tokens only; schema and media that exceed the
budget cause an error. Preflight reports retained and dropped state tokens,
option IDs, and the half-open question and option token spans without loading
neural weights.

## Images and video

To add images, include an `images` array of local file paths. To add video,
include a `videos` array in which each entry is an ordered array of local frame
paths from a 24 fps source. The reference's default 2 fps uniform sampling and
timestamped temporal patch pairs apply. Use either images or videos in one
record. Media preprocessing uses the checkpoint settings, and the runtime
rejects `media_kwargs` overrides.

The 4-bit checkpoints accept up to the following limits:

- 16 images
- Four videos, with up to 768 source frames each
- 128 questions, with up to 256 options each
- 2 MiB of request JSON

## Validation

The checked-in tests compare the complete tiny FP32 joint head and schema token
layout with the pinned upstream MLX loader. They also cover typed answer
decoding, state truncation, malformed requests, model discovery, and CLI
parsing. The real checkpoint tokenizer and CLI preflight match the reference
token IDs and spans. To rerun the optional tokenizer parity test, set
`MERERUN_CLEF_TOKENIZER` to the tokenizer directory. Independent image and
video fixtures match Pillow bicubic RGB pixels, exact FP32 normalization,
odd-frame resize geometry, and BF16 learned-position interpolation.

The full pinned 27B checkpoint completed four short text, image, and video
probes on Apple Silicon. Choices matched, and probabilities were within
`0.0031` of the checkpoint-configured reference. The
[Clef checkpoint report](../benchmarks/clef-native-qualification-2026-10-01.md)
has the results, memory measurements, and limits. These short probes don't
establish exact full-model parity, general accuracy, or maximum-context
capacity.

Flash completed the same four probes in an opt-in native GPU runtime test, with
exact token IDs and spans, matching choices, and rounded probability
differences within `0.0011`. Normal Flash CLI inference was refused by
available-memory admission on the test machine. The
[Flash checkpoint report](../benchmarks/clef-flash-native-qualification-2026-10-02.md)
covers the runtime results, reproduction steps, and that CLI validation gap.

The implementation lives in `Sources/MereRunCore/Clef` and
`Sources/MereRunQwenModel/ClefJointHead.swift`. The reference source is
[`clef_mlx.py`](https://huggingface.co/mlx-community/clef-4bit/blob/e0a23bd4406c15075b7473616429c46f3fd130a9/clef_mlx.py).

## Sources

- `text-decide-clef-4bit` pins
  [`mlx-community/clef-4bit`](https://huggingface.co/mlx-community/clef-4bit/tree/e0a23bd4406c15075b7473616429c46f3fd130a9)
  at `e0a23bd4406c15075b7473616429c46f3fd130a9`. The backbone is affine
  4-bit/group-64; the vision tower and joint head keep BF16 weights.
- `text-decide-clef-flash-4bit` pins
  [`mlx-community/clef-flash-4bit`](https://huggingface.co/mlx-community/clef-flash-4bit/tree/6822f0f244ee9e19df76908ba3302f7fe40ceea6)
  at `6822f0f244ee9e19df76908ba3302f7fe40ceea6`. It uses the same runtime,
  question format, and media preprocessing, and reads backbone and head
  dimensions from the checkpoint configuration.

## Clef Omni

`text-decide-clef-omni` runs the native Qwen3-Omni 30B-A3B MoE thinker with the
shared Clef joint head. It pins
[`Cloudflare/clef-omni`](https://huggingface.co/Cloudflare/clef-omni/tree/0db1cd2607d76a7bdb2a382f659e7b313079f84b)
at `0db1cd2607d76a7bdb2a382f659e7b313079f84b`. This is the original BF16
checkpoint: about 70.5 GB of sharded weights, including upstream speech-output
weights that the runtime doesn't use. The native runtime loads only the thinker
components and the joint head, with no Python subprocess.

```bash
mere.run model pull text-decide-clef-omni
mere.run text decide --model text-decide-clef-omni --input request.json --preflight --pretty
mere.run text decide --model text-decide-clef-omni --input request.json --pretty
```

Omni uses the same `state` and `questions` object and accepts mixed local
`images`, `audio`, and `videos`:

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

- Audio is decoded to mono 16 kHz and Whisper 128-bin log-mel features.
- `videos` can contain local video paths or arrays of frames already sampled at
  2 fps. File videos contribute their soundtracks only when every video has an
  audio track, matching the reference's policy.
- Video and audio tokens interleave by timestamp. Spatial and temporal positions
  and all vision deep-stack features are preserved.
- Remote URLs, data URLs, and `media_kwargs` overrides are rejected.

Omni accepts up to 16 images, 16 audio clips, and four videos with at most 768
sampled frames each. Each audio clip or included soundtrack can be up to 384
seconds. Oversized inputs fail explicitly. Omni defaults to 64,000 context
tokens, and `max_state_tokens` truncates only the state.

Tiny independently exported PyTorch fixtures cover causal MoE states,
interleaved positions, deep-stack injection, vision, chunked audio, and the
joint head's untied output-embedding prior. These component tests use untrained
weights. Eight short real Q4 checkpoint probes also pass on a 36 GB Mac; see the
[Clef Omni Q4 qualification](../benchmarks/clef-omni-q4-qualification-2026-10-09.md).
Native execution of the original BF16 checkpoint and cross-platform GPU
qualification remain unverified.

### Convert a quantized checkpoint for 36 GB Macs

`scripts/model-conversion/convert_clef_omni_mlx.py` streams the pinned original
checkpoint on a CUDA MLX host. It packs only routed expert matrices with affine
4-bit/group-64 quantization. Attention, routers, embeddings, vision, audio, and
the joint decision head keep source BF16 precision, and speech-output weights
are omitted. The native loader accepts the converter's explicit
`quantization.scope: thinker_moe_experts` policy and rejects other policies.

```bash
python scripts/model-conversion/convert_clef_omni_mlx.py \
  --source /workspace/clef-omni-source --output /workspace/clef-omni-mlx-q4
mere.run text decide --model /path/to/clef-omni-mlx-q4 --input request.json --pretty
```

The converted bundle is 22.0 GB. A native Metal run on a 36 GB Mac passed eight
151–294-token text and media probes at 22.06 GB peak MLX allocation. These short
probes don't qualify an 8,192-token context, so set an explicit `max_tokens`
budget and close other model workloads.

Weight size alone doesn't establish memory fit: input length, media features,
loading temporaries, and memory pressure also matter. The conversion manifest
records source hashes, output hashes, logical weights, and bounded
reconstruction errors; those errors aren't decision-quality measurements.
Before you treat a converted artifact as tested on 36 GB hardware, qualify it
with real local execution and independent BF16 decision comparisons.

For other decision models, see [Laya](./laya.md), [D1](./d1.md), and the
[text runtime](./text.md).
