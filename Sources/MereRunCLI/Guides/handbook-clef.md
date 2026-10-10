# Clef structured decisions (Cloudflare)

`text-decide-clef-4bit` and the smaller 9B `text-decide-clef-flash-4bit` evaluate a complete state and schema using the native
Swift/MLX Qwen3.5 backbone and joint schema head. State can be text or JSON;
local images and video frame arrays are optional. The managed checkpoint pins
`mlx-community/clef-4bit` at `e0a23bd4406c15075b7473616429c46f3fd130a9`.
Flash pins `mlx-community/clef-flash-4bit` at
`6822f0f244ee9e19df76908ba3302f7fe40ceea6` and uses the same request format.

## Example to adapt

Save this as `clef-request.json`:

```json
{
  "state": "Checkout errors are blocking orders.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should respond?",
      "criteria": {"billing": "Invoice questions", "technical": "Bugs or outages"}
    },
    "urgency": {"type": "score", "criteria": ["Later", "This week", "Today"]},
    "outage": {"type": "noul", "instructions": "Is the service down?"}
  }
}
```

```bash
mere.run model pull text-decide-clef-4bit
mere.run text decide --model text-decide-clef-4bit --input clef-request.json --preflight --pretty
mere.run text decide --model text-decide-clef-4bit --input clef-request.json --output decisions.json --pretty
mere.run model pull text-decide-clef-flash-4bit
mere.run text decide --model text-decide-clef-flash-4bit --input clef-request.json --pretty
```

The result reports each field's choice, expected ordinal score, or probability
of truth in the upstream SystemOne format. Probabilities round to four decimal
places; no text tokens are generated. Clef occupies about 16.3 GB and Flash
about 6.2 GB before runtime memory. Pull the selected checkpoint explicitly
before inference.

## Controls and variants

- `--input -` or a pipe reads stdin. `--output` saves the JSON result.
- `--preflight` inspects token counts, truncation, option IDs, and token spans
  without loading neural weights. Media is decoded and resized during preflight.
- `max_tokens` bounds the full sequence (default and maximum 16,384).
  `max_state_tokens` limits state independently. Only state is truncated.
- Questions form an object in source order. Choice criteria form an object;
  score criteria form an ordered array; noul optionally describes true/false.
- Add `images` as local paths, or `videos` as arrays of local frame-path arrays
  representing a 24-fps source. Video sampling uses the reference's 2-fps
  defaults and paired temporal patches. Mixed image/video records and
  `media_kwargs` overrides are rejected.

Laya uses a different request format and remains the command's default.
Both Clef checkpoints are available through the CLI; they have no local API route.
Text inference also requires 16 GiB of available admission headroom. Meeting a
checkpoint's physical-memory minimum alone does not guarantee admission.

## Sources and validation

- [Clef checkpoint and model card](https://huggingface.co/mlx-community/clef-4bit/tree/e0a23bd4406c15075b7473616429c46f3fd130a9)
- [Pinned reference loader](https://huggingface.co/mlx-community/clef-4bit/blob/e0a23bd4406c15075b7473616429c46f3fd130a9/clef_mlx.py)
- [Clef Flash checkpoint](https://huggingface.co/mlx-community/clef-flash-4bit/tree/6822f0f244ee9e19df76908ba3302f7fe40ceea6)
- [Cloudflare Clef](https://huggingface.co/Cloudflare/clef)

The complete tiny FP32 head, schema encoding, and real-tokenizer preflight
match the pinned reference fixtures. Independent image/video fixtures also
verify Pillow bicubic pixels, FP32 normalization, odd-frame resize geometry,
and BF16 learned-position interpolation. Four 27B full-checkpoint text/image/video
probes completed with matching choices and probability differences within
0.0031 of the checkpoint-configured reference. These short probes do not
establish exact full-model parity, general accuracy, or capacity
at the maximum context length. See docs/benchmarks/clef-native-qualification-2026-10-01.md.

Flash completed the same four probes through the opt-in native GPU runtime test,
with exact token IDs and spans, matching choices, and rounded probability
differences within 0.0011. Normal Flash CLI inference was refused by available
memory admission on the test machine. See
docs/benchmarks/clef-flash-native-qualification-2026-10-02.md for the runtime
results and the remaining CLI validation gap.


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
