# Clef structured decisions (Cloudflare)

`text-decide-clef-4bit` evaluates a complete state and schema using the native
Swift/MLX Qwen3.5 backbone and joint schema head. State can be text or JSON;
local images and video frame arrays are optional. The managed checkpoint pins
`mlx-community/clef-4bit` at `e0a23bd4406c15075b7473616429c46f3fd130a9`.

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
```

The result reports each field's choice, expected ordinal score, or probability
of truth in the upstream SystemOne format. Probabilities round to four decimal
places; no text tokens are generated. The checkpoint occupies about 16.3 GB
before runtime memory. Pull it explicitly before inference.

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
This Clef checkpoint is available through the CLI; it has no local API route.

## Sources and validation

- [Clef checkpoint and model card](https://huggingface.co/mlx-community/clef-4bit/tree/e0a23bd4406c15075b7473616429c46f3fd130a9)
- [Pinned reference loader](https://huggingface.co/mlx-community/clef-4bit/blob/e0a23bd4406c15075b7473616429c46f3fd130a9/clef_mlx.py)
- [Cloudflare Clef](https://huggingface.co/Cloudflare/clef)

The complete tiny FP32 head, schema encoding, and real-tokenizer preflight
match the pinned reference fixtures. Independent image/video fixtures also
verify Pillow bicubic pixels, FP32 normalization, odd-frame resize geometry,
and BF16 learned-position interpolation. Four full-checkpoint text/image/video
probes completed with matching choices and probability differences within
0.0031 of the checkpoint-configured reference. These short probes do not
establish exact full-model parity, general accuracy, or capacity
at the maximum context length. See docs/benchmarks/clef-native-qualification-2026-10-01.md.
