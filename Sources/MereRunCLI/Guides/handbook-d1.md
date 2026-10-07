# D1 multimodal decisions (LiquidAI)

`text-decide-d1-3b-bf16` and `text-decide-d1-omni-600m-fp32` evaluate named
choice, score, and yes/no questions with native Swift/MLX computation. Both
accept text, JSON, and local images. Omni also accepts one local audio clip.
They score typed answers without generating text tokens.

## Example to adapt

Save this as `d1-request.json`:

```json
{
  "state": "The parcel arrived on time, but its glass was broken.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle the claim?",
      "criteria": {"shipping": "Late delivery", "damage": "Damaged goods"}
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this claim?",
      "criteria": ["low", "medium", "high"]
    },
    "arrived": {"type": "noul", "instructions": "Did the parcel arrive?"}
  }
}
```

Review the LFM Open License v1.0 before pulling the selected checkpoint:

```bash
mere.run model pull text-decide-d1-3b-bf16 --accept-model-license
mere.run text decide --model text-decide-d1-3b-bf16 --input d1-request.json --preflight --pretty
mere.run text decide --model text-decide-d1-3b-bf16 --input d1-request.json --output decisions.json --pretty
mere.run model pull text-decide-d1-omni-600m-fp32 --accept-model-license
mere.run text decide --model text-decide-d1-omni-600m-fp32 --input d1-request.json --pretty
```

The response reports the selected choice and option probabilities, an expected
ordinal score, or `P(yes)`. `usage.output_tokens` is zero. Each question runs an
independent forward; input usage counts every executed row.

## Controls and variants

- Use `--input -` for stdin and `--output` to also save the response JSON.
- `--preflight` decodes media and reports token budgets, option order, dropped
  state tokens, and omni marker positions without loading weights.
- Add `"images": ["/path/to/image.png"]` for either model, or
  `"audio": "/path/to/clip.wav"` for omni. Images and audio cannot be mixed.
  Videos, remote URLs, and processor overrides are unsupported.
- Audio is decoded to 16 kHz mono, capped at 30 seconds, and padded to 0.5 seconds.
- `max_tokens` bounds each complete sequence, including media. D1-3B rejects
  overflow beyond its 32K context. Omni truncates state on the right while
  preserving option markers, within its 16K context and released media budgets.
- An explicit local checkpoint directory works. Inference does not automatically
  download models. D1 has no local API route; Laya remains the command default.

Text inference retains the 16 GiB available-memory admission floor. Preflight
does not reserve inference memory.

## Sources and validation

- [D1 announcement](https://www.liquid.ai/blog/d1-open)
- [Pinned D1-3B checkpoint](https://huggingface.co/LiquidAI/d1-3B/tree/da1fe36a861f24690f27f622dca1d8688503d113)
- [Pinned D1 omni checkpoint](https://huggingface.co/LiquidAI/d1-omni-600M/tree/414f8d6438174f5b2133a9c21a478fc42625e308)

Independent small FP32 fixtures verify trunks, the option head, media isolation,
vision projection, audio features, and FastConformer output. Original tokenizer
comparisons and exact bilinear/bicubic pixel fixtures pass. Synthetic checkpoint
loading exercises text and images for both models and audio for omni.

Full released-checkpoint execution and BF16 numerical parity remain unqualified.
The upstream tree packing optimization is not implemented; fixture results do
not establish model accuracy or latency. See docs/runtime/d1.md for details.
