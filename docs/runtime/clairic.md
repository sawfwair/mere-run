# Clairic extraction

Clairic and Clairic Flash extract named values from text, images, and PDF pages.
The native Swift/MLX runtime reads the published Cloudflare Qwen3.5 BF16
checkpoints directly. Flash is approximately 19 GB on disk. Clairic is
approximately 56 GB. Leave additional unified memory for the runtime and cache.

```bash
mere.run model pull text-extract-clairic-flash-bf16
mere.run text extract --model text-extract-clairic-flash-bf16 --input request.json --preflight
mere.run text extract --model text-extract-clairic-flash-bf16 --input request.json --pretty
```

Use `text-extract-clairic-bf16` for the larger checkpoint. `--model` also
accepts a local Cloudflare snapshot directory. Omit `--input` to read JSON from
standard input. Use `--batch` to process a JSON array of requests.

```json
{
  "state": "Vendor: Northwind Supply. The invoice total is $1080.00.",
  "fields": {
    "vendor": {"type": "text", "instructions": "The invoice vendor"},
    "total": {"type": "number", "decimals": 2}
  },
  "options": {"return_state_text": true}
}
```

The native runtime supports `text`, `number`, `boolean`, `enum`, `object`,
and `list` fields. Lists take a scalar item type or an object item schema.
They default to eight slots and support up to 32 via `max_items`. An object
uses a `properties` map of scalar fields. Text-only requests can use
`source_only` for literal numbers and number lists. For printed date, time,
datetime, and date-list fields, use `source_only: true` to extract and normalize
an explicitly printed value. `state` can be text or a JSON value.

Add `images` and `documents` arrays for local image paths or PDF paths.
Both accept embedded base64 data URIs and objects with `content_type` and
`base64`. Images and PDF pages share a four-item limit. Each input can be
up to 4 MiB, with 8 MiB total. PDF rasterization requires macOS.

The response contains typed `answers`, input and output token counts, and a
`truncated` flag. Each field reports `valid`, `complete`, `confidence`,
`p_null`, and a `span` when a scalar has uniquely matched text evidence. Spans
use half-open Unicode scalar offsets. Inference rejects requests that exceed
the context budget. Inferred dates and times, durations, compute fields,
estimates, and forecasts require the separate typed head. Cloudflare's additional recovery and media
heuristics are outside this implementation.

- [Cloudflare release announcement](https://blog.cloudflare.com/open-clairic/)
- [Pinned Clairic Flash checkpoint](https://huggingface.co/Cloudflare/clairic-flash/tree/36ad68b999560f56b57ef16a22180f8cb83c6d77)
- [Pinned Clairic checkpoint](https://huggingface.co/Cloudflare/clairic/tree/2c161be5cc71bacd6605b99a8c07547e58cfa434)
