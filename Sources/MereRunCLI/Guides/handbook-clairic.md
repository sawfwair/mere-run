# Clairic document extraction (Cloudflare)

## Example to adapt

Save a document and its requested fields in `request.json`.

```json
{
  "state": "Northwind Supply invoiced $1080.00 on November 18, 2026.",
  "fields": {
    "vendor": {"type": "text", "instructions": "The invoice vendor"},
    "total": {"type": "number", "decimals": 2},
    "issued": {"type": "date", "source_only": true}
  }
}
```

```bash
mere.run model pull text-extract-clairic-flash-bf16
mere.run text extract --model text-extract-clairic-flash-bf16 --input request.json --preflight
mere.run text extract --model text-extract-clairic-flash-bf16 --input request.json --pretty
```

## Controls and variants

- `text-extract-clairic-flash-bf16` is the 9B model and uses about 19 GB on disk.
- `text-extract-clairic-bf16` is the 27B model and uses about 56 GB on disk.
- Request fields support text, number, boolean, enum, object, list, and
  explicitly printed dates and times. Use `images` and `documents` for local
  inputs, with up to four images or PDF pages combined.
- Check `valid` and `complete` before consuming a value. The result includes
  confidence and the model probability of the null token. Use `--preflight`
  to check the request without loading weights.

## Sources and validation

The local model uses the release's ordinary slot prompts and Swift/MLX
inference. Cloudflare's separate learned typed heads and supplementary
recovery passes are outside this implementation. Full-checkpoint parity has
yet to be validated.

- [Clairic runtime guide](https://mere.run/runtime/clairic)
- [Cloudflare release announcement](https://blog.cloudflare.com/open-clairic/)
