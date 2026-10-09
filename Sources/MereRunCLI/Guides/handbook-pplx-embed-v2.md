# PPLX Embed v2 (Perplexity)

## Purpose

Create native multi-vector retrieval embeddings from original FP32 checkpoints
or ready-to-load packed Sawfwair variants.
`text-embed-pplx-v2-late-0.6b` and `text-embed-pplx-v2-late-9b` produce normalized
128-dimensional token vectors. `text-embed-pplx-v2-context-9b-preview` produces
2048-dimensional int8-valued vectors for chunks encoded together in a document.

## Example to adapt

```bash
mere.run model pull text-embed-pplx-v2-late-0.6b
mere.run text embed "what drives scientific breakthroughs?" \
  --model text-embed-pplx-v2-late-0.6b --task query --output query.json
mere.run text embed "Curiosity often starts with a question." \
  --model text-embed-pplx-v2-late-0.6b --task document --output document.json
mere.run text embed --image document.png \
  --model text-embed-pplx-v2-late-0.6b --task document
```

Full 9B packed checkpoints have a bounded native diagnostic suite. Evaluate
retrieval on actual queries and corpus content before building an index.

For reduced weight storage, use `text-embed-pplx-v2-late-9b-mixed-4bit` or
`text-embed-pplx-v2-context-9b-preview-8bit`. Pull these directly: no user
conversion is required. Late uses Q4 projections/Q8 embeddings; context uses
Q8 projections and embeddings. Gates, norms, convolution, vision and output
heads remain FP32. Weight packing does not change contextual int8 output.
Maximum-context fit needs separate qualification.

## Controls and variants

`--task query` and `--task document` add the checkpoint's query/document markers;
the default is document. Late models keep token vectors, masking document
punctuation. Compare them with sum-MaxSim: sum each query token's largest dot
product with a document token. Query/document limits are 1024/4096 tokens;
text truncates from the right. Images require a separate document batch and
reject token overflow; resize the image first. `--max-tokens` lowers the limit.

For contextual encoding, save `chunks.json` containing:

```json
{"documents":[["Curiosity begins in childhood.","Scientific breakthroughs often start with a question."],["The curiosity rover explores Mars."]]}
```

```bash
mere.run model pull text-embed-pplx-v2-context-9b-preview
mere.run text embed --chunks-json chunks.json \
  --model text-embed-pplx-v2-context-9b-preview --dimensions 1024 --normalize
```

Chunks within each document share a forward pass. Context vectors use
`round(tanh(projection) * 127)` and are unnormalized by default. `--dimensions`
accepts 1024 or 2048, truncating before optional `--normalize`. Empty chunks
return zero vectors. Compare unnormalized vectors with cosine similarity, or
normalized vectors with dot products. Oversized contextual inputs reject
without truncation; the checkpoint limit is 262144 tokens, and available
memory can impose a lower practical limit.

JSON uses `data[].embeddings`, a matrix of token or chunk vectors, plus
`representation`, `dimensions`, `normalized`, and per-row `tokenCount`.
The OpenAI single-vector embeddings route does not serve these models.
Keep context-preview embeddings separate from future checkpoint versions.

## Read this guide offline

```bash
mere.run guide --model text-embed-pplx-v2-late-0.6b
mere.run text embed --help
```

Pull weights explicitly before inference; PPLX models do not download automatically.
The bundled guide requires no weights or network.

## Sources and validation

Managed revisions are `dd4e95b836a73f6f0c32e46ea127c0b86b02169e` (late 0.6B),
`77e936a1b18ed2ac00b7c76fccd70dc6a1bb1c18` (late 9B), and
`b667039ee8b438a6350fbc91bbcecd86f9d363ba` (context preview).
Packed revisions are `7309bd8d28fd8e0a9a27033f3d13cc2d7a5cf219` (late mixed Q4/Q8)
and `abf77a86a7b84aee72640c736a92f3c5b471e6db` (context Q8). Their bounded full-checkpoint
native suite passed for text/chunks and one synthetic late image.
Synthetic Transformers 5.4.0 fixtures establish native FP32 text, vision,
and multimodal component parity. Tests also cover checkpoint loading,
tokenization, chunk spans, projection, dimensions, and CLI routing.
Full-checkpoint retrieval quality, cross-model embedding alignment, and
large-context memory behavior remain unqualified. Editorial review: October 9, 2026.

- [Perplexity collection](https://huggingface.co/collections/perplexity-ai/pplx-embed-v2)
- [Native usage guide](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/pplx-embed-v2.md)
