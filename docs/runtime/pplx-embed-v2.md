# PPLX Embed v2

Perplexity's PPLX Embed v2 models produce multi-vector embeddings for
retrieval. Late-interaction models return one vector per token, which you
compare with MaxSim. The contextual model encodes all chunks of a document
together and returns one int8 vector per chunk. `mere.run text embed` runs both
natively in Swift/MLX from the original FP32 safetensors or from packed MLX
weights, with no Python process, remote code, or conversion step.

## Choose a model

All five models are explicit installs and are available through the CLI and the
Swift API. The memory column lists the catalog's minimum and recommended unified
memory.

| Model ID | Output | Input | Download | Memory (minimum / recommended) |
| --- | --- | --- | --- | --- |
| `text-embed-pplx-v2-late-0.6b` | Normalized 128-dimensional token vectors | Text queries; text or image documents | About 2.4 GB | 16 GB / 24 GB |
| `text-embed-pplx-v2-late-9b` | Normalized 128-dimensional token vectors | Text queries; text or image documents | About 33.7 GB | 64 GB / 96 GB |
| `text-embed-pplx-v2-late-9b-mixed-4bit` | Same as late 9B | Text queries; text or image documents | About 7.4 GB | 24 GB / 36 GB |
| `text-embed-pplx-v2-context-9b-preview` | Int8-valued 1,024- or 2,048-dimensional chunk vectors | Text queries and document chunks | About 33.7 GB | 64 GB / 96 GB |
| `text-embed-pplx-v2-context-9b-preview-8bit` | Same as context 9B | Text queries and document chunks | About 10.9 GB | 24 GB / 36 GB |

The two packed 9B variants from Sawfwair load directly. The late variant uses Q4
transformer projections and Q8 token embeddings. The context variant uses Q8 for
both, because the Q4 candidate showed more vector drift. Gates, convolution,
norms, vision, and final projections stay FP32. The contextual int8 output
doesn't depend on weight packing.

Admission checks available memory before inference, so active workloads can
temporarily block a run. The checkpoint's maximum context isn't a fit guarantee.

## Encode with a late model

Install the model, then encode a query, a text document, and an image document:

```bash
mere.run model pull text-embed-pplx-v2-late-0.6b
mere.run text embed "what statute governs limitations?" \
  --model text-embed-pplx-v2-late-0.6b --task query --output query.json
mere.run text embed "A document passage" \
  --model text-embed-pplx-v2-late-0.6b --task document --output document.json
mere.run text embed --image ./document.png \
  --model text-embed-pplx-v2-late-0.6b --task document --output image.json
```

The packed 9B late model takes the same options:

```bash
mere.run model pull text-embed-pplx-v2-late-9b-mixed-4bit
mere.run text embed "What drives photosynthesis?" \
  --model text-embed-pplx-v2-late-9b-mixed-4bit --task query
```

Late models keep a vector for every retained token, including the initial query
or document marker. Documents exclude punctuation token IDs through the
checkpoint's mask configuration; queries keep them.

- Queries are limited to 1,024 tokens and documents to 4,096. Text truncates
  from the right, and `--max-tokens` can lower the limit.
- Image batches run separately from text batches and must use the document
  task. Images use the checkpoint's Qwen3-VL resize settings, RGB bicubic
  resampling, vision tower, and multimodal rotary positions. An image that
  exceeds the token limit is rejected whole; resize it before encoding.

To compare late vectors, use sum-MaxSim: for each query token, find its largest
dot product with any document token, then sum those maxima. The Swift API
exposes `PPLXEmbedV2Result.maxSim(query:document:)`. The two late models share
the upstream embedding space, so you can encode queries with the small model and
search an index built with the large one.

## Encode with the contextual model

Save the chunks of each document as `chunks.json`:

```json
{"documents":[["Curiosity begins in childhood.","Scientific breakthroughs often start with a question."],["The curiosity rover explores Mars."]]}
```

Encode the chunks, then encode a query:

```bash
mere.run model pull text-embed-pplx-v2-context-9b-preview
mere.run text embed --chunks-json chunks.json \
  --model text-embed-pplx-v2-context-9b-preview --task document \
  --dimensions 1024 --normalize --output chunks-embeddings.json
mere.run text embed "What drives scientific breakthroughs?" \
  --model text-embed-pplx-v2-context-9b-preview --task query --normalize
```

To use the packed Q8 variant, pull `text-embed-pplx-v2-context-9b-preview-8bit`
and pass it to `--model` instead. `--chunks-json -` reads the document JSON from
stdin.

Chunks in each document share a forward pass. Pooling uses each chunk's
overlapping BPE token span and excludes tokens that belong only to the prefix or
separator. NFC-normalized bytes preserve chunk boundaries for decomposed
Unicode. Queries mean-pool all valid tokens, including the query marker, and
empty chunks return zero vectors. Document prefixes tokenize as ordinary BPE.
Non-special added tokens, such as the context checkpoint's chunk separator, stay
intact, matching upstream `split_special_tokens=True`. Query markers stay
explicit special IDs.

Context vectors use an FP32 projection, `round(tanh(x) * 127)`, and int8
clamping. They're unnormalized by default and appear as integer-valued JSON
numbers. Compare them with cosine similarity, or pass `--normalize` and use dot
products. The trained dimensions are 1,024 and 2,048, and truncation happens
before normalization.

The 262,144-token limit rejects oversized inputs; the runtime never truncates or
splits chunks automatically. Available memory can impose a lower practical
limit. The contextual model is a preview, so keep its embeddings separate from
future checkpoint versions.

## Output format

For PPLX models, `--task` defaults to `document`. Results use a multi-vector
JSON schema: `representation`, `dimensions`, `normalized`, and
`data:[{index,embeddings:[[...]],tokenCount}]`. The single-vector
OpenAI-compatible `/v1/embeddings` route doesn't serve these models.

## Validation

The [PPLX Embed v2 9B qualification report](../qualification/pplx-embed-v2-9b-2026-10-09.md)
records full-checkpoint paired FP32 diagnostics, paired output drift for the
packed variants, and native CLI runs.

Synthetic FP32 fixtures exported from Transformers 5.4.0 cover the hybrid
encoder, bidirectional full attention, and the vision path. Tests also cover
marker and mask semantics, chunk spans, projection and quantization, dimensions,
input rejection, model routing, and CLI parsing. These diagnostics and fixtures
don't establish broad retrieval quality, cross-model alignment, or large-context
memory behavior, which need separate qualification.

## Sources

The original checkpoints pin these immutable upstream revisions:

- Late 0.6B: `dd4e95b836a73f6f0c32e46ea127c0b86b02169e`
- Late 9B: `77e936a1b18ed2ac00b7c76fccd70dc6a1bb1c18`
- Context preview: `b667039ee8b438a6350fbc91bbcecd86f9d363ba`

The packed managed entries pin
[late mixed Q4/Q8](https://huggingface.co/Sawfwair/pplx-embed-v2-late-9b-MLX-Mixed-4bit/tree/7309bd8d28fd8e0a9a27033f3d13cc2d7a5cf219)
and [context Q8](https://huggingface.co/Sawfwair/pplx-embed-v2-context-9b-preview-MLX-8bit/tree/abf77a86a7b84aee72640c736a92f3c5b471e6db).

Upstream: [collection](https://huggingface.co/collections/perplexity-ai/pplx-embed-v2),
[late model](https://huggingface.co/perplexity-ai/pplx-embed-v2-late-0.6b), and
[context preview](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-9b-preview).

For other embedding models, see [EmbeddingGemma 2](./embeddinggemma2.md) and the
[text runtime](./text.md).
