# PPLX Embed v2

Native Swift/MLX inference loads the original Perplexity FP32 safetensors directly.
No Python inference process, remote code execution, or conversion is required.

| Managed model | Output | Input |
| --- | --- | --- |
| `text-embed-pplx-v2-late-0.6b` | Normalized 128-dimensional token vectors | Text queries, text or image documents |
| `text-embed-pplx-v2-late-9b` | Normalized 128-dimensional token vectors | Text queries, text or image documents |
| `text-embed-pplx-v2-context-9b-preview` | Int8-valued 2048-dimensional chunk vectors | Text queries and document chunks |
| `text-embed-pplx-v2-late-9b-mixed-4bit` | Same late token-vector contract | Text queries, text or image documents |
| `text-embed-pplx-v2-context-9b-preview-8bit` | Same contextual chunk-vector contract | Text queries and document chunks |

The two Sawfwair 9B variants contain ready-to-load packed MLX weights. Users
pull them directly; no conversion step is needed. Late uses Q4 transformer
projections and Q8 token embeddings. Context uses Q8 for both because the Q4
candidate showed more vector drift. Gates, convolution, norms, vision, and final
projections remain FP32. Contextual int8 **output** is independent of weight
packing.

```bash
mere.run model pull text-embed-pplx-v2-late-9b-mixed-4bit
mere.run text embed "What drives photosynthesis?" \
  --model text-embed-pplx-v2-late-9b-mixed-4bit --task query
mere.run model pull text-embed-pplx-v2-context-9b-preview-8bit
mere.run text embed --chunks-json chunks.json \
  --model text-embed-pplx-v2-context-9b-preview-8bit --dimensions 1024 --normalize
```

Admission checks available memory before inference; active workloads can
temporarily block a run. The checkpoint's maximum context is not a fit guarantee.
See the [qualification record](../qualification/pplx-embed-v2-9b-2026-10-09.md)
for paired output drift and the bounded validation scope.

Install explicitly before inference; these models do not download automatically:

```bash
mere.run model pull text-embed-pplx-v2-late-0.6b
mere.run text embed "what statute governs limitations?" \
  --model text-embed-pplx-v2-late-0.6b --task query --output query.json
mere.run text embed "A document passage" \
  --model text-embed-pplx-v2-late-0.6b --task document --output document.json
mere.run text embed --image ./document.png \
  --model text-embed-pplx-v2-late-0.6b --task document --output image.json
```

Late models keep a vector for every retained token, including the initial query
or document marker. Document punctuation token IDs are excluded using the
checkpoint's mask configuration; queries retain them. Query/document limits are
1024/4096 tokens. Text truncates from the right; `--max-tokens` can lower the limit.
Image batches must run separately from text batches and must use the document
task. Images use the checkpoint's Qwen3-VL resize settings, RGB bicubic resampling,
vision tower, and multimodal rotary positions. An image exceeding the token limit
is rejected intact; resize it before encoding.

Use sum-MaxSim to compare late vectors: for each query token, find its largest
dot product with any document token, then sum those maxima. The Swift API exposes
`PPLXEmbedV2Result.maxSim(query:document:)`. The two late models share the upstream
embedding space, allowing small-model queries against a large-model index.

For contextual encoding, save `chunks.json`:

```json
{"documents":[["Curiosity begins in childhood.","Scientific breakthroughs often start with a question."],["The curiosity rover explores Mars."]]}
```

```bash
mere.run model pull text-embed-pplx-v2-context-9b-preview
mere.run text embed --chunks-json chunks.json \
  --model text-embed-pplx-v2-context-9b-preview --task document \
  --dimensions 1024 --normalize --output chunks-embeddings.json
mere.run text embed "What drives scientific breakthroughs?" \
  --model text-embed-pplx-v2-context-9b-preview --task query --normalize
```

Chunks in each document share a forward pass. Pooling uses each chunk's overlapping
BPE token span, excluding prefix/separator-only tokens. NFC-normalized bytes
preserve chunk boundaries for decomposed Unicode. Queries mean-pool all valid
tokens, including the query marker. Empty chunks return zero vectors. Document
prefixes tokenize as ordinary BPE; non-special added tokens such as the real
context checkpoint's chunk separator remain intact, matching upstream
`split_special_tokens=True`. Query markers remain explicit special IDs.

Context vectors use FP32 projection, `round(tanh(x) * 127)`, and int8 clamping.
They are unnormalized by default and represented as integer-valued JSON numbers.
Compare them with cosine similarity, or request `--normalize` and use dot products.
The trained dimensions are 1024 and 2048; truncation occurs before normalization.
The 262144-token limit rejects oversized inputs without truncation or automatic
chunk splitting. Available memory can impose a lower practical limit.

`--task` defaults to document for PPLX. These results have a separate multi-vector
JSON schema: `representation`, `dimensions`, `normalized`, and
`data:[{index,embeddings:[[...]],tokenCount}]`. They are CLI/Swift API models;
the OpenAI single-vector `/v1/embeddings` route does not serve them.

The contextual model is a preview. Keep its embeddings separate from future
checkpoint versions.

## Pins and validation

The original checkpoints pin immutable upstream revisions:

- Late 0.6B: `dd4e95b836a73f6f0c32e46ea127c0b86b02169e`
- Late 9B: `77e936a1b18ed2ac00b7c76fccd70dc6a1bb1c18`
- Context preview: `b667039ee8b438a6350fbc91bbcecd86f9d363ba`

The packed managed entries pin [late mixed Q4/Q8](https://huggingface.co/Sawfwair/pplx-embed-v2-late-9b-MLX-Mixed-4bit/tree/7309bd8d28fd8e0a9a27033f3d13cc2d7a5cf219)
and [context Q8](https://huggingface.co/Sawfwair/pplx-embed-v2-context-9b-preview-MLX-8bit/tree/abf77a86a7b84aee72640c736a92f3c5b471e6db).
Full-checkpoint paired FP32 diagnostics and native CLI runs are recorded in
the [qualification report](../qualification/pplx-embed-v2-9b-2026-10-09.md).

Synthetic FP32 fixtures exported from Transformers 5.4.0 cover the hybrid encoder,
bidirectional full attention, and vision path. Tests also cover marker/mask
semantics, chunk spans, projection/quantization, dimensions, input rejection,
model routing, and CLI parsing. Broad retrieval quality, cross-model alignment, and large-context memory
behavior require separate qualification. The bounded full-checkpoint diagnostics
and synthetic fixtures do not establish those properties.

Upstream: [collection](https://huggingface.co/collections/perplexity-ai/pplx-embed-v2),
[late model](https://huggingface.co/perplexity-ai/pplx-embed-v2-late-0.6b), and
[context preview](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-9b-preview).
