# EmbeddingGemma 2 multimodal embeddings (Google)

## Purpose

Create local text, code, image, audio, and video vectors with the native EmbeddingGemma 2 encoder.
The managed model is `text-embed-embeddinggemma2`.

## Example to adapt

```bash
mere.run text embed "find a stable sorting function" \
  --model text-embed-embeddinggemma2 --task code-retrieval --dimensions 256
mere.run text embed "func stableSort() { ... }" \
  --model text-embed-embeddinggemma2 --task document --title "sort.swift" --dimensions 256
```

These examples have not been validated with model inference. Replace the sample
code with actual corpus content and evaluate retrieval on your own queries.

## Controls and variants

Use `--task query` for ordinary retrieval queries and `--task document` for
corpus entries. `--title` supplies a document title; otherwise the title is
`none`. `--task raw` adds no prefix and is the default. Classification, clustering,
similarity, question-answering, fact-checking, and code-retrieval tasks have
their own prefixes. Task text contributes to the embedding.

`--dimensions` accepts 128, 256, 512, or 768. Truncated vectors are normalized
again. Use the same dimensions for queries and documents. `--max-tokens` lowers
the 8,192-token budget, including task text, BOS, and EOS. Text-only bodies are truncated while preserving both special tokens. Media
records reject token overflow instead of truncating soft-token blocks. Use
`--image photo.png`, `--audio clip.wav`, or `--video clip.mp4` for independent
vectors. Use `--input-json inputs.json` for ordered mixed content; see the native
usage guide for its schema. Audio segments must be at most 30 seconds. Video
samples 1 fps, uniformly capped at 32 frames, and uses no timestamps or soundtrack.
Include an audio segment explicitly when the soundtrack should contribute.

## Read this guide offline

```bash
mere.run guide --model text-embed-embeddinggemma2
mere.run text embed --help
```

The guide is bundled and requires no weights or network. Download the model
before running inference offline. Studio's **Help ▸ mere.run Guide** also shows
the model handbook.

## Sources and validation

The checkpoint is pinned to `google/embeddinggemma-2` revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`. Synthetic native tests cover encoder
math, padding, pooling, and loading. On October 6, 2026, released-checkpoint text
and code checks passed on an M3 Max with 36 GiB memory: 30 vectors matched the
upstream FP32 encoder with minimum cosine 0.9999293, and a native 8,192-token
case used 4.79 GiB peak process footprint. Broad retrieval quality, sustained
throughput, and broad media retrieval quality remain unqualified. See the native usage
guide for the reproducible checks. Editorial review date: October 6, 2026.

- [Google model card](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md)
- [Native usage guide](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/embeddinggemma2.md)
