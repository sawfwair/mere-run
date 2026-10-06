# EmbeddingGemma 2 multimodal embeddings

`text embed` runs Google's EmbeddingGemma 2 text, vision, and audio encoders
natively in Swift/MLX. Video contributes ordered vision-frame tokens.
The managed model ID is `text-embed-embeddinggemma2`, pinned to
`google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b`.
Qwen3 remains the default when `--model` is omitted.

```bash
mere.run model pull text-embed-embeddinggemma2
mere.run text embed "What causes the northern lights?" \
  --model text-embed-embeddinggemma2 --task query --dimensions 256
mere.run text embed "Charged particles from the sun cause the northern lights." \
  --model text-embed-embeddinggemma2 --task document --title "Auroras" --dimensions 256
mere.run text embed "find a stable sorting function" \
  --model text-embed-embeddinggemma2 --task code-retrieval
```

Pass a local checkpoint directory to `--model` to use an existing installation.
It must contain the official config, tokenizer JSON/config, and safetensors
weights (single file or sharded index). Output uses the existing OpenAI-style
embedding JSON, with the EmbeddingGemma 2 model ID and actual token usage.
`--output` saves it; `--pretty` formats it.

## Task prefixes and dimensions

`--task` defaults to `raw`, which adds no task prefix. Use `query` for retrieval
queries and `document` for corpus entries. Documents use
`title: {title or none} | text: {content}`. Other supported tasks are
`code-retrieval`, `question-answering`, `fact-checking`, `classification`,
`clustering`, and `similarity`. `--title` requires `--task document`.

Choose `--dimensions 128`, `256`, `512`, or `768` (default). The runtime truncates
the trained 768-dimensional vector and then L2-normalizes it. Queries and stored
documents must use the same dimensions.

The input budget is 8,192 tokens including prefixes, BOS, and EOS. `--max-tokens`
can lower this limit and must be at least two. Overlong text-only input bodies are truncated
while preserving BOS and EOS. Mean pooling includes prefixes and special tokens
and excludes padding. Batches are packed by token length and restored to input
order. Inference uses BF16 by default; the Swift API also supports FP32. FP16 is
rejected, matching Google's precision guidance.

## Scope and validation

Text, code, images, audio, and video share the trained embedding space. The
original safetensors download includes every tower; the runtime loads vision or
audio tensors only when the request needs them. This model is CLI-only and is
not advertised as an API embedding backend.

## Image, audio, and video inputs

Refresh older installations to add the official processor configuration:

```bash
mere.run model pull text-embed-embeddinggemma2 --force
mere.run text embed --image ./photo.png --model text-embed-embeddinggemma2
mere.run text embed --audio ./clip.wav --model text-embed-embeddinggemma2
mere.run text embed --video ./clip.mp4 --model text-embed-embeddinggemma2
```

Each direct text or media path produces an independent vector. Direct output
order is texts, images, audio, then videos, preserving order within each group.
For a single vector combining modalities, use an ordered JSON document:

```json
{
  "inputs": [
    {
      "content": [
        {"type": "text", "text": "A bird in the garden. "},
        {"type": "image", "path": "bird.png"},
        {"type": "audio", "path": "birdsong.wav"}
      ]
    },
    {"content": [{"type": "video", "path": "garden.mp4"}]},
    {"content": [{"type": "video-frames", "frames": ["first.png", "second.png"]}]}
  ]
}
```

```bash
mere.run text embed --input-json ./inputs.json \
  --model text-embed-embeddinggemma2 --dimensions 256 --output ./vectors.json
```

`--input-json -` reads stdin. Paths are local files and resolve relative to the
JSON file, or the current directory for stdin. JSON cannot be combined with
direct input flags. Repeated/interleaved blocks preserve content order; each
record produces one vector and retains its input index. Text content in a media
record must not contain reserved media-token markers.

Images use aspect-preserving bicubic resizing, RGB rescaling, and spatial patch
pooling, with up to 280 soft tokens per image. Audio is decoded to mono 16 kHz
and semicausal log-mel features; each segment must be longer than 10 milliseconds
and at most 30 seconds. Split longer recordings into explicit ordered segments.
Video samples at 1 fps; sequences exceeding 32 frames are uniformly subsampled.
Explicit `video-frames` preserves the supplied 1–32 frame order. Each frame uses
up to 140 soft tokens. Video does not insert timestamps or include the soundtrack;
add an audio segment explicitly when it should contribute to the vector.

The 8,192-token limit includes task text, BOS/EOS, media markers, and projected
soft tokens. A media record that exceeds its token budget fails; media blocks
are never partially truncated. Decode uses Apple's native media frameworks on
macOS. Inference never starts an external Python or FFmpeg process on macOS.
Codec and color conversion can differ between Apple and other video decoders;
use the same decoded frames when comparing encoders across runtimes.

## Validation

Synthetic runtime tests cover bidirectional/sliding masks, per-layer geometry,
padding invariance, projection/pooling, and deterministic encoder outputs.
Released-checkpoint checks also passed on October 6, 2026 on an Apple M3 Max
with 36 GiB unified memory, using the pinned original BF16 weights. Thirty-one
vectors across 15 native cases covered all dimensions, retrieval, code,
multilingual/empty inputs, batching, and truncation. Thirty vectors across 14
cases matched the upstream Transformers 5.19.0 / PyTorch 2.14.1 FP32 CPU encoder
with a minimum cosine similarity of 0.9999293; token counts matched in every
comparison. The checkpoint SHA-256 matched the pinned Hub LFS metadata.

Short CLI invocations, including model loading, took approximately 3.7–6.6
seconds and used 0.79–0.84 GiB peak process footprint. The 723-token window check
used 1.26 GiB; the 8,192-token native case took 5.4 seconds and used 4.79 GiB.
System swap remained zero. Measurements use macOS `time -l`'s peak footprint,
which includes GPU allocations; resident set size alone understates MLX memory.
These are debug CLI measurements from one run, not steady-state throughput.
The 8,192-token case was a native stress check without an upstream CPU parity
comparison. Retrieval examples are smoke checks; broad retrieval quality
remains outside this qualification.

Media qualification on the same machine covered 14 native cases and a
four-record batch. Ten cases matched the upstream FP32 CPU model with a minimum
cosine similarity of 0.9999203, including all four output dimensions, image,
audio, video, ordered frames, repeated media, mixed content, and document
prefixes. Token counts matched in every comparison. Video comparisons used
identical native-decoded frames. The audio decoder preserved every source
sample; native log-mel features matched the upstream frontend within 0.002.

Native boundary checks passed for 30-second audio, 48 kHz stereo resampling,
32-frame video sampling, and an 8,192-token record combining text, image, audio,
and video frames. The full mixed record used approximately 5.7 GiB peak process
footprint; system swap remained zero. These boundary cases have no upstream CPU
comparison. Broad media retrieval quality and the full codec matrix remain
outside this qualification.

To repeat the released-checkpoint checks, first install the managed model, then
run the qualification script with the built CLI and the installed checkpoint:

```bash
python3 scripts/validation/qualify-embeddinggemma2.py \
  --cli .build/debug/mere.run --model /path/to/embeddinggemma-2 \
  --out .tmp/embeddinggemma2-qualification --long-context
```

The script checks finite unit vectors at all four dimensions, query/document and
code retrieval examples, multilingual and empty inputs, batch consistency,
truncation, and optionally an 8,192-token input. It saves JSON vectors, checkpoint
hashes, timings, and process memory measurements. These retrieval examples are
smoke checks, not a quality benchmark. Use `--reference` in a Python environment
with `torch` and `transformers` to compare against the upstream FP32 CPU text
encoder; `--reuse-native` reuses matching native receipts. The reference checks
require Transformers with `EmbeddingGemma2TextModel` support.

Reference: [Google announcement](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/),
[pinned model card](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md),
and [Transformers encoder](https://github.com/huggingface/transformers/tree/main/src/transformers/models/embedding_gemma2).

To qualify media inputs against the upstream FP32 model, use a Python environment
with NumPy, Pillow, soundfile, PyTorch, and Transformers 5.19.0. FFmpeg authors a
small test clip; it is a qualification dependency, not a native runtime dependency.
Run from the repository root:

```bash
python3 scripts/validation/qualify-embeddinggemma2-media.py \
  --cli .build/debug/mere.run --model /path/to/installed/embeddinggemma-2 \
  --cli-model text-embed-embeddinggemma2 \
  --out .tmp/embeddinggemma2-media --reference --stress
```

`--cli-model` selects the managed installation; `--model` must point to that same
checkpoint for hashes and reference weights. Omitting `--cli-model` uses the local
folder directly, with the generic local-model memory admission policy. The script
checks all four dimensions, audio, video, explicit frames, repeated media,
ordered mixtures, document prefixes, exact token counts, and overflow rejection.
It builds and exports native preprocessing through an opt-in Swift test before
capturing the CLI hash. Avoid concurrent Swift builds during qualification.
It compares the
upstream encoder on identical decoded video frames. Receipts include checkpoint
and CLI hashes, vectors, reference versions, timings, and process footprints.
This is a deterministic functional/parity qualification, not a media retrieval
quality benchmark or a full codec matrix. `--stress` adds native-only 30-second
audio, 48 kHz stereo resampling, and a 34-second video exercising the 32-frame
uniform cap, plus an 8,192-token record combining text, an image, 30-second audio,
and 32 video frames. `--stress` additionally requires the Python `tokenizers`
package. These boundary checks do not run the upstream CPU reference.
The managed model uses the existing small-model admission class with 6 GiB
reclaimable headroom; arbitrary local model folders retain the 16 GiB floor.
