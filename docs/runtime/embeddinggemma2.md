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

## Apple Core ML export

To run the preconverted Core ML graphs on Apple Silicon macOS 15 or later,
including macOS 26, install the separate Core ML model:

```bash
mere.run model pull text-embed-embeddinggemma2-coreml
mere.run text embed "What causes the northern lights?" --model text-embed-embeddinggemma2-coreml --task query --dimensions 256
mere.run text embed --image ./photo.png --audio ./clip.wav --model text-embed-embeddinggemma2-coreml
mere.run text embed --input-json ./inputs.json --model text-embed-embeddinggemma2-coreml --dimensions 512
```

This install uses [FluidInference's Core ML export](https://huggingface.co/FluidInference/embeddinggemma-2-coreml/tree/3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4),
revision `3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4`. It downloads three
`.mlpackage` graphs, the tokenizer, a BF16 token table, and FP16 vision
position tables. Core ML compiles the packages locally on first use. No source
checkpoint conversion or external inference process is required.
You can pass a local snapshot directory to `--model`.

The runtime uses Core ML available on macOS 26. It does not depend on Apple's
macOS 27 Core AI framework or its `.aimodel` format. The original MLX model ID,
Qwen3 default, and Studio media selection retain their existing behavior.
Inference requires an explicit install; it does not download this model
automatically. Text prefers CPU and Neural Engine. Vision and audio use CPU
and GPU. Media graphs load only when a request includes those modalities.

### Input limits of the downloaded bundle

The export uses a **512-token** budget, including task prefixes and BOS/EOS.
`--max-tokens` lowers this budget. Text-only inputs truncate the body while
preserving BOS/EOS. Mixed records preserve content order and reject overflow
without truncating media blocks.

Images use aspect-preserving bicubic RGB resizing, row-major normalized patches,
explicit 3-by-3 pooling, and the published position tables. EXIF rotation is not
applied. One image uses at most 280 soft tokens.

Audio files decode to mono 16 kHz; the audio graph computes log-mel features.
Each segment must be longer than 10 milliseconds and at most **10 seconds**.
Longer audio is rejected; split it into explicit segments before embedding.
Video and `video-frames` require `text-embed-embeddinggemma2`.

Task names, document titles, dimensions, input indices, and token usage follow
the existing CLI contract. Returned 128/256/512/768-dimensional vectors are
L2-normalized. This variant is CLI-only.

### Expand to 8K context and 30-second audio

To embed longer records on macOS 26, convert the original Google checkpoint
into an expanded Core ML bundle. The expanded bundle accepts **8,192 total
tokens** and **30 seconds per audio segment**. Multiple ordered segments share
the token budget. Task prefixes, media tokens, markers, and BOS/EOS count
toward that budget.

Install the original checkpoint and the preconverted Core ML bundle first:

```bash
mere.run model pull text-embed-embeddinggemma2
mere.run model pull text-embed-embeddinggemma2-coreml
```

Run the converter from the repository checkout. Replace the two input paths
with your installed model directories and choose a new output directory:

```bash
uv run scripts/model-conversion/convert_embeddinggemma2_coreml.py \
  --source /path/to/text-embed-embeddinggemma2 \
  --base-bundle /path/to/text-embed-embeddinggemma2-coreml \
  --output ./embeddinggemma2-coreml-8k
mere.run text embed "A longer document" --model ./embeddinggemma2-coreml-8k --max-tokens 8192
mere.run text embed --audio ./recording-30s.wav --model ./embeddinggemma2-coreml-8k
mere.run text embed --input-json ./ordered-records.json --model ./embeddinggemma2-coreml-8k
```

The converter pins Python dependencies and verifies the original checkpoint,
configuration, and base bundle hashes before exporting. It preserves both input
directories. To resume a previous conversion, pass `--resume`; it verifies the
retained base files before reusing staged graphs. `conversion.json` records
source revisions, tool versions, graph source hashes, and artifact hashes.
The output includes the source license and notices. Python runs only during
conversion; inference uses the native Swift runtime.

Longer text functions cover 1,024, 2,048, 4,096, and 8,192 positions. They
preserve absolute RoPE positions and bidirectional local attention through
distance 512, alongside the four global attention layers. They use fused
Core ML attention and float32 computation on CPU/GPU. The 30-second audio
encoder also uses float32, with log-mel features computed by native Accelerate.
These choices increase bundle size and memory use. Short inputs retain the
downloaded functions and their existing compute preferences.

The runtime reads limits from the bundle configuration. `--max-tokens` can
lower the limit. Text truncation preserves BOS/EOS; mixed records reject
overflow. Audio longer than 30 seconds is rejected instead of truncated.
Video support remains available through the original MLX model.

The managed Core ML download retains its published 512-token/10-second limits.
Use the converted directory with `--model` to select the expanded bundle.

To repeat the expanded checkpoint qualification, set the bundle, original
checkpoint, and output receipt paths and run the opt-in test:

```bash
MERERUN_EMBEDDINGGEMMA2_COREML_EXPANDED=/path/to/embeddinggemma2-coreml-8k \
MERERUN_EMBEDDINGGEMMA2_SOURCE=/path/to/text-embed-embeddinggemma2 \
MERERUN_EMBEDDINGGEMMA2_COREML_RECEIPT=/path/to/expanded-parity.json \
MERERUN_TEST_MLX_DEVICE=gpu \
swift test --filter EmbeddingGemma2CoreMLQualificationTests
```

The test compares released-checkpoint vectors against the native MLX float32
runtime across text bucket boundaries, the full context limit, 11/20/30-second
audio, mixed text/audio, and two ordered 30-second segments. It rejects audio
over the segment limit. Set `MERERUN_EMBEDDINGGEMMA2_AUDIO_FIXTURE` to include an
additional local audio fixture. This bounded corpus does not establish broad
retrieval quality or throughput.

On macOS 26, all 14 expanded checkpoint cases passed. Text parity through
8,192 tokens had minimum cosine **0.999999991** against native MLX float32.
The 30-second tone, mixed audio, ordered pair, and repeated speech cases had
minimum cosine **0.999999858**. The 11/20-second cases use the retained short
text functions and had minimum cosine **0.999929803**. The public CLI returned
normalized 128-dimensional vectors for an 8,192-token text and two ordered
30-second audio segments, reporting 9,701 tokens total. It rejected audio
exceeding 30 seconds by one sample. The
[expanded qualification receipt](../public/qualification/embeddinggemma2-coreml-expanded-2026-10-13.json)
records vectors, input construction, artifact hashes, and CLI results.

### Validation on macOS 26

The native Core ML target compiles with the installed macOS 26 SDK. The released
text, vision, and audio graphs executed against the pinned published FP32
reference corpus:

| Modality | Cases | Minimum cosine |
| --- | ---: | ---: |
| Text | 3 | 0.999934670 |
| Image | 10 | 0.999726019 |
| Audio | 14 | 0.986651154 |

The 35-second audio reference was excluded because it exceeds this export's
10-second window. Audio has measurable FP16 drift, especially the synthetic
chirp/siren and piano cases. These results establish bounded compatibility,
not broad retrieval quality or sustained throughput. The expanded conversion
provides longer context; the original MLX backend remains available.
The [raw qualification receipt](../public/qualification/embeddinggemma2-coreml-2026-10-13.json)
preserves each vector, token count, cosine, and skipped input. Managed-ID CLI
checks also passed for 256-dimensional text and a 128-dimensional mixed
text/image/audio record (312 tokens).

To repeat the graph check, download the pinned model artifact and the reference
JSON files and matching media from EverPaper/embeddinggemma-2-coreai revision
`305fc2a96790d9343f38038aeca211dfca56d9a5`. Build the native target and
compile `scripts/validation/qualify-embeddinggemma2-coreml.swift` against it:

```bash
swift build --target MereRunCoreMLModel --disable-index-store -debug-info-format none
build_dir="$(swift build --show-bin-path)"
swiftc -parse-as-library -module-cache-path "$build_dir/ModuleCache" -I "$build_dir/Modules" scripts/validation/qualify-embeddinggemma2-coreml.swift "$build_dir"/MereRunCoreMLModel.build/*.o -framework CoreML -framework Accelerate -framework AVFoundation -framework ImageIO -framework CoreGraphics -o /tmp/qualify-embeddinggemma2-coreml
/tmp/qualify-embeddinggemma2-coreml /path/to/coreml-model /path/to/reference-tests
```

### Reproduce tokenizer parity

The native tokenizer matched all 3,758 published cases (608,225 token IDs) at
the pinned reference revision with zero mismatches.
The [tokenizer receipt](../benchmarks/embeddinggemma2-coreml-tokenizer-2026-10-13.json)
records input hashes. This check does not execute the embedding graphs.

```bash
swiftc -module-cache-path "$build_dir/ModuleCache" -I "$build_dir/Modules" scripts/validation/verify-embeddinggemma2-coreml-tokenizer.swift "$build_dir"/MereRunCoreMLModel.build/*.o -framework CoreML -framework Accelerate -framework AVFoundation -framework ImageIO -framework CoreGraphics -o /tmp/verify-embeddinggemma2-coreml-tokenizer
/tmp/verify-embeddinggemma2-coreml-tokenizer /path/to/coreml-model/tokenizer.json /path/to/reference-tests/tokenizer-cases.json
```
