# EmbeddingGemma 2 multimodal embeddings

Google's EmbeddingGemma 2 maps text, code, images, audio, and video into one
embedding space. `mere.run text embed` runs its text, vision, and audio encoders
natively in Swift/MLX; video contributes ordered vision-frame tokens. Use it for
local retrieval and similarity search across mixed media. A separate
[Core ML export](#apple-core-ml-export) runs text, image, and audio graphs on the
Neural Engine, CPU, and GPU.

When you omit `--model`, `text embed` uses `text-embed-qwen3-0.6b`, so select
EmbeddingGemma 2 explicitly.

## Install

The managed model ID is `text-embed-embeddinggemma2`, pinned to
`google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b`. The download
is about 1.5 GB. The catalog lists 8 GB of unified memory as the minimum and
16 GB as recommended.

```bash
mere.run model pull text-embed-embeddinggemma2
```

If you installed the model before media support, pull it again with `--force`
to add the official processor configuration.

To use an existing installation, pass its local checkpoint directory to
`--model`. The directory must contain the official config, the tokenizer JSON
and config, and safetensors weights as a single file or a sharded index.

The managed model uses the small-model admission class, which requires 6 GiB of
reclaimable headroom. Arbitrary local model folders keep the 16 GiB floor. This
model is available through the CLI only, not as an API embedding backend.

## Embed text

```bash
mere.run text embed "What causes the northern lights?" \
  --model text-embed-embeddinggemma2 --task query --dimensions 256
mere.run text embed "Charged particles from the sun cause the northern lights." \
  --model text-embed-embeddinggemma2 --task document --title "Auroras" --dimensions 256
mere.run text embed "find a stable sorting function" \
  --model text-embed-embeddinggemma2 --task code-retrieval
```

Output uses the OpenAI-style embedding JSON, with the EmbeddingGemma 2 model ID
and actual token usage. `--output` saves it, and `--pretty` formats it.

### Task prefixes

`--task` defaults to `raw`, which adds no task prefix. Use `query` for retrieval
queries and `document` for corpus entries. Documents use the format
`title: {title or none} | text: {content}`, and `--title` requires
`--task document`. The other supported tasks are `code-retrieval`,
`question-answering`, `fact-checking`, `classification`, `clustering`, and
`similarity`.

### Dimensions

Choose `--dimensions 128`, `256`, `512`, or `768` (default). The runtime
truncates the trained 768-dimensional vector and then L2-normalizes it. Queries
and stored documents must use the same dimensions.

### Token budget and precision

The input budget is 8,192 tokens, including prefixes, BOS, and EOS.
`--max-tokens` can lower this limit and must be at least two. An overlong
text-only input is truncated, keeping BOS and EOS. Mean pooling includes
prefixes and special tokens and excludes padding. Batches are packed by token
length and returned in input order.

Inference uses BF16 by default; the Swift API also supports FP32. FP16 is
rejected, matching Google's precision guidance.

## Embed images, audio, and video

The original safetensors download includes every tower, but the runtime loads
vision or audio tensors only when a request needs them.

```bash
mere.run text embed --image ./photo.png --model text-embed-embeddinggemma2
mere.run text embed --audio ./clip.wav --model text-embed-embeddinggemma2
mere.run text embed --video ./clip.mp4 --model text-embed-embeddinggemma2
```

Each direct text or media path produces an independent vector. Output lists
texts, images, audio, then videos, keeping the order within each group.

### Combine modalities in one vector

To produce a single vector from several inputs, write an ordered JSON document
such as `inputs.json`:

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

Each record produces one vector and keeps its input index. Content blocks keep
their order, including repeated or interleaved blocks.

- `--input-json -` reads stdin.
- Paths are local files. They resolve relative to the JSON file, or to the
  current directory for stdin.
- You can't combine JSON input with direct input flags.
- Text in a media record must not contain reserved media-token markers.

### Media processing

- **Images** use aspect-preserving bicubic resizing, RGB rescaling, and spatial
  patch pooling, with up to 280 soft tokens per image.
- **Audio** is decoded to mono 16 kHz and semicausal log-mel features. Each
  segment must be longer than 10 milliseconds and at most 30 seconds. Split
  longer recordings into explicit ordered segments.
- **Video** samples at 1 fps, and sequences longer than 32 frames are uniformly
  subsampled. Explicit `video-frames` keeps the supplied order of 1–32 frames.
  Each frame uses up to 140 soft tokens. Video doesn't insert timestamps or
  include the soundtrack; add an audio segment explicitly when it should
  contribute.

The 8,192-token limit includes task text, BOS and EOS, media markers, and
projected soft tokens. A media record that exceeds its budget fails; media
blocks are never partially truncated.

On macOS, decoding uses Apple's native media frameworks, and inference never
starts an external Python or FFmpeg process. Codec and color conversion can
differ between Apple and other video decoders, so use the same decoded frames
when you compare encoders across runtimes.

## Validation

Synthetic runtime tests cover bidirectional and sliding masks, per-layer
geometry, padding invariance, projection and pooling, and deterministic encoder
outputs.

### Text checkpoint results

Released-checkpoint checks passed on October 6, 2026 on an Apple M3 Max with
36 GiB of unified memory, using the pinned original BF16 weights. Thirty-one
vectors across 15 native cases covered all dimensions, retrieval, code,
multilingual and empty inputs, batching, and truncation. Thirty vectors across
14 cases matched the upstream Transformers 5.19.0 / PyTorch 2.14.1 FP32 CPU
encoder with a minimum cosine similarity of 0.9999293, and token counts matched
in every comparison. The checkpoint SHA-256 matched the pinned Hub LFS metadata.

Short CLI invocations, including model loading, took approximately 3.7–6.6
seconds and used 0.79–0.84 GiB peak process footprint. The 723-token window
check used 1.26 GiB; the 8,192-token native case took 5.4 seconds and used
4.79 GiB. System swap stayed at zero. These measurements use the peak footprint
from macOS `time -l`, which includes GPU allocations; resident set size alone
understates MLX memory. They're debug CLI measurements from one run, not
steady-state throughput. The 8,192-token case was a native stress check without
an upstream CPU parity comparison. Retrieval examples are smoke checks; broad
retrieval quality is outside this qualification.

### Media results

Media qualification on the same machine covered 14 native cases and a
four-record batch. Ten cases matched the upstream FP32 CPU model with a minimum
cosine similarity of 0.9999203, including all four output dimensions, image,
audio, video, ordered frames, repeated media, mixed content, and document
prefixes. Token counts matched in every comparison. Video comparisons used
identical native-decoded frames. The audio decoder preserved every source
sample, and native log-mel features matched the upstream frontend within 0.002.

Native boundary checks passed for 30-second audio, 48 kHz stereo resampling,
32-frame video sampling, and an 8,192-token record that combines text, image,
audio, and video frames. The full mixed record used approximately 5.7 GiB peak
process footprint, and system swap stayed at zero. These boundary cases have no
upstream CPU comparison. Broad media retrieval quality and the full codec matrix
are outside this qualification.

### Repeat the text checks

Install the managed model, then run the qualification script with the built CLI
and the installed checkpoint:

```bash
python3 scripts/validation/qualify-embeddinggemma2.py \
  --cli .build/debug/mere.run --model /path/to/embeddinggemma-2 \
  --out .tmp/embeddinggemma2-qualification --long-context
```

The script checks finite unit vectors at all four dimensions, query and document
and code retrieval examples, multilingual and empty inputs, batch consistency,
truncation, and, with `--long-context`, an 8,192-token input. It saves JSON
vectors, checkpoint hashes, timings, and process memory measurements. Its
retrieval examples are smoke checks, not a quality benchmark.

To compare against the upstream FP32 CPU text encoder, add `--reference` in a
Python environment with `torch` and a Transformers release that includes
`EmbeddingGemma2TextModel`. `--reuse-native` reuses matching native receipts.

### Repeat the media checks

To qualify media inputs against the upstream FP32 model, use a Python
environment with NumPy, Pillow, soundfile, PyTorch, and Transformers 5.19.0.
FFmpeg authors a small test clip; it's a qualification dependency, not a native
runtime dependency. Run from the repository root:

```bash
python3 scripts/validation/qualify-embeddinggemma2-media.py \
  --cli .build/debug/mere.run --model /path/to/installed/embeddinggemma-2 \
  --cli-model text-embed-embeddinggemma2 \
  --out .tmp/embeddinggemma2-media --reference --stress
```

`--cli-model` selects the managed installation, and `--model` must point to that
same checkpoint for hashes and reference weights. If you omit `--cli-model`, the
script uses the local folder directly, with the generic local-model memory
admission policy.

The script checks all four dimensions, audio, video, explicit frames, repeated
media, ordered mixtures, document prefixes, exact token counts, and overflow
rejection. It builds and exports native preprocessing through an opt-in Swift
test before it captures the CLI hash, so avoid concurrent Swift builds during
qualification. It compares the upstream encoder on identical decoded video
frames. Receipts include checkpoint and CLI hashes, vectors, reference versions,
timings, and process footprints. This is a deterministic functional and parity
qualification, not a media retrieval quality benchmark or a full codec matrix.

`--stress` adds native-only 30-second audio, 48 kHz stereo resampling, a
34-second video that exercises the 32-frame uniform cap, and an 8,192-token
record that combines text, an image, 30-second audio, and 32 video frames. It
also requires the Python `tokenizers` package. These boundary checks don't run
the upstream CPU reference.

### References

- [Google announcement](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/)
- [Pinned model card](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md)
- [Transformers encoder](https://github.com/huggingface/transformers/tree/main/src/transformers/models/embedding_gemma2)

## Apple Core ML export

To run the preconverted Core ML graphs on Apple Silicon with macOS 15 or later,
including macOS 26, install the separate Core ML model. Inference doesn't
download it automatically.

```bash
mere.run model pull text-embed-embeddinggemma2-coreml
mere.run text embed "What causes the northern lights?" --model text-embed-embeddinggemma2-coreml --task query --dimensions 256
mere.run text embed --image ./photo.png --audio ./clip.wav --model text-embed-embeddinggemma2-coreml
mere.run text embed --input-json ./inputs.json --model text-embed-embeddinggemma2-coreml --dimensions 512
```

This install uses
[FluidInference's Core ML export](https://huggingface.co/FluidInference/embeddinggemma-2-coreml/tree/3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4),
revision `3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4`. It downloads three
`.mlpackage` graphs, the tokenizer, a BF16 token table, and FP16 vision position
tables, about 1.5 GB in total. Core ML compiles the packages locally on first
use, and no source checkpoint conversion or external inference process is
required. You can also pass a local snapshot directory to `--model`.

The runtime uses the Core ML framework available on macOS 26. It doesn't depend
on Apple's macOS 27 Core AI framework or its `.aimodel` format. Text prefers the
CPU and Neural Engine; vision and audio use the CPU and GPU. Media graphs load
only when a request includes those modalities. The original MLX model, the
Qwen3 default, and Studio media selection are unchanged by this export.

### Compilation cache and repeated requests

Compiled packages persist in `~/Library/Caches/MereRun/CoreML/EmbeddingGemma2/`.
The first device load can spend substantial time specializing Neural Engine
functions, and later CLI processes reuse the compiled package at a stable path.
Changes to the package graph, replaced or modified weights, and operating-system
updates select a new cache entry. The cache contains generated models; you can
remove it when no embedding process is active, and the next request compiles
again. Original downloaded artifacts stay separate.

Text batches pack up to eight records into 256 total tokens per prediction,
with isolated attention and pooling. Up to eight short predictions can overlap.
Single queries use the smallest fitting function. Longer text and media records
keep their validated graph paths and output order.

For repeated Swift requests, keep a caller-owned session:

```swift
let session = try EmbeddingGemma2CoreMLSession(root: modelDirectory)
let result = try await session.embed(
    inputs: [EmbeddingGemma2Input(content: [.text("What causes auroras?")])],
    task: .query,
    dimensions: 256
)
```

Releasing the session releases its loaded models. Serving applications must own
admission and apply their own residency and eviction policy to the session. The
static `EmbeddingGemma2CoreMLRuntime.embed` convenience API creates one session
per call, and a fresh CLI process still pays model-loading overhead. Warm
inference, batch throughput, and fresh-process latency are separate
measurements. The optional 8K FP32 GPU extension doesn't inherit the short
Neural Engine graph's speed claims.

The dated [short-text performance receipt](/qualification/embeddinggemma2-coreml-performance-2026-10-13.json)
records release-build comparisons with the native BF16 MLX backend. It separates
warm requests from fresh CLI processes and first-use specialization. These local
observations cover short text; they don't establish an 8K or media throughput
advantage. The receipt includes raw timing samples and precision, cache, and
harness limitations.

### Input limits of the downloaded bundle

The export uses a 512-token budget, including task prefixes, BOS, and EOS.
`--max-tokens` lowers this budget. Text-only inputs truncate the body and keep
BOS and EOS. Mixed records keep content order and reject overflow without
truncating media blocks.

- **Images** use aspect-preserving bicubic RGB resizing, row-major normalized
  patches, explicit 3-by-3 pooling, and the published position tables. EXIF
  rotation isn't applied. One image uses at most 280 soft tokens.
- **Audio** files decode to mono 16 kHz, and the audio graph computes log-mel
  features. Each segment must be longer than 10 milliseconds and at most 10
  seconds. Longer audio is rejected; split it into explicit segments first.
- **Video** and `video-frames` require `text-embed-embeddinggemma2`.

Task names, document titles, dimensions, input indices, and token usage follow
the same CLI contract as the MLX model. Returned 128-, 256-, 512-, and
768-dimensional vectors are L2-normalized. This variant is available through the
CLI only.

### Expand to 8K context and 30-second audio

To embed longer records on macOS 26, convert the original Google checkpoint into
an expanded Core ML bundle. The expanded bundle accepts 8,192 total tokens and
30 seconds per audio segment. Multiple ordered segments share the token budget,
and task prefixes, media tokens, markers, BOS, and EOS count toward it.

Install the original checkpoint and the preconverted Core ML bundle first:

```bash
mere.run model pull text-embed-embeddinggemma2
mere.run model pull text-embed-embeddinggemma2-coreml
```

Run the converter from the repository checkout. Replace the two input paths with
your installed model directories, and choose a new output directory:

```bash
uv run scripts/model-conversion/convert_embeddinggemma2_coreml.py \
  --source /path/to/text-embed-embeddinggemma2 \
  --base-bundle /path/to/text-embed-embeddinggemma2-coreml \
  --output ./embeddinggemma2-coreml-8k
mere.run text embed "A longer document" --model ./embeddinggemma2-coreml-8k --max-tokens 8192
mere.run text embed --audio ./recording-30s.wav --model ./embeddinggemma2-coreml-8k
mere.run text embed --input-json ./ordered-records.json --model ./embeddinggemma2-coreml-8k
```

The converter pins its Python dependencies and verifies the original
checkpoint, configuration, and base bundle hashes before it exports. It doesn't
modify either input directory. To resume a previous conversion, pass `--resume`;
the converter verifies the retained base files before it reuses staged graphs.
`conversion.json` records source revisions, tool versions, graph source hashes,
and artifact hashes, and the output includes the source license and notices.
Python runs only during conversion; inference uses the native Swift runtime.

Longer text functions cover 1,024, 2,048, 4,096, and 8,192 positions. They keep
absolute RoPE positions and bidirectional local attention through distance 512,
alongside the four global attention layers. They use fused Core ML attention
and float32 computation on the CPU and GPU. The 30-second audio encoder also
uses float32, with log-mel features computed by native Accelerate. These choices
increase bundle size and memory use. Short inputs keep the downloaded functions
and their compute preferences.

The runtime reads limits from the bundle configuration, and `--max-tokens` can
lower them. Text truncation keeps BOS and EOS; mixed records reject overflow.
Audio longer than 30 seconds is rejected instead of truncated. Video still
requires the original MLX model.

The managed Core ML download keeps its published 512-token and 10-second
limits. To select the expanded bundle, pass the converted directory to
`--model`.

To repeat the expanded checkpoint qualification, set the bundle, original
checkpoint, and output receipt paths, and run the opt-in test:

```bash
MERERUN_EMBEDDINGGEMMA2_COREML_EXPANDED=/path/to/embeddinggemma2-coreml-8k \
MERERUN_EMBEDDINGGEMMA2_SOURCE=/path/to/text-embed-embeddinggemma2 \
MERERUN_EMBEDDINGGEMMA2_COREML_RECEIPT=/path/to/expanded-parity.json \
MERERUN_TEST_MLX_DEVICE=gpu \
swift test --filter EmbeddingGemma2CoreMLQualificationTests
```

The test compares released-checkpoint vectors with the native MLX float32
runtime across text bucket boundaries, the full context limit, 11-, 20-, and
30-second audio, mixed text and audio, and two ordered 30-second segments. It
also checks that audio over the segment limit is rejected. To include an
additional local audio fixture, set `MERERUN_EMBEDDINGGEMMA2_AUDIO_FIXTURE`. This
bounded corpus doesn't establish broad retrieval quality or throughput.

On macOS 26, all 14 expanded checkpoint cases passed:

- Text parity through 8,192 tokens had a minimum cosine of 0.999999991 against
  native MLX float32.
- The 30-second tone, mixed audio, ordered pair, and repeated speech cases had a
  minimum cosine of 0.999999858.
- The 11- and 20-second cases use the retained short text functions and had a
  minimum cosine of 0.999929803.

The public CLI returned normalized 128-dimensional vectors for an 8,192-token
text and two ordered 30-second audio segments, reporting 9,701 tokens in total.
It rejected audio that exceeded 30 seconds by one sample. The
[expanded qualification receipt](/qualification/embeddinggemma2-coreml-expanded-2026-10-13.json)
records vectors, input construction, artifact hashes, and CLI results.

### Validation on macOS 26

The native Core ML target compiles with the installed macOS 26 SDK. The released
text, vision, and audio graphs ran against the pinned published FP32 reference
corpus with the following results:

| Modality | Cases | Minimum cosine |
| --- | ---: | ---: |
| Text | 3 | 0.999934670 |
| Image | 10 | 0.999726019 |
| Audio | 14 | 0.986651154 |

The 35-second audio reference was excluded because it exceeds this export's
10-second window. Audio shows measurable FP16 drift, especially in the synthetic
chirp, siren, and piano cases. These results establish bounded compatibility,
not broad retrieval quality or sustained throughput. The expanded conversion
provides longer context, and the original MLX backend remains available. The
[raw qualification receipt](/qualification/embeddinggemma2-coreml-2026-10-13.json)
preserves each vector, token count, cosine, and skipped input. Managed-ID CLI
checks also passed for 256-dimensional text and a 128-dimensional mixed text,
image, and audio record (312 tokens).

To repeat the graph check, download the pinned model artifact and the reference
JSON files and matching media from `EverPaper/embeddinggemma-2-coreai` revision
`305fc2a96790d9343f38038aeca211dfca56d9a5`. Build the native target, then compile
`scripts/validation/qualify-embeddinggemma2-coreml.swift` against it:

```bash
swift build --target MereRunCoreMLModel --disable-index-store -debug-info-format none
build_dir="$(swift build --show-bin-path)"
swiftc -parse-as-library -module-cache-path "$build_dir/ModuleCache" -I "$build_dir/Modules" scripts/validation/qualify-embeddinggemma2-coreml.swift "$build_dir"/MereRunCoreMLModel.build/*.o -framework CoreML -framework Accelerate -framework AVFoundation -framework ImageIO -framework CoreGraphics -o /tmp/qualify-embeddinggemma2-coreml
/tmp/qualify-embeddinggemma2-coreml /path/to/coreml-model /path/to/reference-tests
```

### Reproduce tokenizer parity

The native tokenizer matched all 3,758 published cases (608,225 token IDs) at the
pinned reference revision with zero mismatches. The
[tokenizer receipt](../benchmarks/embeddinggemma2-coreml-tokenizer-2026-10-13.json)
records input hashes. This check doesn't run the embedding graphs. Using the
`build_dir` from the preceding graph check, run:

```bash
swiftc -module-cache-path "$build_dir/ModuleCache" -I "$build_dir/Modules" scripts/validation/verify-embeddinggemma2-coreml-tokenizer.swift "$build_dir"/MereRunCoreMLModel.build/*.o -framework CoreML -framework Accelerate -framework AVFoundation -framework ImageIO -framework CoreGraphics -o /tmp/verify-embeddinggemma2-coreml-tokenizer
/tmp/verify-embeddinggemma2-coreml-tokenizer /path/to/coreml-model/tokenizer.json /path/to/reference-tests/tokenizer-cases.json
```

For other embedding models, see [PPLX Embed v2](./pplx-embed-v2.md) and the
[text runtime](./text.md).
