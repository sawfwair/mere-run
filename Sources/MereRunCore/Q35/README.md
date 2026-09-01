# Q35

Qwen 3.5/3.6/3.8 dense and hybrid MoE text and vision-language runtime.

- `Q35Config.swift`: typed text/vision configuration.
- `Q35TokenizerAndTemplate.swift`: checkpoint-native chat-template rendering,
  image-token expansion, and tokenization.
- `Q35Model.swift`: native model entry point.
- Attention and MoE files own model math only.

Keep tokenizer/tool template compatibility isolated here; model layers should
not know about CLI or managed-model concerns.

Official Hugging Face Qwen 3.5/3.8 checkpoints store zero-centered RMSNorm
offsets, while converted MLX checkpoints store direct scales. The loader detects
the checkpoint layout from embedded MTP keys or PyTorch Conv1d shapes and keeps
both conventions compatible with the native offset RMSNorm module.

The official Qwen3.8 27B shards embed a dense one-layer MTP head. The loader
reads only the shards that contain `mtp.*` tensors, maps its dense SwiGLU layout,
and also discovers a bare `model.safetensors` MTP component under `mtp/`. The
managed 4-bit lane pairs the MLX Fast reference target with a matching
4-bit/group-64 proposal head. `MERERUN_Q35_MTP_SPECULATION=1` enables greedy
speculation from short prompts. It stays opt-in because multi-token target
verification can choose a different greedy path from serial target decode.
Qwen3.6 hybrid MoE keeps the existing adaptive long-context threshold.

Ornith 1.5's official MLX quants omit the MTP tensors advertised by their
configuration. Managed Q4/Q6/Q8/BF16 pulls therefore install one shared,
revision-pinned final shard from the authoritative Ornith base checkpoint; the
runtime reads only its `mtp.*` tensors. Routed-expert gate/up fusion is prepared
one decoder layer at a time after target weights load. Each evaluated fused
stack replaces its two source arrays before the MLX cache is cleared, bounding
transient preparation memory instead of retaining a model-wide duplicate.
Verified Q4 measurements showed a short-prompt decode win, so managed Ornith
1.5 targets enable MTP from token zero while preserving target verification.

Greedy Qwen3.8 MTP uses a proposal-only compact vocabulary projection containing
the first 98,304 tokenizer rows and the official control-token rows. A fused
Metal reduction maps its argmax back to the full tokenizer without materializing
the unused vocabulary tail. A request-local MTP cache is primed from up to 4,096
prompt hidden states and retains only target-confirmed transitions; speculative
rows execute on a disposable fork. The exact target projection still verifies
every emitted token. Per-request acceptance estimates adapt the draft depth and
can fall back to target-only rounds when proposals stop paying for their repair
cost. Sampled MTP keeps the full-vocabulary probability path.

Qwen3.8 target verification marks its model forward explicitly. On macOS, only
that marked path may use the fused BF16 GDN prework kernel, and only for batch
one, a four-tap depthwise convolution, 128-wide key/value heads, and sequence
widths 3–9. The kernel replaces convolution-state concatenation, depthwise
Conv1d, SiLU, q/k/v preparation, RMS normalization, scaling, and next-state
capture. Decode, prefill, unsupported shapes, and non-Metal platforms retain
the composed operations. Metal tests require bit-exact parity for widths 3, 4,
5, 7, and 9. Flash-Next does not use this BF16 normalization kernel: its GDN
Q/K normalization and recurrent inputs retain FP32, matching Qwen4Exp's
sum-of-squares epsilon and avoiding an intermediate BF16 rounding. The recurrent
output returns to the model dtype before gated output normalization. Other Qwen
architectures retain their existing path.

The streaming tool-call parser walks the Qwen XML structure rather than using
delimiter search. A closing tag is accepted only at its structural position,
so strings containing tag-like text remain parameter data. Streaming reparses
are bounded; EOS still receives a final structural parse.

Qwen3.8-Flash-Next uses the `qwen4_exp` text architecture. Its native path
implements four-stream gated residual hyper-connections, PLE n-gram hashing and
embedding lookup, grouped RMS normalization, sigmoid gated-delta output, and
the published hybrid full/linear-attention schedule. PLE lookup keeps only the
two preceding token IDs in each incremental cache. `Q38DiskNGramTable` maps its
128 logical embedding parts read-only from the existing safetensors shards.
Only requested packed rows are copied to MLX and dequantized with the same
affine operation as resident embeddings. No conversion or additional download
is required. The roughly 29.8 GiB table stays in reclaimable OS file cache
instead of the MLX allocation. Packed row reads begin before the first decoder
block, whose nonblocking MLX evaluation overlaps the storage work before PLE is
injected at the second block. PLE cache history is committed only when those
exact rows are consumed, preserving speculative rollback and replacement-input
vision prefills. Storage latency can still affect cold lookups when it exceeds
the first block's compute time.

The optional `MERERUN_PLE_STORE.json` placement manifest makes this transparent
for complete packaged checkpoints. When the checkpoint is on an external
volume, first load copies only the manifest's checksum-pinned PLE shard files
to `MereRunModelPaths.modelCacheBase` on the internal SSD and builds a PLE-only
safetensors index there. Later loads reuse the verified cache; all non-PLE
files remain in the configured model store. Internal checkpoints are used in
place, and insufficient internal free space falls back to the original shards.
Four-layer evaluation boundaries keep the 48-layer lazy graph below
the macOS Metal watchdog while preserving each hybrid block.

The same four-layer evaluation boundaries keep other deep MoE lazy graphs below
the macOS Metal watchdog while preserving each decoder block. Full-precision BF16 expert
weights use selected batched matrix multiplication because MLX's dense
`gatherMM` kernel accepts only FP32 weights.

Qwen3 learned-position vision towers materialize checkpoint weights in bounded
batches and synchronize after each vision block and final projection. This
keeps external-volume page-in and the full BF16 Ornith vision graph below the
macOS Metal watchdog without changing the checkpoint tensors.

QSA indexers run in every full-attention layer. Raw index keys are mean-pooled
in four-token blocks in FP32, normalized, and rotated at the block's first
position. Normalized/rotated index queries score those blocks by summed ReLU
head scores. Future blocks are masked before selecting the best 512 complete
blocks; the current partial block is always included. At or below the 2,048-token
budget every visible token is selected, so the original causal SDPA path remains
exact. Above it, 16-query tiles bound gathered KV temporaries without allocating
a dense sequence-squared attention matrix. The model's 262,144-token context
limit remains subject to total KV, MTP-history, and model residency.

`Q38QSACache` snapshots raw index keys, rotary positions, and completed pooled
keys alongside the main KV cache. Decode only pools, normalizes, and rotates
newly completed blocks; prefix forks and accepted-prefix MTP rollback retain
only valid complete blocks. Right-padded ragged batches recompute pooled keys
because padding changes as each row advances. Quantized main KV retains
unquantized indexer history. Tests
cover selector causality, GQA dense-mask parity, pool-before-norm/first-position
RoPE, chunk/serial parity, cache lifecycle, and the published 2,048-token boundary.

Qwen4Exp's bundled one-layer MTP head consumes the target's four-stream
hidden state, drafts through its trained hyper-connection/full-attention/MoE
block with the same QSA history, and leaves every emitted token under exact
target verification. MTP history priming uses 256-token chunks. Managed
Flash-Next models enable this path from short prompts; API startup warmup runs a
representative eight-token decode so its lazy target and draft graphs are paid
before the server reports healthy. `MERERUN_Q35_MTP_SPECULATION=0` keeps the
target-only path available for comparison or memory pressure.

Flash-Next retains a four-token verification block (three proposals), separately
from the dense 27B model's eight-token block. Small affine-Q4 projections whose
320/640-wide inputs or four-output gates do not fit the weight-reusing kernel
use native single-row QMV within a verification block, preserving serial
reduction order. Small BF16 readout, router, indexer, and hyper-connection batches
also use single-row GEMV, since the native matrix path can round differently
even at three/four rows. Flash-Next verification evaluates dense attention and
QSA against each row's exact causal history. This preserves serial attention
arithmetic and the per-token dense/QSA transition while retaining batched
Q/K/V, output projections, and expert work. Larger prefills remain batched.
Metal fixtures cover these projection shapes; component parity is not a substitute for
full-checkpoint greedy-output qualification.

Flash-Next expert selection and top-k normalization retain FP32 softmax
probabilities, matching the reference router. Only selected, normalized weights
are cast back to the model dtype; BF16 probabilities can otherwise collapse
distinct experts into ties before selection.

When verification rejects a proposal, PLE's n-gram token context and short
convolution state rewind to the same accepted prefix as attention. The
verification block retains only the inputs needed to slice those histories;
committing or restoring the block releases that temporary replay state.

Greedy Flash-Next prefills prime the draft history in aligned 256-token blocks.
Prefix checkpoints snapshot that history, its incomplete block, the final
four-stream hidden state, and target caches together. Reads and writes fork
both mutable cache trees, and a target-only entry cannot seed greedy MTP.
Only final prompts and semantic conversation boundaries are checkpointed,
not every intermediate prefill chunk. The four-entry retention bound remains
in force. The full prompt's hidden history is no longer retained or re-primed
on every eligible request.

Flash-Next also reclaims reusable MLX buffers when they reach 4 GiB at a
prefill chunk boundary, decode round, or new request (including exact prefix
replay, which skips prefill). Growing QSA shapes otherwise accumulate differently
sized temporary buffers until the much larger device-wide limit. This does
not evict model tensors, target/draft KV history, or prefix-cache entries, and
does not change the process-wide MLX allocation limits.

`Q38FlashNextCheckpointTests` provides explicit installed-model gates for 32k, 64k, and 128k
retrieval, exact prompt replay, a cached follow-up turn, and MTP agreement with
final-target decoding from the same saved prompt checkpoint. Set
`MERERUN_TEST_Q38_FLASH_NEXT_CHECKPOINTS=1`,
`MERERUN_TEST_Q38_FLASH_NEXT_MODEL_ROOT` to the mixed checkpoint directory, and
`MERERUN_TEST_MLX_DEVICE=gpu` when running a single context-specific test. Check
live memory headroom first and run these separately from the fixture suite;
the default test suite skips them and never downloads the checkpoint.
The optional `MERERUN_TEST_Q38_FLASH_NEXT_KV_CACHE_MODE=affine8` qualifies
the existing 8-bit KV-cache profile separately; it does not change model weights
or establish qualification for the default full-precision KV profile.
The 64k/128k `CleanTargetBaseline` tests additionally disable prefix caching,
and the `TargetPrefixReuse` tests exercise cache replay without MTP. Both require
`MERERUN_Q35_MTP_SPECULATION=0`. These isolate baseline retrieval, cache reuse,
and speculative decoding rather than treating a cached target comparison as an
independent clean prefill. Answers must complete within their token budget.

On the 2026-08-28 M4 Max / 128 GiB mixed-checkpoint runs with default KV, the
combined memory-lifetime/GDN-precision candidate passed all five 64k and 128k
cases. Initial prompts contained 64,740 and 129,700 tokens; peak physical
footprints were 53.1 and 59.9 GiB respectively, with normal pressure and no
positive swap growth. This covers retrieval, exact replay, MTP/serial output
agreement, a cached follow-up, and original-checkpoint isolation. The earlier
32k result predates the GDN precision change and was not rerun on this candidate.

The mixed checkpoint passes shapes and image order but fails the original exact OCR
fixture (`ORBIT 7429` becomes `ORBIT 7420`). Upcasting the encoder did not fix
it, so the mixed checkpoint's vision weights remain BF16. A layer-streamed independent Hugging Face
Qwen4Exp diagnostic, using BF16 reconstructions of the same mixed weights,
also ranks `0` above `9` after the teacher-forced prefix `ORBIT 742`, both with
native and independently computed image embeddings. Subsequent free-running
checks recover the original transcription with same-source Q4 expert weights,
and the all-Q4-experts control passes four additional regression cards.
Smaller selective-Q4 recipes have not yet passed the strict selection checks.
The published mixed checkpoint's OCR remains unqualified. These are bounded
local checks, not a general model-quality certification or a throughput claim.

A later Q4-derived low-bit experiment also remains unqualified. Q3/group-64
experts passed 12/12 synthetic calibration cases but only 16/17 earlier OCR
regressions. A fixed-code activation-weighted affine refit recovered all 17
regressions at a 57.5–58.3 GiB peak footprint, then passed 14/16 untouched
validation cases: one OCR identifier and one Python indexing answer failed.
Q2/group-32 passed 9/12 calibration cases under exact-output scoring. These
tests do not rule out quantizing from the original higher-precision checkpoint,
but they do rule out publishing either tested Q4-derived recipe as qualified.

With packed-vision loading fixed, the published Q4 checkpoint passes 17 exact
OCR cards: nine selection/regression examples and eight separate validation
examples. A native optimized-build repeat confirms all 17 transcriptions with
context 8,192 and a peak footprint of about 72.9 GiB at normal memory pressure.
The unchanged mixed model passes 4/8 on the independent set with the same
debug binary that gives Q4 8/8. These OCR timings are not decode benchmarks.
This does not extend the mixed checkpoint's 64k/128k qualification to Q4 or
establish general OCR accuracy.

`Q38FlashNextPerformanceTests/testInstalledMixedCodeAndProse` measures the
installed checkpoint with prefix caching and continuous batching disabled.
Despite the historical test name, the model-root override also supports Q4.
Build optimized tests with
`swift build -c release --build-tests -Xswiftc -enable-testing -Xswiftc -DDEBUG`
(the suite's existing test-only helpers require `DEBUG`; optimization stays on),
then run that single test from the release XCTest bundle with
`MERERUN_TEST_Q38_FLASH_NEXT_BENCHMARK=1`, the model-root and GPU variables above,
and `MERERUN_Q35_PREFILL_CHUNK_TOKENS=256`. Use separate processes with
`MERERUN_Q35_MTP_SPECULATION=0` and `1`. The JSON records separate loading,
prefill, and decode time, preserve output and acceleration diagnostics, and
include one initial request plus three repeated trials per workload. Compare
the median of trials 1 through 3; the first request does not imply a cold OS
file cache. Do not enable decode tracing or logprob capture for speed tests.

On the same 2026-08-28 M4 Max / 128 GiB machine, optimized Q4 runs at context
8,192 measured warm median decode rates of 50.0 tokens/s for code and 30.7
tokens/s for prose with MTP. A repeated target-only run measured 25.1 and 24.6
tokens/s, respectively: about 2.0x and 1.25x acceleration on these two prompts.
All generated text and token counts match across both target-only runs and
the MTP run. Each workload generates 256 tokens; these are short-context
decode measurements, not long-context or image-prefill throughput.

The current mixed checkpoint measured 39.4 tokens/s for code and 12.0 tokens/s
for prose with MTP, using the same binary and prompts. Its prose trials ranged
from 11.7 to 25.2 tokens/s. Q4 also had slow outliers, and the first target-only
code run had a median of 8.7 tokens/s before a repeat recovered to 25.1.
Normal desktop activity continued throughout; all trials are retained. These
results do not establish a stable Q4-versus-mixed speed advantage. Text-only
MTP peak footprints were 70.9 GiB for Q4 and 41.3 GiB for mixed, with at most
0.5 MiB swap growth. Concurrent residency with another model was not tested.

`Q38FlashNextVisionCheckpointTests` checks that all vision checkpoint tensors
map to runtime parameters, then tests color/shape recognition, OCR, and image
ordering on deterministic fixtures. It uses the same opt-in variables plus
`MERERUN_TEST_Q38_FLASH_NEXT_VISION_FIXTURES` for a writable fixture directory.
Vision stays on final-target decoding and does not seed text-only prefix caches.
Flash-Next's patch embedding includes bias even when the vision config omits
`patch_embed_bias`; an explicit config value still takes precedence.
Image-embedding prefills also retain the original prompt token IDs for PLE;
replacing image placeholder vectors does not replace their token identities.
The vision loader supports affine-quantized embeddings and linear layers,
including mixed bit widths and group sizes inferred from tensor shapes. It
reads only indexed vision tensors, preserves file ownership for duplicate
tensor names, and does not open language-only shards. Packed position tables
use quantized row lookup before interpolation; they are not loaded as BF16
embedding matrices.

`Q38ProjectionOCRCheckpointTests/testInstalledOCRManifest` runs exact OCR checks
against a separate JSON manifest. Use the installed-checkpoint and model-root
variables described in this section, and set
`MERERUN_TEST_Q38_FLASH_NEXT_OCR_MANIFEST` to the manifest path. Each item in the
`cases` array contains `id`, `image`, `prompt`, and `expected` strings. Expected
text is used for scoring only. The test trims leading and trailing whitespace;
internal spacing, punctuation, and case must match. Keep calibration examples
separate from validation examples when comparing quantization recipes.

`Q38VisionOracleTests/testInstalledVisionEmbeddingExport` exports the exact
preprocessed pixels, image grid, and native image embeddings for independent
encoder diagnostics without loading the language model. Set
`MERERUN_TEST_Q38_VISION_ORACLE=1`, the model-root/GPU variables above,
`MERERUN_TEST_Q38_VISION_ORACLE_IMAGE` to an image, and
`MERERUN_TEST_Q38_VISION_ORACLE_OUTPUT` to a new `.safetensors` path.
`MERERUN_TEST_Q38_VISION_ORACLE_DTYPE=fp32` optionally upcasts the encoder
for a precision diagnostic; the default export uses the runtime's BF16 weights.
The export refuses to overwrite existing evidence; it is not an OCR-quality gate.
