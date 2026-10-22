# Flint CUDA kernel and decode investigation

This report is for developers qualifying the native Flint runtime. It follows
the [source-build performance investigation](./flint-spark-performance-2026-10-21.md).
It separates kernel correctness, decode throughput, and Mac memory qualification.

## Quantized kernel defects

Repeated forwards with the same five input tokens and released INT4 weights
produced different logits. This persisted with a fresh PTX cache and with CUDA
graphs disabled. Per-operation traces isolated inconsistent routed projections.

The pinned MLX SM80 matrix kernel overlays its input and output shared-memory
buffers. It started writing output without draining outstanding asynchronous
input copies. Compute Sanitizer's race checker reported 21 error groups in a
small BF16 affine-INT4 gather reproducer. Waiting for all copies and synchronizing
the thread block before the epilogue reduced that result to zero hazards. The
released-checkpoint probe's repeated-forward maximum difference also fell to zero.
These are bounded results for the tested shapes, not a qualification of all CUDA
kernels or released-weight accuracy.

A second reproducer isolates accumulation precision. The decode matrix-vector
kernel used BF16 partial sums for affine INT4. With unit weights and four
512-element input blocks containing 256, 1, -256, and 1, the exact sum is 1,024.
The original CUDA kernel returned 512 for every output row. The patch retains
BF16 inputs and weights while accumulating BF16 products in FP32; FP16 policy
is unchanged.

`scripts/patch-mlx-cuda-quantized.sh` applies the source fixes during Linux native
preparation. It changes the SM80 module cache key so an older PTX entry cannot
mask the fix. Preparation rejects a compiled bridge without the updated key.
The patch test verifies ordering, repeat application, and rejection of source
layouts it does not recognize. No vendored distribution files are modified.

## Decode execution

Full-attention caches use a backing buffer sized to the request's
prompt-plus-output token budget. Updates write into that buffer and expose only
populated positions. Sliding caches retain their absolute-position semantics.

Single-token decoding submits four-layer blocks asynchronously. The decode
engine still evaluates the final logits before accepting a token. Prefill keeps
layerwise evaluation to bound its graph. Reference tests cover static cache
capacity, mixed sliding/full layers, chunk boundaries, and asynchronous versus
layerwise decode. Single-token attention also omits the all-zero mask because
every populated cache row is already valid for the query.

## Measured throughput

Both binaries use the corrected CUDA kernels and release-optimized libraries.
The baseline retains the original layerwise execution and growing caches; the
optimized binary uses the request-sized full-attention buffers, asynchronous
decode blocks, and single-token attention change.

| Requested output cap | Corrected baseline, tokens/s | Optimized Swift execution, tokens/s | With tuned CUDA graph grouping, tokens/s |
| --- | --- | --- | --- |
| 128 tokens | 32.04, 28.51 | 55.51, 55.82 | 72.76, 74.24 |
| 1,024 tokens | 24.32, 24.57 | 54.96, 55.63 | 73.58, 74.26 |

Each configuration has a warmup request before measurement. All requests use temperature
0 and the numbered-list prompt from the preceding investigation, under a normal
48 GiB scheduler reservation, forced native quantized kernels, and TF32 disabled.
No builds run during timing. Token counts include
reasoning; rates exclude loading and prefill. The optimized 1,024-token requests
take 33.92 and 33.79 seconds including approximately 14.4 seconds of loading.
These fixed-prompt trials do not reproduce the publisher's context protocol.

The tuned 1,024-token requests take 29.26 and 29.17 seconds including loading.
Their decode phases take 13.92 and 13.79 seconds. The optimized binary passes the
API cases and scalar fixture with both graph configurations. The larger graph grouping also retains exact
repeated forwards in the released-weight probe.

### Dedicated Flint CUDA profile

The pinned MLX backend gives GB10 conservative graph defaults: 20 operations and
a nominal 25 MB buffer threshold. Its accounting includes the full referenced
expert tensors. In the profile, the standard settings result in 207 graph
launches per decode forward, despite the four-layer asynchronous blocks. The
tuned profile reduces this to 16 launches per forward in the matching trace.
Quantized projections consume approximately 74% of measured decode GPU kernel
time. The profile region spans the first through final greedy token selection;
it excludes loading and prefill and includes profiler overhead.

This process-scoped profile was tested with the dedicated Flint CLI and API:

```bash
MERERUN_MLX_CUDA_NATIVE_QUANT=native MLX_ENABLE_TF32=0 \
MLX_MAX_OPS_PER_BUFFER=256 MLX_MAX_MB_PER_BUFFER=2048 \
mere.run api serve --engine text-chat-flint --model text-chat-flint-4bit
```

The variables control command-buffer grouping. The scheduler reservation remains
48 GiB. Other runtime families have not been qualified with this profile. This
investigation does not change process-wide defaults for other engines.

The optimized product also passes the five loopback API checks: arithmetic,
parsed Python addition, weather tool selection, synthetic tool-result history,
and HTTP 400 for unsupported JSON-object mode. The test does not contact a
weather service or execute generated code.

The scalar GPU fixture compares all 221 logits with a maximum absolute error of
`3.874302e-7`. The source-matched released-weight diagnostic selects the frozen
reference token at 5/5 positions for the full forward and one-token chunks,
and 4/5 for three-token chunks. Full-forward cosine similarity is 0.999861,
with maximum absolute difference 2.447390. Repeated full forwards agree exactly
in this probe. Chunk agreement and broad numerical equivalence remain open.
The diagnostic compiles the same model sources and portable quantization helper
as one module; it is separate from the CLI product exercised by the API tests.

The [machine-readable receipt](./flint-spark-kernels-2026-10-21.json) records
source hashes, executable hashes, timings, memory observations, and the
qualification boundaries.

## The 36 GB Mac

The local machine is an M3 Max with 36 GiB physical memory. Metal reports a
recommended maximum working set of 30,150,672,384 bytes, approximately 28.08 GiB.
All 13 focused Flint tests pass on this Metal device, including the cancellation
case and repeated routed projections. They do not load the released 22.6 GB
checkpoint.

From the published architecture, persistent BF16 keys and values require these
amounts for one sequence:

| Context tokens | Persistent KV cache |
| --- | --- |
| 4,096 | 0.144 GiB |
| 8,192 | 0.253 GiB |
| 32,768 | 0.909 GiB |
| 131,072 | 3.534 GiB |
| 262,144 | 7.034 GiB |

These figures exclude weights, loading peaks, temporary arrays, allocator
caches, and other applications. The earlier short Spark process peaked at
22.43 GiB. That Linux measurement supports investigating a bounded Mac run;
it does not establish a Metal memory limit. The catalog retains its 48 GB floor.
Released-weight Metal execution and a 36 GB configuration remain unqualified.
Recursive's specialized Metal engine supports M4 Pro and M5 Pro/Max devices;
the M3 fixtures exercise this implementation's general Metal path.

## Validation boundaries

The independent scalar fixture remains the math and cache reference. Separate
GPU tests exercise repeated BF16 routed projections and the exact cancellation
case. CPU execution of BF16 quantized weights is not qualified by those GPU
checks. The CPU backend also fails the cancellation stress case; this CUDA patch
does not qualify CPU execution of the released quantized model.

The full Mac gate encounters protected run-record read failures while the
session is locked: 52 failures among 5,529 XCTest cases, with 461 skips. The 194
Swift Testing cases pass. None of these failures belongs to the Flint suites.
The protection remains enabled. Full-gate completion,
released-weight numerical agreement, broader quality, long-context throughput,
publisher performance parity, and distribution qualification remain separate
requirements.
