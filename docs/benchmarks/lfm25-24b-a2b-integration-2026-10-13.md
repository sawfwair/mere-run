# LFM2.5 24B-A2B native integration and mixed-precision evidence

This report records installation, numerical checks, and qualification limits for
maintainers of the native LFM2 runtime. Date: October 13, 2026.

## Source and native installation

`text-chat-lfm25-24b-a2b-bf16` pins
[`LiquidAI/LFM2.5-24B-A2B`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/tree/1769516ac588c4c558089c94f27d96b0dc67b183)
at revision `1769516ac588c4c558089c94f27d96b0dc67b183`.
The native Swift runtime loads its ten original safetensor shards without
conversion, Python inference, remote code, or a DSpark companion. The LFM Open
License v1.0 acceptance gate applies.

The checkpoint has 40 hybrid layers, two dense feed-forward prefix layers,
64 experts in each of 38 sparse layers, and four selected experts per token.
Its vocabulary has 128,000 entries. It declares a 131,072-token context with
RoPE theta 10,000,000. The operational default remains 32,768 tokens.
Thinking visibility does not disable the template's reasoning generation.
Existing model IDs and defaults remain unchanged.

The [header audit](./lfm25-24b-a2b-header-audit-2026-10-13.json) covers every
original tensor name, shape, and dtype. Remote conversion also verifies each
complete source shard against the immutable Hub SHA-256 and size. The source
contains 7,610 tensors and 47,943,180,288 logical weight bytes. Expert correction
biases remain FP32; other source tensors are BF16.

## Mixed 6/8-bit derivative

The experimental `mixed6-8-quality` conversion contains **23,934,984,192 logical
weight bytes** in 41 safetensor shards. It uses affine group-32 quantization:

- Routed expert matrices in layers 2, 3, 6, 10, 14, 18, 22, 26, 30, 34, 38,
  and 39 use eight bits. These include attention-adjacent and boundary banks.
- The other 26 sparse layers use six bits.
- Quantization computes scales and biases in FP32 and stores them in FP16.
- Embedding, attention, convolution, dense prefix, routers, and norms retain
  their original BF16 values. Router correction biases retain FP32 values.
- The derivative explicitly requests FP32 activations. Original checkpoints
  retain their existing activation policy.

The [artifact audit](./receipts/lfm25-24b-mixed-artifact-audit-2026-10-13.json)
verifies all 41 shard hashes, 114 packed expert matrices, 228 affine metadata
tensors, and **314 bit-exact non-expert tensors**. There are 656 converted
entries. The conversion receipt SHA-256 is
`3c277d55ee8b50a5c08a4d6f80f9320e651c5d44f390bccf7c3558bd711654be`.
The original license and tokenizer files accompany the derivative.

This is a local-folder artifact, not a published managed download. To recreate
it from the pinned original snapshot, run the developer conversion tool:

```bash
python3 scripts/model-conversion/convert_lfm25_24b_mixed_mlx.py \
  --source /path/to/LFM2.5-24B-A2B \
  --output /path/to/LFM2.5-24B-A2B-MLX \
  --profiles mixed6-8-quality
```

To run the resulting folder, use explicit source sampling defaults and start
with a bounded context:

```bash
MODEL_ROOT=/path/to/LFM2.5-24B-A2B-MLX/mixed6-8-quality
MLX_ENABLE_TF32=0 mere.run text chat --model "$MODEL_ROOT" --model-root "$MODEL_ROOT" \
  --temperature 0.1 --top-p 1 --top-k 50 --kv-bits 8 \
  --context-size 8192 --max-tokens 1024 --prompt "Calculate 17 + 25."
```

Local family detection reads the typed checkpoint config and preserves path
case. A logical weight size is not a guarantee of available runtime memory. Admission still
applies. High-precision derivatives use 128-token prefill chunks, or 32 tokens
above a 32,768-token prompt, and release unused allocation caches per chunk.

## Reference and quantization accuracy

The authored suite has 20 calibration cases with 440 scored tokens and 40
heldout cases with 1,341 scored tokens. It covers prompts in all 16 supported
languages, coding, reasoning, extraction, and conversation history. Many
reasoning continuations are English. This is a short diagnostic, not a general
multilingual or capability benchmark.

Scoring uses exactly the same frozen teacher-forced tokens and all 128,000
vocabulary logits. The comparison computes KL(reference || candidate),
top-token agreement, and target perplexity. It selects by calibration KL and
a 24 GB logical weight budget. Heldout metrics do not select the policy.

BF16 arithmetic and MLX's default TF32 arithmetic introduced substantial
cross-implementation variation. The strict comparison disables TF32 and
streams FP32 computation over the original BF16 weight values. Independent
Transformers and MLX references then achieve **100% top-token agreement** and
mean KL **2.76e-11** over all 1,781 tokens. The
[reference receipt](./receipts/lfm25-24b-strict-reference-2026-10-13.json)
records every case. Software pins are Transformers
`28034545c35acac05793695bd5c878f66e6b544e`, MLX-LM
`ba8a08fd2ed72bbc53de84f89933f91fe4743e4a`, and MLX 0.32.1.

Heldout results from the
[complete comparison](./receipts/lfm25-24b-mixed-comparison-2026-10-13.json):

| Expert policy | Logical GB | Mean KL | Top-token agreement | Target perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| Eight-bit control, FP32 metadata | 27.86 | 0.010190 | 97.02% | 0.993386 |
| Six-bit group 32, FP32 metadata | 24.99 | 0.015151 | 95.67% | 1.003183 |
| Six-bit group 64, FP32 metadata | 22.12 | 0.017441 | 95.23% | 0.949846 |
| Mixed 6/8-bit group 32, FP16 metadata | **23.94** | **0.011956** | **96.27%** | **0.996894** |

The mixed policy has the lowest calibration KL among tested policies under
the weight budget. Its heldout agreement is 0.75 percentage points below the
larger eight-bit control, and its target perplexity is 0.35% above that control.

**No candidate passes the absolute diagnostic gates.** The mixed policy misses
the 0.01 mean-KL limit, 98% agreement limit, and some per-case limits. Its worst
heldout case has KL 0.038108 and target perplexity ratio 1.104297. The eight-bit
control also fails the absolute gates. The conversion remains
`quality_qualified: false`; neither improved average target perplexity nor
closeness to the control proves general quality preservation.

## Native correctness fixes

The MoE router uses sigmoid scores, applies expert bias only to selection,
normalizes unbiased selected scores with epsilon 1e-6, and applies the routed
scaling factor. This matches the pinned Transformers and MLX-LM implementations.
Per-projection typed metadata controls mixed packing and fused-kernel eligibility.
Synthetic native tests compare both FP16 and FP32 affine metadata with a
dequantized FP32 reference while preserving the BF16 trunk.

The native tokenizer now honors `ignore_merges` before applying BPE merges.
Real-checkpoint verification caught three mismatches in Indonesian, Hindi,
and English; all 60 frozen cases pass after the fix. A small synthetic fixture
protects whole-piece lookup, ordinary merge fallback, and special boundaries.
Three additional upstream vectors cover tool declarations, current-round
reasoning, and later-turn reasoning removal. Native formatting matches canonical
sorted tool schemas, including Jinja JSON spacing and Python scalar syntax.
Quoted strings remain distinct from booleans and nulls.

## Verification status

The work is isolated in `codex/lfm25-24b-a2b`, based on `01feae34`.
The original checkout is preserved. The final `./scripts/check.sh` passes strict
lint, build, CLI help, hygiene, and package checks, plus 5,450 XCTest cases
(451 skipped, zero failures) and 194 Swift Testing cases across seven suites.
The Linux delivery gate also passes. The iOS Release build and 20 test cases
(seven skipped, zero failures) pass; the arm64 executable is 41,696,648 bytes.
These platform checks do not establish trained-checkpoint inference on Apple
hardware. No hosted CI, PR, or publication has run.

Real GPU generation emits a complete `READY` answer, the arithmetic answer
`42`, a Python square function, and the expected
`calculate(expression='17 + 25')` tool call. An ambiguously worded arithmetic
probe exhausts its reasoning budget and is retained as a limitation.

Budget probes at 8,192, 32,768, and 131,072 tokens complete remotely with
eight-bit KV storage, bounded prefill chunks, and allocation-cache release.
Each generates 128 tokens and ends during reasoning. These establish execution,
not completed retrieval answers. The first 128K attempt was cancelled when
allocation approached host capacity; the bounded retry completes. The reference
tool records a dequantized attention fallback for CUDA KV shapes that lack a
native quantized matmul implementation. That fallback does not establish native
Swift CUDA performance.

The complete checkpoint and all 120 strict-reference and selected-candidate raw
logit files are preserved on external storage with independent SHA-256 checks.
The RunPod GPU is stopped and its pod reports `EXITED`; disks remain retained.

**Native short generation and the full-vocabulary diagnostic suite pass on
October 15.** Local long-context fit, general capability quality, hosted CI,
and release qualification remain separate.

### Native Metal qualification on October 15, 2026

The native CLI loads all 41 mixed-checkpoint shards and returns
**`17 + 25 = 42.`** through Swift MLX/Metal. The request uses 8-bit KV, an
8,192-token context limit, and a 1,024-token output limit; the actual prompt
is 17 tokens and generation is 111 tokens.

The run takes 67.91 seconds: 1.98 seconds loading, 33.17 seconds prefill, and
32.71 seconds decode, or **3.39 decode tokens per second**. System swap reaches
about 2.3 GiB. This establishes a bounded local request with tight headroom;
it does not establish long-context fit or comfortable concurrent app use.

Both installed-checkpoint tests pass with **zero skips and zero failures**.
The full-logit test completes all 60 cases in 74.54 seconds, and the independent
real tokenizer/tool-template test passes. It saves all 128,000 vocabulary logits
for the same 1,781 scored tokens used by the frozen remote diagnostic.
The [native comparison receipt](./receipts/lfm25-24b-native-metal-parity-2026-10-15.json)
verifies each saved reference hash before comparing distributions:

| Native Metal comparison | Scored tokens | Top-token agreement | Mean KL | Target perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| Same mixed checkpoint on MLX-CUDA, all cases | 1,781 | **100%** | **6.68e-9** | 1.00000624 |
| Same mixed checkpoint on MLX-CUDA, heldout cases | 1,341 | **100%** | **3.26e-11** | 1.00000052 |
| Original strict reference, heldout cases | 1,341 | **96.27%** | **0.011956** | 0.996895 |

Native execution reproduces the selected quant's remote distributions closely.
Its deviation from the original reference remains the quantization limitation
reported above: **absolute quality gates still fail**.

The original test-only startup rule required weight bytes plus 2 GiB already
available and repeatedly skipped before loading, including after cache clearing.
The final harness uses the same standard machine admission as production and
checks the default runtime memory-pressure policy before every case, failing at
critical pressure. It requires `MERERUN_TEST_MLX_DEVICE=gpu`, disables TF32, and
clears unused allocation caches between cases. Shared admission thresholds and
OS GPU-memory limits are unchanged. All cases complete without the pressure
stop firing. The earlier CPU-mode attempt was interrupted before producing an
output and is excluded from the result.

During the full diagnostic, sampled system-wide wired memory reaches
28.69 GiB and system swap reaches 6.19 GiB. These are host
totals, not model GPU-allocation measurements. The full diagnostic contains short
prompts only; native long-context fit remains unqualified.
