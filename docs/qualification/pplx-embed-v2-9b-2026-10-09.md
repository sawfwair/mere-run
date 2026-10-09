# PPLX Embed v2 9B qualification — October 9, 2026

Both packed checkpoints passed bounded native public CLI checks. No Python
inference process is used locally. Users pull the packed artifacts directly;
conversion is a publisher task.

| Checkpoint | Weight files | FP32 vs packed mean cosine | Minimum cosine | Retrieval top choices |
| --- | --- | --- | --- | --- |
| text-embed-pplx-v2-late-9b-mixed-4bit | 7.32 GB | 0.999065 | 0.991924 | 8/8 agree |
| text-embed-pplx-v2-context-9b-preview-8bit | 10.81 GB | 0.999873 | 0.999813 | 8/8 agree |

These short-input checks do not establish maximum context or concurrent
residency. Original FP32 9B managed entries retain their separate admission
guidance.

Late uses affine Q4/group-64 transformer projections and Q8/group-64 token
embeddings. Context uses Q8/group-64 for both. Recurrent gates, convolution,
norms, vision, and final projections retain FP32. Context's int8-valued output
contract is independent of weight quantization.

The context Q4 candidate kept the same eight top choices but had mean cosine
0.980567 and minimum cosine 0.970761. Q8 improved these to 0.999873 and 0.999813,
so Q8 was selected. Selection used this same small suite: do not interpret it
as an independent heldout benchmark. `quality_qualified` remains false.

## What was measured

Conversion used MLX CUDA 0.32.2. Independent PyTorch integer unpacking was checked
against MLX dequantization samples, then FP32 and reconstructed packed weights
ran with Torch 2.8.0 and Transformers 5.4.0. The multilingual suite includes
eight queries and eight documents, marker/Unicode/punctuation cases, contextual
empty chunks and shared chunk boundaries, and one synthetic image for late.
There are 19 late and 20 context cases. Both profiles pass minimum cosine 0.95,
mean cosine 0.98, finite vectors, and complete top-choice agreement gates.
The legacy late report field `passes_heldout_gates` names these diagnostic gates;
it does not establish broader heldout quality.

Native CLI validation compares token/vector counts and every output vector
against the exported reconstructed-weight reference. Late covers text and a
synthetic image; context covers query/document chunks. A separate real-tokenizer
fixture verifies NFC-normalized decomposed accents, Hangul, empty chunks, emoji,
and literal markers against Transformers token IDs and offsets. A further full-checkpoint check
passed 1024-dimensional normalization across all 11 document cases, preserving
the empty chunk as a zero vector.

Broad retrieval benchmarks, image retrieval quality, small/large cross-model
alignment, maximum-context behavior, and future context-preview compatibility
remain unqualified. Context-preview embeddings must stay separate from later
checkpoint versions.

## Artifacts and reproduction

- [Sawfwair/pplx-embed-v2-late-9b-MLX-Mixed-4bit](https://huggingface.co/Sawfwair/pplx-embed-v2-late-9b-MLX-Mixed-4bit/tree/7309bd8d28fd8e0a9a27033f3d13cc2d7a5cf219), source `perplexity-ai/pplx-embed-v2-late-9b@77e936a1b18ed2ac00b7c76fccd70dc6a1bb1c18`. [CUDA diagnostic](https://huggingface.co/Sawfwair/pplx-embed-v2-late-9b-MLX-Mixed-4bit/blob/7309bd8d28fd8e0a9a27033f3d13cc2d7a5cf219/PPLX_QUALIFICATION.json), [native proof](https://huggingface.co/Sawfwair/pplx-embed-v2-late-9b-MLX-Mixed-4bit/blob/7309bd8d28fd8e0a9a27033f3d13cc2d7a5cf219/PPLX_APPLE_QUALIFICATION.json), [checksums](https://huggingface.co/Sawfwair/pplx-embed-v2-late-9b-MLX-Mixed-4bit/blob/7309bd8d28fd8e0a9a27033f3d13cc2d7a5cf219/SHA256SUMS).
- [Sawfwair/pplx-embed-v2-context-9b-preview-MLX-8bit](https://huggingface.co/Sawfwair/pplx-embed-v2-context-9b-preview-MLX-8bit/tree/abf77a86a7b84aee72640c736a92f3c5b471e6db), source `perplexity-ai/pplx-embed-v2-context-9b-preview@b667039ee8b438a6350fbc91bbcecd86f9d363ba`. [CUDA diagnostic](https://huggingface.co/Sawfwair/pplx-embed-v2-context-9b-preview-MLX-8bit/blob/abf77a86a7b84aee72640c736a92f3c5b471e6db/PPLX_QUALIFICATION.json), [native proof](https://huggingface.co/Sawfwair/pplx-embed-v2-context-9b-preview-MLX-8bit/blob/abf77a86a7b84aee72640c736a92f3c5b471e6db/PPLX_APPLE_QUALIFICATION.json), [checksums](https://huggingface.co/Sawfwair/pplx-embed-v2-context-9b-preview-MLX-8bit/blob/abf77a86a7b84aee72640c736a92f3c5b471e6db/SHA256SUMS).

See the [conversion runbook](https://github.com/sawfwair/mere-run/blob/main/scripts/model-conversion/pplx-embed-v2.md)
and [native runtime guide](../runtime/pplx-embed-v2.md). Each artifact preserves
its actual converter, source provenance, upstream license information and model card, and weight
reconstruction measurements. Publication staged privately, verified every file's
size and SHA-256, then verified the public immutable revision anonymously. The
Hugging Face write token stayed on the Mac. The conversion pod was terminated
after all selected weights and evidence were preserved locally; its attached
volume was removed with it.

The [JSON record](../benchmarks/receipts/pplx-embed-v2-9b-2026-10-09.json) contains immutable pins, per-batch parity, measured
verified file hashes, and the labeled allocation-cost estimate. Machine metadata
is omitted from this derived record; the linked immutable artifacts preserve
the original evidence.

## Repository validation

The full repository gate (`./scripts/check.sh`) passed on the isolated PPLX
branch against `main`:
5370 XCTest cases (442 skipped, zero failures), 192 Swift Testing cases,
build, lint, policy checks, CLI help checks, and hygiene scans. The documentation
build also passed. Real-tokenizer fixtures separately passed for both 9B
tokenizers.

Both managed `model pull` commands recognize their complete local installs.
The full-checkpoint native runs above passed with normal admission. A later
managed-ID inference retry was blocked by available-memory admission. Other
active workloads can affect admission even when an individual checkpoint run
has passed.
