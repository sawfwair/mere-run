# Flint INT4 Spark validation

On 2026-10-21, the released Recursive Flint INT4 checkpoint ran through the
native Swift/MLX runtime on NVIDIA DGX Spark. The CLI and local API completed
bounded generation and tool-use checks. Numerical comparisons preserved the
tested next-token choices, with floating-point differences between execution
paths. This receipt does not establish exact numerical equivalence, broad
model quality, or Metal behavior.

## Source and environment

| Item | Value |
| --- | --- |
| Runtime source | `9b03cb2a37975a4efea01f20ff69b1177f3681d1` |
| Checkpoint | `RecursiveSI/flint-int4@3b93f9053b530488405ae9fca51d4900d6ce21f0` |
| Architecture reference | `recursive-oss/rec@e53663b262af4cb1ca498e596ca8b1a2d79e0800` |
| MLX Swift CUDA bridge | `sawfwair/mlx-swift@001d1aa3e5655d7efc073fb9632f952f57cdf528` |
| Host | Linux aarch64, Ubuntu 24.04.4, NVIDIA GB10 |
| Visible RAM | Approximately 116 GiB |
| NVIDIA driver | `580.159.03` |
| Toolchain | Swift 6.3.3, CUDA 13.0 |
| CLI build | SwiftPM debug |
| Independent evaluator | PyTorch `2.10.0+cu130` with explicit affine INT4 unpacking |
| Admission | Normal shared scheduler tickets; 48 GiB reservation for checkpoint checks |

All five safetensors shards and the pinned configuration, template, and
tokenizer files matched the SHA-256 values enforced by `FlintArtifactVerification`.
The native CLI exercised that production loader. Validation used an isolated
source worktree and model directory. Existing services retained their normal
scheduler lifecycle; no service configuration or deployed model identity was
changed.

The CUDA bridge was reused after checking its exact source revision and native
layout. Linux SwiftPM wrote its own dependency resolution; that resolution and
the built executable's digest are retained with the raw receipts. The only
source patch during Linux testing restricted an existing macOS thermal benchmark
to macOS. Runtime sources remained at the revision above.

## Numerical checks

The synthetic three-layer fixture compared all 221 logits against the independent
scalar reference. With `MLX_ENABLE_TF32=0`, GPU maximum absolute error was
`3.874302e-7`. Chunk sizes 1 and 3 differed from the full forward by at most
`3.9115548e-7` and `2.719462e-7`, respectively.

The default CUDA TF32 mode exceeded this fixture's strict FP32 tolerance:
maximum absolute error was `0.0007788837`. The TF32-disabled run isolates that
precision difference; it does not change the product's default settings.

The released-weight probe used token IDs `[1, 33, 204, 1024, 42]` and compared
all 750,720 logits. The independent evaluator used PyTorch operations rather
than importing the Swift model. Per-layer outputs and complete logit arrays are
retained with the receipts.

| Comparison | Maximum absolute difference | RMS difference | Cosine similarity | Matching next-token choices |
| --- | --- | --- | --- | --- |
| Native versus PyTorch | 1.332644 | 0.239264 | 0.99990165 | 5/5 |
| Native chunk size 1 versus full forward | 1.183489 | 0.230541 | 0.99994332 | 5/5 |
| Native chunk size 3 versus full forward | 2.120371 | 0.298055 | 0.99984664 | 5/5 |

All three comparisons selected `[3157, 10444, 332, 5023, 332]` at the five input
positions. These observations establish bounded agreement, not an exact parity
pass. The independent PyTorch run also warned that its advertised CUDA
capability range ended at 12.0 while GB10 reports 12.1; its operations completed,
but that warning remains part of the reference receipt.

CUDA automatic dispatch selected native `quantized_mm` and dense compatibility
`gather_qmm` in the numerical probe. The CLI and API selected dense compatibility
for both operations. Do not describe these runs as qualification of native
quantized routed-expert CUDA kernels.

## Product checks

The public debug CLI answered `What is 2 + 2? Answer with a single digit.` with
`4`. That one cold request reported:

| Measurement | Result |
| --- | --- |
| Model load | 15.96 seconds |
| Prefill | 1.50 seconds |
| Decode | 10.93 seconds |
| Generated tokens, including reasoning | 47 |
| Decode rate | 4.30 tokens/second |
| Process maximum RSS | 23,516,032 KiB |

The separate numerical probe reported an MLX peak allocation of 22,822,948,104
bytes and process maximum RSS of 23,777,748 KiB. These are short-prompt
measurements. They do not justify lowering the 48 GiB admission estimate.

The loopback OpenAI-compatible API passed these checks with the same released
checkpoint:

| Request | Verified result |
| --- | --- |
| Arithmetic | HTTP 200; visible response `4` |
| Python code | HTTP 200; parsed `add(a, b)` function returning a sum |
| Tool selection | HTTP 200; `get_weather` call with `{"city":"Toronto"}` |
| Tool-result history | HTTP 200; final reply used the supplied 17 C weather result |
| Unsupported JSON-object response format | HTTP 400 |

The API model remained loaded across these requests. Tool results were synthetic
fixtures; validation did not contact a weather service or execute model-generated
code. The temporary API process exited after the checks.

Managed-ID preflight with `--model text-chat-flint-4bit --model-root "$CHECKPOINT"`
also reported `status: ok`, `installed: true`, and no diagnostics. To repeat the
CLI generation check after preparing the native CUDA environment, run:

```bash
mere.run text chat --model text-chat-flint-4bit --model-root "$CHECKPOINT" \
  --prompt "What is 2 + 2? Answer with a single digit." \
  --temperature 0 --max-tokens 128 --quiet --stats
```

After qualification, both authenticated owner-service health checks returned
HTTP 200 with `ready: true`. The three owner service units remained active,
the scheduler had no live tickets, and the temporary API port was closed.

## Test and release boundary

The focused Mac and Linux runs each passed all 18 Flint tests. The unrelated
installed Bonsai benchmark skipped on the Mac without its opt-in assets.
Linux tests used `MERERUN_TEST_LINUX_MLX=1` to execute their CPU math instead of
the default Linux skip path. Test discovery also needed the prebuilt MLX bridge's
include paths passed with `-Xswiftc -I -Xswiftc <path>`; target-specific package
settings alone do not reach SwiftPM's generated discovery target.

The [machine-readable receipt](./flint-spark-2026-10-21.json) records the bounded
results and synthetic API responses. Complete arrays, reference scripts, build
logs, and unsuccessful attempts are retained in the private evidence bundle.

The full Mac repository gate remains blocked by protected run-record reads while
the session is locked. Spark checks do not replace that gate. The released BF16
checkpoint, long-context accuracy, broad tool-use quality, sustained throughput,
Metal inference, and packaged Linux distribution checks remain outstanding.
