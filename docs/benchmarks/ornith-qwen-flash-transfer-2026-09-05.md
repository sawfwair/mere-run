# Flash-Next optimization transfer to Ornith and Qwen 27B

This work transfers three parts of the Flash-Next implementation to
Qwen3.8-27B Q4 and Ornith 1.5 Q4: consistent MTP admission, streamed prompt
history with prefix-cache snapshots, and exact verification through width
nine. The source baseline is `03d619a7` (abbreviated), dated September 5, 2026.

Both installed Q4 checkpoints pass long-prompt retrieval, exact cache replay,
follow-up reuse, and agreement with serial greedy decoding. Generation block
defaults remain eight for Qwen 27B and four for Ornith. Verification throughput
and complete generation throughput are measured separately.

## What the Flash-Next result establishes

[Merged PR #421](https://github.com/sawfwair/mere-run/pull/421) records these
warmed results on its final head, `01d4e6ff8488dbecdf7a1d66dccc97b99dc67479`:

| Workload | Serial decode, tokens per second | Forced MTP, tokens per second | End-to-end speedup |
| --- | ---: | ---: | ---: |
| Code | 27.4 | 55.8 | 1.94× |
| Math | 27.5 | 58.3 | 1.98× |
| Prose | 27.8 | 33.7 | 1.20× |

The PR reports exact output agreement for these greedy workloads. Its
width-32 verifier averages 180.5 verified tokens per second, excluding draft
cost. The production Flash-Next block remains four tokens. The experimental
tree decoder is excluded from the merge. These measurements don't predict
Ornith or Qwen 27B speed, or isolate the contribution of each kernel.

## Shared architecture and implemented paths

The [Qwen 27B configuration](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json)
uses a dense Qwen3.5 architecture with 64 layers and hidden size 5,120.
The [Ornith configuration](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/raw/main/config.json)
uses a Qwen3.5 mixture of experts with 40 layers, hidden size 2,048, and eight
selected experts from 256. Both use Gated DeltaNet, full attention, and a
one-layer multi-token prediction (MTP) head.

The following paths were already shared at the source baseline:

| Optimization | Qwen3.8-27B Q4 | Ornith 1.5 Q4 | Relevant boundary |
| --- | --- | --- | --- |
| MTP decoding | Enabled by default; block size eight | Enabled by default; block size four | Vision, constrained output, and tool-stop paths have separate admission rules |
| Compact draft vocabulary and GPU token selection | Shared | Shared | Greedy proposals use a smaller vocabulary; target verification retains the full vocabulary |
| Adaptive draft depth and accepted-prefix rollback | Shared | Shared | A shared fixed draft-cost ratio of 0.18 remains a tuning hypothesis |
| Exact affine Q4 projection kernel | Shared for supported widths two through nine | Shared for supported ordinary projections | Requires affine Q4, group size 64, and compatible BF16 tensors; this doesn't cover every expert route |
| Fused GDN preparation | Shared for supported widths one through nine | Shared for supported widths one through nine | Preserves each architecture's normalization convention |
| Fused expert gate/up weights | Not applicable to dense FFN | Implemented | Preparation is bounded to one decoder layer at a time |

The BF16 Qwen 27B model keeps MTP opt-in. Its checkpoint hasn't received the
Q4 qualification described here.

## Consistent admission and streamed draft history

Previously, `Q35Generator.chat` checked prefill MTP eligibility with
`enabledByDefault: loadedConfig.textConfig.usesMoE`. The decode path adds
`modelId == Q35Resources.q38TwentySevenB4BitModelId` to the same policy.

Consequently, a default Q4 request could enter MTP decode without preparing
the prompt history used to prime its drafter. Prefill and decode now call the
same model-admission policy, preserving their request-specific restrictions.
Tests cover default Q4, explicit enable/disable, BF16 opt-in, and Ornith.

Eligible greedy Qwen 27B and Ornith requests stream confirmed prompt
transitions into the MTP cache in aligned 256-token blocks. This reuses the
Flash-Next history mechanism with the models' ordinary KV cache, without
retaining every target hidden state. Semantic prefix checkpoints store the
target caches, MTP cache, pending transitions, and last hidden state together.
Cache reads and speculative rounds fork mutable state.

The optional `draftHistoryTokens` diagnostic reports confirmed prompt
transitions available before decode, including pending transitions. A prompt
of N tokens has N-1 transitions. `MERERUN_Q35_MTP_STREAM_HISTORY=0` restores
the earlier history strategy: retain dense history up to 4,096 tokens; otherwise
start without prompt history. Flash-Next's existing behavior is unchanged.
The value `none` omits prompt history entirely for ablation while preserving
the shared MTP admission policy.

## Exact verification kernels and benchmark admission

Qwen Q4 fixtures cover its 5,120-wide dense projections at widths four, eight,
and nine. They verify the existing small-batch Q4 kernel against serial QMV
with exact tensor agreement. The production block override remains capped at
nine; no new width-16 or width-32 Qwen path is enabled.

The gathered expert kernel now supports affine Q4/group-64 alongside the
existing Flash-Next Q3 specialization. Ornith target verification uses it at
widths two through nine for the 2,048/512, 256-expert, top-eight geometry.
Router and shared-gate projections retain serial arithmetic, including the
installed checkpoint's Q8 gates. The expert path supports fused gate/up
weights and rejects unsupported layouts. `MERERUN_Q35_EXACT_Q4_EXPERTS=0`
selects the previous expert execution path. Q6, Q8, BF16, prefill, and other
expert geometries retain their existing paths.

Full-model qualification exposed a separate attention issue: MLX's Metal
vector SDPA requires `query_rows * grouped_query_factor <= 32`. The previous
five-row split works for Qwen 27B's factor of six but exceeds Ornith's limit
of four rows at a factor of eight. Verification now derives this limit from
the head geometry. Blocks crossing a Metal pass-count or reduction-partition
boundary use each row's serial KV window. GPU fixtures exercise those
boundaries through 65,536 cached tokens. The complete Ornith checkpoint now
preserves the oracle at widths one, four, eight, and nine.

`model benchmark q38-verification` now accepts Qwen 27B and Ornith Q4, with
default widths `1,4,8,9`. Flash-Next retains `1,4,8,16,32`. The command rejects
unsupported widths before loading weights. Oracle-fed verification excludes
draft cost and doesn't establish end-to-end generation speed.

Keep Flash-Next PLE disk tables, QSA indexers, four-stream hyper-connections,
and its FP32 GDN normalization specific to that architecture. Ornith and Qwen
27B don't have those components. Reuse the verification techniques while
preserving each model's math.

## Checkpoint correctness

The opt-in `Q35TransferCheckpointTests` ran against both installed checkpoints.
Each used a 4,944-token retrieval prompt and a 4,973-token follow-up:

| Check | Qwen 27B Q4 | Ornith 1.5 Q4 |
| --- | --- | --- |
| Cold prompt has 4,943 draft-history transitions | Pass | Pass |
| Exact replay reuses all 4,944 prompt tokens | Pass | Pass |
| MTP response and token count agree with serial greedy decoding | Pass | Pass |
| Follow-up reuses a prefix and has 4,972 transitions | Pass | Pass |
| Replaying the original after the follow-up preserves its output and history | Pass | Pass |

GPU fixtures also pass for streamed-versus-retained draft proposals, snapshot
isolation, Qwen Q4 projections, Ornith experts with Q8 gates, and every
accepted-prefix GDN rollback at widths four, eight, and nine. The existing Q3
gather specialization remains exact in the same kernel test. A separate run
with `MERERUN_Q35_FUSED_SWITCH_GLU=0` verifies unfused Ornith expert execution.

The installed checkpoint tests run only when
`MERERUN_TEST_Q35_TRANSFER=1`, `MERERUN_TEST_Q35_TRANSFER_MODEL_ID`, and
`MERERUN_TEST_Q35_TRANSFER_MODEL_ROOT` are supplied. The normal test suite
doesn't load checkpoints or download weights. These first correctness runs
overlapped an unrelated inference job; their timings are not speed receipts.

## Measurement method

The machine is an Apple M4 Max with 128 GiB unified memory, running macOS
26.5.2 (25F84). The installed manifests identify these weight sources:

| Target | Target repository and revision | MTP repository and revision |
| --- | --- | --- |
| Qwen 27B Q4 | `EigenLabs/Qwen3.8-27B-4bit` at `eda45ab47f465d08d6558f0353a2346e2eb9d5b3` | `morgan/qwen38-27b-mtp-r20k-lr3-q4-g64-q2-rerank` at `fd4a99c590dd6e468c0e2a28168c235e32151a4b` |
| Ornith Q4 | `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` at `19504d912fa8fc7622bf6b1de3db5d5d890b1f02` | `ornith-ai/Ornith-1.5-35B-A3B` at `e4dfb35a93d4b6822a811a7676f3488514abe7e2` |

The existing `q36-mtp` command supports both models and warms each variant.
With installed checkpoints and an optimized CLI built from the assessed tree,
collect a code workload:

```bash
.build/release/mere.run model benchmark q36-mtp \
  --model vision-chat-q38-27b-4bit \
  --prompt 'Write a complete Python LRU cache using a dictionary and a doubly linked list.' \
  --decode-tokens 256 --context-size 8192 --temperature 0 --top-p 1 --json
```

Repeat with `--model text-agent-ornith-35b-mlx-4bit`. Each report includes
baseline, adaptive, and forced MTP variants. The adaptive variant unsets the
MTP environment override; the forced variant enables it. Their Q4 histories
now use the same admission decision. The actual `acceleration.route` and
`draftHistoryTokens` fields identify the runtime path and preparation.

The component comparison uses the same candidate binary for both arms. The
ablation sets `MERERUN_Q35_MTP_STREAM_HISTORY=none` and
`MERERUN_Q35_EXACT_Q4_EXPERTS=0`; the candidate sets both to `1`. This compares
unprimed drafting and the previous expert path with the transferred work. It
doesn't measure an older executable: both arms include the admission and
attention fixes. The target-only variant in every run provides the complete
serial-generation reference. Each variant receives a 16-token warm-up before
the measured 256-token request, with prefix caching disabled.

Run at least three warmed trials for code, math, and prose, then repeat with
long prompts and a cached follow-up. Record the checkpoint revision, build,
prompt, output tokens, load/prefill/decode time, accepted and drafted tokens,
verification and replacement passes, memory, and swap growth. Run without
competing GPU jobs. Compare each candidate with the same model and precision.

Require exact greedy output and cache rollback agreement. Require a repeatable
end-to-end improvement, with an initial target of at least 8% and no material
memory regression. Treat that percentage as an experiment decision threshold.
Run sampled quality checks separately from deterministic parity checks.

Before a runtime PR, run `./scripts/check.sh`. Report installed-checkpoint and
GPU qualification separately, following the repository's validation boundary.
