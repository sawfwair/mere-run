# Glimmer Q8 tiled execution with sampled decoding

After the reasoning/tool channel correction, full tiling with seven draft tokens
passes the sampled speed criterion: 31.52 tokens/s on the short prompt and
26.96 on the long prompt, versus 29.40 and 22.35 for the bracketing reference.
The corrected adapter passes the frozen paired-likelihood and repeated SDK
task-quality bars. A reference and two fresh optimized coding sessions pass.
Both configurations pass all seven advanced API phases.
The mode remains experimental and unselected. The earlier four-row comparisons
found no qualifying gain.
The [machine-readable results](glimmer-q8-tiled-sampled-2026-11-08-summary.json)
retain the production request samples, baseline blocks, and qualification scope.

## Sampling and measurement

Requests use temperature 1.0, top-p 0.95, top-k 64, and high reasoning. These
sampling parameters follow [Glimmer's published guidance](https://huggingface.co/meta-models/Muse-Glimmer-30B#best-practices),
which recommends high or xhigh reasoning for coding and agent tasks. Startup's
speculative warmup also uses the recommended sampled preset. Greedy throughput
is not used to qualify this experiment.

The unchanged checkpoint uses symmetric affine Q8/group-64 matrices and BF16
activations, KV, dense parameters, and DFlash2 assistant. Original normalization
is retained. No Q4 conversion or FP16 staging is selected. The reference uses the
previously qualified opt-in `head4` path, deferred scheduling, and cost fallback.

Each comparison uses one production Release executable, with a 512-token output
cap and normal EOS stopping. An opening and closing reference block each retain
three requests per workload. Each candidate retains three. The first attempt
per prompt/configuration is retained as warmup and excluded. Requests are
selected by resource qualification, never by speed or answer. The two workloads
have 86 and 7,173 prepared prompt tokens.

Decode throughput counts all generated tokens, including reasoning, over the
complete native decode loop. It excludes admission, model loading, prefill, and
tools. HTTP wall time excludes the separate token-count request. These are
throughput probes; a capped response is not evidence that a coding task finished.
HTTP requests are unseeded because the Messages wire protocol has no seed field.
Continuations and speculative acceptance can vary. Baseline drift and the small
sample count limit causal claims.

The corrected candidate uses these server-side experiment controls:

```sh
export MERERUN_MUSE_GLIMMER_Q8_VERIFY=tiled-all
export MERERUN_MUSE_GLIMMER_Q8_TILED_HEAD_SPLITS=1
export MERERUN_MUSE_GLIMMER_DFLASH_TOKENS=7
export MERERUN_MUSE_GLIMMER_DFLASH_DEFERRED=1
export MERERUN_MUSE_GLIMMER_DFLASH_COST_FALLBACK=1
export MERERUN_MUSE_GLIMMER_FUSED_NORM=0
```

The reference changes `MERERUN_MUSE_GLIMMER_Q8_VERIFY` to `head4` and
`MERERUN_MUSE_GLIMMER_DFLASH_TOKENS` to `3`. These controls select an experiment,
not a qualified default. Request sampling remains temperature 1.0, top-p 0.95,
top-k 64, with adaptive thinking and high effort. Under API version 2026-10-21,
sampling fields are omitted and these Glimmer defaults apply.

## Broad-scope production results

| Mode | Short-prompt median | Long-prompt median |
| --- | ---: | ---: |
| Reference, pooled opening/closing blocks | 37.24 tokens/s | 26.74 tokens/s |
| Tiled head | 34.26 tokens/s | 23.91 tokens/s |
| Tiled decoder projections | 29.58 tokens/s | 22.50 tokens/s |
| Tiled projections and fused MLPs | 27.56 tokens/s | 24.06 tokens/s |

All 40 request attempts passed resource monitoring. The ten warmups are retained
and excluded, leaving six reference samples and three samples per candidate for
each workload. The short-prompt reference shifts from 38.98 to 31.38 tokens/s
between blocks; the long-prompt reference shifts from 25.61 to 27.03. None of the
candidates beats both reference blocks on both workloads or passes the frozen
minimum 5% gain criterion. These measurements do not support selecting a broad
tiled mode for this setup.

## FFN-only production follow-up

This follow-up uses a fresh Release binary and repeats reference/FFN-only/
reference controls with the same sampled settings. Its rates are not pooled with
the earlier binary's reference samples.

| Workload | Pooled reference | FFN-only | Observed change |
| --- | ---: | ---: | ---: |
| 86 prompt tokens | 31.91 tokens/s | 31.24 tokens/s | −2.1% |
| 7,173 prompt tokens | 25.10 tokens/s | 24.29 tokens/s | −3.2% |

All 24 attempts pass resource monitoring. Six warmups are retained and excluded.
The short-prompt reference medians are 29.06 and 33.89 tokens/s in the opening
and closing blocks; the long-prompt medians are 25.47 and 24.73. These spreads
limit interpretation of small differences. FFN-only does not demonstrate an
end-to-end gain or meet the selection criterion on either workload. No prefill
gain is established. Its candidate HTTP wall medians are 16.87 and 35.22 seconds;
reference wall medians by block are retained in the JSON summary.

The quality bar remains unchanged. These three-draft-token candidates fail the
speed criterion, so their changed-arithmetic likelihood, coding, vision, and
desktop qualification is deferred. The existing selected Q8 configuration
remains in place.

## Eight-row follow-up

The existing seven-draft-token setting fills all eight target rows in the tiled
matrix geometry. This comparison uses the final Release binary, the same
recommended sampling, and new reference blocks. It separates wider speculation
from FFN tiling, full tiling, and head-only tiling. Head split setting 1 is fixed
before these runs; the earlier broad four-row comparison uses setting 2.

| Mode | Short-prompt median | Long-prompt median |
| --- | ---: | ---: |
| Three-draft reference, pooled opening/closing blocks | 32.23 tokens/s | 26.57 tokens/s |
| Seven drafts, reference kernels | 27.29 tokens/s | 21.89 tokens/s |
| Seven drafts, tiled FFNs | 42.42 tokens/s | 27.41 tokens/s |
| Seven drafts, full tiling | 46.08 tokens/s | 29.57 tokens/s |
| Seven drafts, tiled head only | 31.28 tokens/s | 26.30 tokens/s |

All 56 attempts pass resource monitoring, leaving 42 retained requests after
14 warmups. Head-only runs use their own immediately bracketing reference
blocks, whose pooled medians are 35.77 and 26.83 tokens/s. The reference-only,
FFN-only, and head-only seven-draft controls fail the frozen speed bar.

Full tiling improves decode throughput by 43.0% and 11.3% against its pooled
reference and beats each separate reference-block median on both workloads.
Its HTTP wall medians are 11.61 and 28.05 seconds, versus 16.38 and 29.97 for
the reference. Prefill medians are 0.474 and 10.751 seconds, versus 0.451 and
10.700; no prefill improvement is established. The gain depends on the combined
execution mode and wider verification geometry. Widening speculation alone
does not help in these samples.

The paired likelihood check passes on 32,768 held-out positions across 16
windows. Perplexity is 5.993286 for the four-row reference and 5.992928 for the
eight-row candidate: ratio 0.999940, with a 95% window-bootstrap upper bound of
1.000294, below the frozen 1.001 limit. Top-choice token agreement is 98.843%.
Both runs pass resource monitoring. This bounded text diagnostic does not
establish general quality or bit-exact native-Q8 behavior.

The diagnostic uses Release optimization with Swift DEBUG test hooks required
by existing unrelated H3 test fixtures. Glimmer, Tensor, Decode, KV-cache, and
Gemma computation has no DEBUG conditionals. The two completed Debug windows
from an intentionally stopped slow attempt match the optimized reference exactly
on 4,096 positions; the incomplete attempt is not used as a full qualification.
The first SDK comparison uses the exact frozen production executable from the
timing runs. Three reference cohorts score 20/20, 18/20, and 18/20; candidate
cohorts score 18/20, 19/20, and 18/20. The candidate has an extra failure on the
escape fixture and a duplicate tool call absent from the reference cohorts.
It fails the frozen per-fixture task-quality bar. Coding and desktop results
are not inferred from the likelihood result.

Investigation found an independent adapter bug: raw tool parsing includes
private reasoning, so a tool example in reasoning can become an executable
call. The corrected parser excludes reasoning from tool validation and preserves
its literal content. Real duplicate calls remain intact. Five new channel
regressions and the focused Messages/OpenAI suite pass: 93 Swift tests total.
The prior duplicate's exact origin remains unproven because the older stream
decoder removed tool markup from returned reasoning.

The corrected production results follow, using the same sampling, fixtures,
and quality bar. Model computation is unchanged, so the paired numerical
evidence remains applicable. The selected configuration stays unchanged pending
fresh desktop trials and the unlocked repository gate.

## Corrected production comparison

The corrected adapter uses a fresh standard Release executable. Its 24 request
attempts all pass resource monitoring, leaving 18 retained samples after six
warmups. Reference/candidate/reference controls retain three requests per
workload and use the same frozen sampling, output budget, and selection rule.

| Workload | Pooled reference | Seven drafts, full tiling | Decode change |
| --- | ---: | ---: | ---: |
| Short prompt | 29.40 tokens/s | 31.52 tokens/s | +7.2% |
| Long prompt | 22.35 tokens/s | 26.96 tokens/s | +20.6% |

The candidate beats both separate reference-block medians on both workloads.
These rates differ from the earlier eight-row comparison. They are kept
separate rather than pooled or replaced with the larger historical gains.
Unseeded continuations and adaptive speculative behavior vary; three retained
requests per group do not establish a universal speedup.

The long-prompt reference prefill medians shift from 17.060 to 12.263 seconds
between blocks, while the candidate is 13.793. No reliable prefill gain is
established. Fresh SDK checks pass on this corrected executable with the unchanged fixtures.
The mode remains unselected pending fresh desktop trials and an unlocked
repository-gate rerun.

## Corrected task comparison

Three unchanged 20-case SDK cohorts score 18/20, 18/20, and 19/20 on the
reference, versus 19/20, 18/20, and 19/20 with full tiling and seven drafts. The
candidate passes the frozen per-fixture criterion: no fixture has more failures
than its reference cohorts. Both configurations miss the requested sort key
twice; the newline fixture fails three reference and two candidate cases. One
candidate newline attempt exhausts its output budget on reasoning before a
visible answer. These failures remain failures.

All three nested-schema tool cases pass for each configuration, including exact
JSON types, Unicode, and the supplied reference schema. Prepared counts agree
with usage in the completed SDK responses. The same recommended sampling,
high effort, 2,048-token output budget, prompts, and grading apply throughout.
This bounded comparison does not establish general task equivalence. Fresh
Claude Code coding and advanced-combination results follow.

## Corrected Claude Code coding sessions

Claude Code 2.1.316 completes one reference session and two fresh optimized
sessions through the native Messages server. Each inspects requirements and
source, reproduces failing tests, repairs the two fixture modules, exercises the
requested missing-test tool error exactly once, and recovers with a successful
test command. Tests and requirements remain unchanged. Manual source and trace
review plus independent execution pass all 10 fixture and three holdout tests
in every session.

Client wall times are 429.82 seconds for the reference and 452.44 and 278.84
seconds for the optimized sessions. These unseeded agent trajectories use
different numbers of tool calls and do not establish a coding-task speedup.
Server loading and warmup are excluded from these session times. Both modes use
high effort, a 65,536-token context, and the advertised 4,096-token output limit.
The reference recovers from one output-limit stop during reasoning. Both
optimized sessions finish without an output-limit stop.

The client retries without `thinking.display.updates` after the server explicitly
rejects that unsupported option. No permission settings, MCP registrations, or
model identities are changed to make the tasks pass. Calculator and website
trials remain unqualified for this corrected candidate while macOS is locked.

## Corrected advanced combinations

Both reference and optimized configurations pass all seven phases with high
effort: regex search, BM25 search, image ordering and cache identity, PDFs,
capability errors, active/queued disconnect recovery, and prefill cancellation.
The official Anthropic SDK 2.6.0 accumulates the native streams. Search phases
combine deferred discovery, signed reasoning replay, nested PDF tool-result
errors, context edits, zero-output warming, and cache-marker boundaries.
Only catalog search executes on the server; the discovered client tool is
returned for client execution.

Ordered and swapped ALPHA/BETA images return the correct first-image label,
including after reuse of a preceding text-only prefix. Original and reversed
two-page PDFs return their codes in page order. Each PDF variant writes 136
cache tokens with the one-hour TTL and reads all 136 on generation; its count
is 138 prepared tokens. These checks verify immediate reuse, not an elapsed
one-hour retention guarantee. Generated citations remain unsupported and return
actionable 400 errors, as do unreadable PDFs.

Both modes cancel queued and active streams, return to idle admission, and
complete a subsequent READY request. The long-prompt case cancels during
prefill at 2,048 of 18,786 prepared tokens, before any generated token. Awaited
cleanup takes 3.37 seconds for the reference and 3.33 for the candidate; both
then complete recovery. Every phase ends with zero active and queued requests
and only the selected Q8 model loaded. Owned servers stop after qualification.

Ordinary advanced generations use a 2,048-token output budget. Cancellation
workloads request 4,096 tokens and close before exhaustion; warming requests
use zero. These functional probes are separate from the controlled throughput
comparison. Desktop qualification and the unlocked full repository gate remain
outstanding.

## Implementation and checks

`M5TiledAffineQ8` reorders original Q8 words and BF16 metadata without
requantization. Its M5 matrix kernels use 32-channel BF16 stages, FP32
accumulation, and optional FP32 partial reduction. The fused kernel interleaves
gate/up work and preserves the separate BF16 projection and activation
boundaries. The upstream backend's quantized-NAX safety guard is unchanged.

The modes separate the shared vocabulary head, all eligible decoder
projections, and added gate/up/SwiGLU fusion. The FFN-only control keeps
attention projections and the vocabulary head on the reference path. Independent
assistant and vision layers are unchanged. Both scope and head partition
settings enter the prefix-cache adapter identity.

All 144 synthetic GPU projection comparisons pass, including activations beyond
FP16 range. Inverse permutations preserve words and metadata exactly. Serial
and batched rows agree within each split setting. Relative RMS error stays below
0.005 against FP32 matmul of BF16-dequantized weights. Fused activations match
separate tiled operations exactly, including the installed checkpoint fixture.
This is not native-Q8 bit parity or a general task-quality result.

A parameter-update regression exposed that MLX can change array contents while
retaining Swift object identity. The adapter now invalidates and releases packed
views when quantized parameters are updated and uses current parameters through
the reference path. The final GPU regression passes.

The first broad cold warmup hit a GPU command-buffer timeout. A diagnostic that
materialized each layer before the full graph passed. Broad startup now bounds
first-use materialization with disposable one-row and four-row caches and checks
cancellation between layers. Two fresh full-model sampled smoke tests then
passed; the final production server also loaded successfully. These observations
do not establish the driver-level cause of the original failure.

Packing adds resident quantized buffers. The broad adapter selects 417
projections and 52 fused MLPs, adding 28.165 GB of packed views. Its Debug smoke
test peaked at 64.792 GB of active MLX allocation. FFN-only selects 156
projections and 52 fused MLPs, adding 22.029 GB; its smoke test peaked at
58.651 GB. These allocation measurements are diagnostics on the 128 GB M5 Max,
not claims about lower-memory machines. The FFN-only smoke overlapped a small
GPU test from the repository gate, so its timing is excluded.

No optimization is promoted without the frozen quality checks and corresponding
task evidence. The four-row broad candidates failed the speed criterion, so their
full quality sweeps were deferred. Historical SDK cohorts remain in the JSON
summary. Their results do not qualify the corrected production executable.

The corrected-source repository gate runs 5,919 tests: 5,396 pass, 493 skip, and
30 fail in protected run-record storage paths while macOS reports the screen
locked. Cocoa permission errors prevent those records from being reopened. No
Glimmer test fails. Strict lint passes across 2,775 files; the separate unchanged
post-test CLI help and hygiene checks pass. The docs build and three Python
qualification-command regressions pass. An unlocked full-gate rerun and fresh
desktop trials remain validation gaps.
