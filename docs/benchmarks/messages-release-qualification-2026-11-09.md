# Unlocked native Messages qualification on November 9

The full repository gate and all six fresh desktop cases pass on the established
Glimmer Q8 configuration at source commit `8ea41731`. This closes the unlocked-gate
and current-baseline desktop gaps. Experimental tiled execution remains off and
unselected. The [machine-readable record](messages-release-qualification-2026-11-09-summary.json)
retains counts, timing boundaries, configuration, and evidence hashes.

## Repository gate

The unmodified `./scripts/check.sh` completes with exit code zero. It runs
5,919 XCTest cases, with 493 skips, and 197 Swift Testing cases. Every one of the
30 protected run-record cases that failed while locked passes while unlocked;
file protection is unchanged. The official Anthropic SDK environment and
installed Ornith/Glimmer tokenizer fixtures are enabled. Strict lint, build,
CLI help, package boundaries, and hygiene checks pass.

These are CPU unit/contract checks. Optional checkpoint and GPU cases remain
opt-in; the earlier numerical and native qualification evidence is retained.

## Configuration and reference loop

Inference uses native mere.run Swift/MLX, the unchanged symmetric Q8/group-64
Glimmer target, and its BF16 DFlash2 assistant on the 128 GB M5 Max. Activations
and KV remain BF16. The established selection is `head4`, three drafts, deferred
scheduling, cost-checked fallback, and original normalization. No tiled mode,
normalization fusion, or Q4 conversion is selected.

Requests retain the published sampled preset: temperature 1.0, top-p 0.95,
top-k 64, and high effort. The context is 65,536 tokens and the advertised output
limit is 4,096. Claude Code 2.1.316 owns the agent loop and tool execution. Cua
Driver 0.38.0 uses standard permission mode and the unchanged 13-tool reference
MCP configuration. Only the connection-helper executable path changes to the
frozen production build; task prompts and tool behavior remain unchanged.
No permission-policy or MCP-registration changes are made.

The browser trials use installed Chrome 157.0.8103.38, verified against the
[Google stable release notice](https://chromereleases.googleblog.com/2026/11/stable-channel-update-for-desktop.html).
Each launches a separate temporary profile with the browser sandbox enabled.
The model operates Chrome through Cua UI tools. Personal profiles, browser
scripting, and direct method URLs are excluded from the task path. All model
inference is local; the browser accesses the public MDN website. Nonessential
Claude Code network traffic is disabled for the trials.

## Independently verified outcomes

| Task | First fresh run | Second fresh run |
| --- | ---: | ---: |
| Calculator 6 × 7 | Pass, 292.94 s | Pass, 245.04 s |
| Calculator 47 × 23 | Pass, 245.02 s | Pass, 261.94 s |
| MDN search → interface → getAll() | Pass, 389.37 s | Pass, 429.87 s |

Each Calculator trial starts a distinct process and has an observed initial
zero. Independent final window state and screenshots verify both the expression
and result: `6 × 7 = 42` or `47 × 23 = 1,081`. The trusted verifier checks the
Calculator executable and closes only the owned process after verification.

Each browser trial starts a distinct Chrome process and profile. Independent
trace reconstruction verifies a visible Search click, the `URLSearchParams`
query, the interface-result click, and the `getAll()` method-link click in that
order. Final window state, page text, screenshot, and isolated browser history
verify the method page and the answer about repeated and absent values. The
trusted verifier closes only the owned browser after inspection. Model-declared
success alone is insufficient.

The runtime records 90 completed native generation rounds and one failed
generation. The client records one tool error. Both errors occur in the second
browser case and are followed by successful recovery. One stream ends with an
unparseable-tool-parameter error; the server withholds the invalid executable
input, and Claude Code retries in buffered mode. The separate Cua focus error
dispatches no key; subsequent UI actions complete the required task. The precise
invalid parameter is not available in the validated-wire evidence.

These observations qualify the listed cases; they do not establish a general
agent error rate or an error-free model.

## Timing scope

The server reports 8.17 seconds loading and 41.04 seconds warmup before
readiness; process launch to readiness is 49.52 seconds. Compilation and
materialization remain in startup accounting. Those costs are excluded from
client task wall times.

Client wall times include inference, reasoning, Cua actions, and client work.
Native counter differences separate request load, prefill, and decode for
completed requests. Decode includes every generated reasoning, text, and
tool-input token in those completed requests. The failed generation has no
phase/token accounting in these counters; its failure and retry remain included
in client wall time.
Observed native decode rates span 15.44–19.44 tokens/s.
These unseeded agent trajectories and variable cache state are not a controlled
speed comparison. The first Calculator run overlaps the end of CPU repository
validation. Queue wait is not separately measured; requests are sequential,
and session-boundary snapshots have zero active and queued requests and only
the selected Q8 target loaded. Its draft companion belongs to that runtime.

## Qualification boundaries

The core API evidence includes official SDK coverage of both API versions,
HTTP authentication/error/cancellation tests, and the earlier signed reasoning,
search, context-edit, cache, document, and image combinations. This follow-up
changes no runtime source and does not promote experimental tiling.

Strict tools, constrained JSON-schema output, exact thinking budgets, generated
citations, scanned-PDF rendering, and server compaction remain unsupported and
rejected. Earlier extended-decimal and Gemma desktop-repeat failures remain in
the [qualification history](../internals/messages-qualification.md). Success on
these six cases does not qualify unlisted models, workflows, or hardware.

All six owned app processes, the native server, the task-owned Cua daemon, and
the keep-awake helper are stopped after verification. Other worktrees and
user processes remain untouched.
