# Messages integration qualification

This record separates transport evidence from model task outcomes. A successful
connection preview or a model's own success statement does not qualify a model.

## Environment

- Base: committed `main`, `0b64fac1`.
- Branch: `feat/native-messages-api` in a separate worktree.
- Hardware: Apple M5 Max (`Mac17,7`), 128 GiB unified memory.
- Claude Code: `2.1.316` installed.
- Official Anthropic Python SDK: `2.6.0`, installed in a temporary test environment.
- Glimmer, Qwen, and Gemma checkpoints are present in the local inventory.

## Protocol evidence

`MessagesHTTPTests` starts actual loopback HTTP routes with a deterministic
runtime fixture. `scripts/test-messages-sdk.py` uses the official SDK against
that fixture. It covers buffered responses, stream accumulation, typed custom
tools, reasoning replay, token counting, both API versions, and local search.
The fixture does not run a model and provides no inference-performance evidence.

To repeat the official SDK checks, install `anthropic==2.6.0` in a temporary
Python environment, then run:

```sh
MERERUN_ANTHROPIC_SDK_PYTHON=/path/to/environment/bin/python \
  swift test --filter MessagesHTTPTests
```

Without this variable the SDK case is skipped; a skipped case does not qualify
the protocol. All SDK results in this record used the installed SDK environment.

`MessagesContentTests`, `ChatContextEditingTests`, `ChatToolSearchTests`, and
`ChatPrefixCacheTests` cover split delimiters, nested JSON, malformed arguments,
stop boundaries, context retention, deterministic search, cache expiry, eviction,
authentication identity, and changed-prefix invalidation.

The buffered and streaming HTTP disconnect cases both verify cancellation,
queued-request removal, lease release, and recovery. The streaming case exposed
a missing producer-lifetime channel observer, which was fixed by reusing the
existing connection-cancellation handler.

`MessagesDocumentTests` exercises system PDF text extraction, page boundaries,
encrypted documents, page limits, and citation-capability rejection.

`ClaudeCodeConnectionTests` covers preview defaults, local routing, effective
limits, atomic merge, idempotence, credential-helper conflicts, and preservation
of unrelated settings and MCP registrations.

The docs site builds successfully. Focused protocol, shared execution, command,
Glimmer parser, cache-policy, and context-edit tests pass, including the official
SDK against both API versions. The full `./scripts/check.sh` gate passed on
November 7, 2026, after unlocking macOS, with the official SDK environment and
installed Ornith and Glimmer tokenizer fixtures enabled. Earlier locked runs had 30 protected
run-record failures: files written with `completeUnlessOpen` could not be reopened
while `IOConsoleLocked = Yes`. Those cases passed in the unlocked run without
changing file protection. A green gate does not establish desktop task success.

## Model and task evidence

| Adapter | Protocol fixture | Native inference | Coding session | Calculator and website | Performance |
| --- | --- | --- | --- | --- | --- |
| Glimmer | Covered | Native count, cache warm/reuse, stream, typed tool and tool-error replay observed | Earlier fixture passed; independently verified | Calculator and MDN search passed twice after corrections | Optimized probe and app rounds below |
| Qwen | Text, counting, streams and typed tools | Ornith 35B text/count/stream/tool and signed replay probes passed | Fixture passed twice after encoding guidance; independently verified | AX-only Calculator and MDN search passed twice | Debug probe below |
| Gemma | Text, counting, streams and typed tools | Gemma Turbo text/count/stream/tool and signed replay probes passed | Fixture passed twice after effort omission; independently verified | AX first cases passed; fresh repeat failed | Observations below |
| Other text backends | Buffered route boundary only | Flint 4-bit text probe passed; count rejected explicitly | Unsupported adapter capabilities | Not qualified | Debug probe below |

The local Cua reference configuration was later located beside the llama.cpp
server; see the live Cua trials below. Calculator qualification requires
independent observations of
`6 × 7 = 42` and `47 × 23 = 1081`, repeated from fresh app state. Website and
coding tasks require independent evidence. The coding fixture below passed;
the live Cua evidence below records Calculator and website outcomes separately.

## Native Glimmer probe

A local debug build loaded `vision-chat-muse-glimmer-30b` with its installed
DFlash companion. Only that resident model was used for these requests. The
server reported 89.891 seconds for model loading. No queue-wait measurement was
collected; the process printed its admission marker before loading.

| Request | Wall time | Actual result |
| --- | --- | --- |
| Count prepared prompt | 0.044 s | 64 input tokens |
| First zero-output warm | 1.081 s | 62 cache-write tokens, 2 uncached tokens, zero output |
| Second zero-output warm | 0.226 s | 62 cache-read tokens, 2 uncached tokens, zero output |
| Adaptive-thinking stream | 11.789 s | Separate thinking and text; visible text `42`; 69 output tokens |
| Typed multiply tool | 10.057 s | `multiply` with numeric arguments `a: 6`, `b: 7` |
| Client tool-error replay | Observed | Signed reasoning and matching ID accepted; model requested a retry |

The stream's runtime telemetry reports 0.346 seconds prefill and 11.389 seconds
decode, or 6.06 output tokens/second. Cache-enabled decoding used the target
alone. These are one-off debug observations collected alongside unit tests,
not an isolated performance benchmark or a comparison with llama.cpp.

The tool probe exposed a recipient-header parsing defect. A regression fixture
checks every split of that observed protocol shape. A native repeat after the
fix returned the same numeric tool arguments without protocol headers in
reasoning, and accepted the signed tool-error replay. That repeated tool
request took 6.989 seconds. Neither probe ran Calculator or executed a coding
tool.

## Representative backend probes

The server unloaded each model before loading the next. Runtime snapshots
confirmed exactly one resident model for every generation below.

| Model | Load plus count | Prepared tokens | Generation wall time | Observed result |
| --- | --- | --- | --- | --- |
| `text-agent-ornith-35b-mlx-4bit` | 20.383 s | 17 | 10.442 s | `READY.`, 2 output tokens |
| `text-chat-gemma4-turbo` | 2.334 s | 18 | 18.897 s | `READY.`, 2 output tokens |
| `text-chat-flint-4bit` | Count rejected explicitly | Generation reported 7 input tokens | 39.263 s including load | `READY`, 23 output tokens |

Qwen and Gemma generation usage matched their independent count responses.
Before the streaming adapter was added, streaming requests returned explicit
errors. Qwen telemetry
separated 9.946 seconds prefill from 0.488 seconds decode; Gemma separated
18.432 seconds prefill from 0.351 seconds decode. Queue wait was not measured.
These short debug probes do not establish coding or desktop task competence.

After integrating typed native streaming, sequential SDK probes observed:

| Model | Load plus count | Text stream | Typed tool stream | Signed tool-error replay |
| --- | --- | --- | --- | --- |
| Ornith 35B | 19.766 s, 17 tokens | 9.548 s, `READY.` | 3.018 s, 56 output tokens | 2.160 s, retry call |
| Gemma Turbo | 3.577 s, 18 tokens | 19.599 s, `READY.` | 4.220 s, 97 output tokens | 1.937 s, retry call |

Both streams returned separate thinking and tool blocks with numeric `a: 6`,
`b: 7` arguments. Their text-stream input usage matched independent counts.
Runtime snapshots showed one resident checkpoint at a time. These probes did
not execute the tool or establish coding or desktop task success.

An exploratory Claude Code coding session used a temporary two-test Python
fixture and the standard client tool catalog. It reached two completed native
requests but timed out after 600 seconds without editing the fixture; independent
tests still failed. Its repeated 19,044-token prompt exceeded the original
8,192-token cache bound. Runtime telemetry reported 51.206 seconds average
prefill and 41.869 seconds average decode, with one subsequent request cancelled
and its admission lease released. This is a failed task probe, not successful
Claude Code qualification. The prefix-cache bound was increased to 32,768 tokens
for the repeat. A subsequent native zero-output probe wrote 19,021 prefix
tokens in 80.652 seconds and read those exact cached tokens in 1.891 seconds;
10 uncached tokens remained on both requests. Both requests generated zero
output tokens, and total input usage matched the separate 19,031-token count.
The count-only load did not warm a decoder.

A fresh exploratory repeat reached client-executed `Bash` and parallel `Read`
calls, then client-side compaction. It was interrupted before an edit to correct
initial SSE input usage, which had been zero until the terminal event. The
adapter now sends confirmed input/cache usage in `message_start`; out-of-band
pings keep prefill connections alive. HTTP and SDK fixtures cover the ordering.
No coding-task success is claimed for the interrupted repeat. A later run
confirmed nonzero initial usage but exposed inherited stdin in the exploratory
launcher: launcher code had been appended to the task. That run is invalid as
model-quality evidence. The corrected harness isolates stdin and stores evidence
outside the fixture. The clean 32k run confirmed 18,681 real cache-read tokens
on its second tool request, but again entered early client compaction after
file inspection. It was interrupted before editing. The installed Glimmer
configuration declares 131,072 positions; the next trial explicitly uses a
65,536-token server context and the corresponding client limit.

## Glimmer coding fixture

A fresh Claude Code `2.1.316` session passed the Python coding fixture with
`vision-chat-muse-glimmer-30b`, a 65,536-token context, and a 4,096-token output
limit. The connection command generated project-local settings with the actual
model ID. The harness isolated stdin, stored logs outside the fixture, and
allowed `Read`, `Edit`, and the requested unittest commands without bypassing
client permissions. Claude Code owned all file and shell operations.

The observed client calls inspected both files, reproduced the failing tests,
changed `return a + b` to `return a * b`, ran the deliberately missing test
module, and recovered with the correct test command. An independent process
then verified that both tests passed and that the test file was unchanged.
All seven checks passed: inspection, edit, baseline failure, tool error,
recovery, unchanged tests, and independent verification.

Cold model loading took 82.900 seconds separately from the coding session's
399.687 seconds. Seven inference rounds generated 1,249 tokens; runtime
telemetry reported 5.782 seconds average prefill, 48.580 seconds average decode,
and 3.673 output tokens/second. The final request used 20,271 input tokens,
1.442 seconds prefill, and 128.997 seconds decode for 427 output tokens.
The end snapshot had no queued or active requests and one resident model.
These debug timings are observations, not a controlled performance comparison.

This qualifies the coding fixture only. It does not qualify desktop tasks.

## Qwen and Gemma coding outcomes

Claude Code exposed a checkpoint-template restriction on mid-conversation system
turns in Ornith. The Qwen adapter preserves those turns as native ChatML system
messages at their original boundaries. Regression tests check ordering and
reject templates whose token prefixes cannot preserve a boundary. The complete
21-tool client catalog, including the `uri` schema format, counted 18,470 tokens
through the native preparation route after the correction.

Two fresh Qwen coding sessions then inspected the files and ran the baseline
tests but failed before editing. The model repeatedly emitted invalid JSON for
the boolean `Edit.replace_all` parameter. Each completed call failed validation;
the server sent an SSE error and released the lease. Independent tests still
failed, and the source remained unchanged. Session wall times were 60.192 and
59.299 seconds. The final runtime snapshot had no active or queued requests.
These are failed coding tasks, despite successful native protocol probes.

A diagnostic replay identified the rejected value as Python-style `False`.
Encoding guidance at the start of the large system prompt did not fix the
failure: another fresh session failed after 67.106 seconds. The Qwen adapter
then placed native parameter-encoding instructions immediately before the
generation boundary. The prepared-token path includes these instructions for
both generation and counting. It preserves the supplied conversation and keeps
strict output validation; `False`, quoted booleans, numeric substitutes, and
empty values remain invalid for boolean parameters.

Two fresh Claude Code sessions with this change passed all seven independent
fixture checks described for Glimmer. They read both files, reproduced failures,
edited the multiplication expression, exercised the missing-test error, and
recovered with passing tests. Independent processes confirmed both tests passed
and the test files remained unchanged. The sessions took 82.198 and 72.445
seconds. The first recovered from one rejected boolean call; the second had no
additional adapter-output failures. This establishes the fixture outcome, not
a guarantee that unconstrained model output always satisfies a tool schema.

Model loading was measured separately at 4.033 seconds. Across the two sessions,
17 completed native rounds generated 2,908 tokens; telemetry reported 2.049
seconds average prefill, 6.106 seconds average decode, and 28.013 output
tokens/second. Failed-attempt output is excluded from those completed-round
counters. Queue-wait time was not measured. The final snapshot had one resident
model and no queued or active requests. These debug observations are not a
controlled benchmark; the repeat overlapped repository validation startup.
These coding fixtures do not qualify Calculator or website tasks; see the later
live Cua evidence.

Gemma's session failed before generation because Claude Code `2.1.316` requested
`output_config.effort.high`. The Gemma adapter has no native effort control and
rejects that request explicitly. The connection preview warns about this
combination and recommends a local adapter advertising high effort or a client
configuration that omits the unsupported field. No Gemma coding success or
generation-performance measurement is claimed for this session.

A live `api connect claude-code --json` preview reported the installed Claude
Code version and Glimmer's effective 32,768-token context and 4,096-token output
limits. It did not create `.claude/settings.local.json`. Automated merge tests
cover writes, preservation, and idempotence.

## Advanced combinations

Combined official SDK tests cover both API versions and both response modes.
They combine cache accounting, documents inside tool results, signed thinking,
thinking/result clearing, deferred discovery, and client tool-error replay.
They also cover invalid-regex recovery, shared output-budget exhaustion, and
rejection of modified thinking even when that thinking would be cleared.
Context-management stream accumulation uses the SDK's beta interface, which
defines those response fields.

The combination tests exposed and fixed these defects:

- Removing thinking could move a cache marker past its original content boundary.
- Search continuation could report an already-cleared result as another edit.
- Tool markers before deferred definitions were ignored or covered later tools.
- Markers inside result content were ignored, and nested TTL order was reversed.
- Internal search history used plain text instead of typed references and error state.
- Stream fragments became separate history blocks, inserting newlines between tokens.
- Untyped adapter search output was classified as a client error instead of a server failure.
- Moving a marker earlier could reuse a previously cached prefix beyond the requested boundary.
- Search definitions could ignore unsupported caller/example/schema options and supplied descriptions.

Actual HTTP disconnect tests cancel buffered and streaming requests during an
internal search continuation. Both release the original lease, admit a queued
request, and serve a subsequent request successfully.

Sequential native SDK probes used one resident checkpoint at a time. The prior
tool exchange was scripted; generated searches and lookup calls were checked
against the actual catalog. No client tool was executed by the server.

| Adapter | Search | Prepared rounds | Reported total input | Output | Generation wall time |
| --- | --- | --- | --- | --- | --- |
| Glimmer | Regex | 560 + 902 | 1,462 including cache reads/writes | 316 | 68.820 s |
| Ornith 35B | BM25 | 538 + 687 | 1,225 | 140 | 5.728 s |
| Gemma Turbo | BM25 | 221 + 436 | 657 | 213 | 10.329 s |

Each case discovered `artifact_lookup`, returned its typed `key: "alpha"` call,
streamed separate signed thinking, and accepted replay with a PDF tool-error
result. Independently counting each internal continuation exactly reproduced
total reported input usage. Context edits were reported once for the request.
Model loading was measured separately: Glimmer 58.546 seconds, Ornith 19.594
seconds, and Gemma 2.606 seconds. These are debug observations, not controlled
performance comparisons. Queue-wait time was not measured; final snapshots had
no active or queued requests.

Glimmer also passed real native prefix checks:

| Combination | First write | Subsequent read | Result |
| --- | --- | --- | --- |
| Tool marker before later/deferred definitions | 190 tokens | 190 tokens | Later definition changed; zero output |
| Thinking clearing with a marker before later text | 487 tokens | 487 tokens | Later text changed; prepared input was 545 tokens after editing from 700 |
| Marker inside a PDF tool-error result | 679 tokens | 679 tokens | Later result text changed; zero output |

A moving-marker regression kept the prepared input identical at 555 tokens.
The first request cached 553 tokens. After the marker moved earlier, the backend
skipped that longer entry, wrote a 466-token prefix, and read only those 466
tokens on the repeat. Every warm generated zero output tokens. The reusable
harness's `--cache-only` selection checks this boundary and passed with real KV
state.

The PDF fixture had two pages independently extracted with macOS PDFKit.
Unsupported Ornith and Gemma cache backends reported zero reads/writes and
rejected zero-output warming explicitly.

`scripts/qualify-messages-native.py` makes the core native combination checks
repeatable against an already-running server. Its generated PDF was separately
verified with PDFKit. The harness passed on Glimmer with actual warming/reuse
and Gemma with regex search and explicit unsupported-cache handling. The Gemma
repeat counted 215 + 381 input tokens, matching usage of 596, and generated
144 tokens in 6.615 seconds. These direct Gemma API requests omit unsupported
effort; its Claude Code compatibility boundary remains unchanged.

Native Glimmer multi-image probes interleaved text with two independently
verified PNG labels, then swapped their order. At low effort with adaptive
thinking, the visible responses were `ALPHA` and `BETA`, matching the first
image in each request. Both prepared counts were 400 tokens and matched usage;
generation took 25.842 and 29.973 seconds for 32 and 38 output tokens.
Vision cache reads/writes stayed zero, and zero-output image warming failed
explicitly. The native harness includes an opt-in image-order check for supplied
labeled fixtures. These image cases do not qualify desktop interaction.

## Live Cua trials

The local reference setup uses unmodified Claude Code `2.1.316` and
[Cua Driver](https://github.com/trycua/cua) `0.38.0` over MCP. Its existing
13-tool configuration was reused unchanged, including the tool filter and
private-socket wrapper. The idle llama.cpp server was stopped before loading
native Glimmer. Driver permissions remained in standard mode; no permission
policy or existing MCP registration was changed. The client received scoped
approval for the supplied Cua tools and executed all app actions itself.

The first fresh Calculator trial was interrupted after 513.263 seconds without
computing `6 × 7`. It launched a new process, inspected its window, and cleared
the display. An independent accessibility read and screenshot both confirmed
`0` afterward. This is an incomplete task, not a pass. Cancellation released the
request lease, leaving no active or queued requests.

Cold native loading took 82.980 seconds separately from the client session.
The first image-bearing request prepared 26,551 tokens, spent 141.089 seconds in
prefill and 88.522 seconds decoding, and generated 146 tokens. Its cache reads
and writes were zero. A stack sample located the delay in native GPU prefill;
no queue wait was observed. These debug timings are not a controlled comparison
with llama.cpp.

The trial exposed a runtime gap: image-bearing prompts used one full prefill,
while text prompts used cancellable chunks. Both paths now use 512-token
language-model chunks, with image features embedded once in their original
positions. A tiny-model regression compares two-image chunking against whole
prefill across sliding-window boundaries, checking logits, captured layer
states, and cache offsets. The full repository gate passed again with this
change, including the official SDK tests.

A real 25,580-token image request cancelled during chunked prefill released its
lease in 2.039 seconds, with no active or queued work remaining. A subsequent
request using the same image and input returned `READY`: its reported input
usage exactly matched the independent count, with zero cache reads or writes.
Generation took 126.004 seconds, including 97.010 seconds prefill and 22.793
seconds decode for 31 output tokens. The long input differs from the live Cua
prompt, so these timings do not establish a controlled before/after speedup.
Native app repeats are required to assess task outcomes.

The fresh Calculator repeat on the chunked build reached the first digit, then
timed out at 900.093 seconds. Independent accessibility and screenshot evidence
confirmed a display of `6`, not `42`. Its first screenshot round used 26,713
input tokens, 134.698 seconds prefill, and 86.935 seconds decode for 136 output
tokens. No queue wait was observed. That trial did not qualify Glimmer for
Calculator. The server released the cancelled request and ended with no active
or queued work.

The repeatable short-image harness passed again on the chunked build: swapped
images produced `ALPHA` and `BETA`, with independently counted input usage of
400 tokens for each request and zero cache usage. The two generations took
32.065 and 29.807 seconds. Zero-output vision warming still returned an explicit
400 response. Short image ordering does not qualify a desktop task.

### Ornith accessibility cases

Ornith's current adapter declares no ordered-image input and accepts high effort.
Its Cua cases use the existing reference filter's documented `--text-only`
option: the model receives accessibility text, while an independent verifier
captures screenshots. The 13-tool MCP catalog and Cua implementation remain the
same. These results qualify accessibility-driven interaction, not native vision.
Claude Code and Cua execute all UI actions; the inference server executes none.

| Task | First fresh run | Second fresh run | Independent outcome |
| --- | --- | --- | --- |
| Calculator `6 × 7` | 152.819 s | 150.952 s | Both expression and display confirmed `6 × 7 = 42` |
| Calculator `47 × 23` | 155.757 s | 137.525 s | Both expression and display confirmed `47 × 23 = 1,081` |
| Visible MDN search for `Promise.all` | 104.992 s | 202.933 s | Both search-result clicks, final URLs/titles, and rejection explanations confirmed |

Every Calculator run launched a new process and cleared its state before entering
the expression. Independent accessibility reads and screenshots verified all
four results. Each Claude Code session finished normally. Both website runs used
separate temporary Chrome profiles, with the installed executable and version
checked before navigation. Chrome `157.0.8103.38` retained its normal sandbox.
The trusted launcher prepared each browser; Claude Code performed the visible
MDN search, opened its result, and read the reference page through Cua. It had
approval for the UI tools, with browser launching reserved to the trusted
fixture setup. Trials had a 900-second timeout. Initial runs allowed 20 client
turns; the final two allowed 40. No successful run exhausted its turn limit.

Website verification matched the clicked accessibility token to an observed
`Promise.all()` search result, then independently checked the final window's
address, title, page text, and screenshot. The isolated profile's browser history
also contained the MDN home page and the final reference URL in each run. Model-declared
success alone was not used. These session wall times include client and tool
work; they are not isolated inference benchmarks.

The six sessions recovered from six rejected model-generated calls: three
undeclared `run_steps` calls, two invalid boolean encodings, and one call missing
a required field. None was published as an executable tool call. Each error
terminated its stream with an API error; Claude Code continued and completed
the task. Every new session also recovered from the client's unsupported
`thinking.display.updates` option through its normal fallback. These task passes
do not guarantee schema-valid output from an unconstrained model.

Ornith loaded in 19.557 seconds before the sessions began. Across the six cases,
native telemetry recorded 80 completed inference rounds and 17,184 output tokens,
with 2.158 seconds average prefill, 6.983 seconds average decode, and 30.758 output
tokens/second. The six failed-output rounds are counted separately; their output
is excluded from completed-round performance counters. Queue-wait timing was not
collected; sampled snapshots showed no queue. The final snapshot had one resident
model and no active or queued requests. These are debug observations, not a
controlled backend comparison.

## Glimmer precision and client effort corrections

A BF16 tiny-model regression reproduced float32 promotion from image patches
through fused language embeddings into the attention cache. The patch projection
now uses its floating weight or quantization-scale dtype. Position interpolation
and rotary trigonometry run in float32 before conversion to the activation dtype.
Tests cover ordinary and quantized projections, large rotary positions, and
BF16 language/cache storage. Image/text ordering tests remain enabled.

Two optimized builds were compared using the same 25,580-token prepared prompt,
screenshot bytes, adaptive thinking, low effort, zero temperature, and 128-token
output limit. The baseline used the pre-correction Glimmer implementation from
`1d75734f`; the corrected build used `ef9627d0`. Both retained the same current
CLI. Runs were sequential with one resident model, independently counted input,
and no cache reads or writes. Each returned the visible word `READY`. Native
startup was measured separately: 81.754 seconds baseline and 80.736 corrected.

| Measurement | Baseline run 1 / 2 | Corrected run 1 / 2 |
| --- | --- | --- |
| Request wall time | 121.040 / 119.742 s | 42.071 / 44.348 s |
| Native prefill | 94.042 / 100.551 s | 36.655 / 40.388 s |
| Native decode | 20.549 / 18.999 s | 3.997 / 3.751 s |
| Output tokens | 32 / 32 | 33 / 33 |

Average request wall time improved by 2.79 times in this bounded probe. Output
lengths differed, and this experiment combines dtype correction with restored
assistant eligibility when cache markers produce no reusable checkpoint. It
does not isolate either change or establish desktop task success. Sampled
admission snapshots had no queued work; queue latency was not instrumented.

Glimmer also supports explicit text cache breakpoints before the first image.
No image-derived KV is retained. A tiny BF16 model test compares fresh and forked
text-prefix caches with two different future images and obtains identical
logits. The tokenizer fixture checks that retained boundaries precede image
expansion and that authentication scopes remain separate. At `948cbe3c`, only
markers before an image created native checkpoints. Image identities beyond the retained boundary do not affect
the earlier causal text state. These additions retain the existing bounded
cache, expiry, eviction, and changed-prefix rules.

The corrected native build warmed 3,037 text-prefix tokens in a 3,212-token
image prompt with zero output. Subsequent requests reused all 3,037 tokens while
changing the future image from `ALPHA` to `BETA`; the observed answers changed
accordingly. Both usage totals matched independent counts. Requests took 4.060
and 4.043 seconds, and warming after the image returned an explicit 400. The
checked-in native vision harness also passed both two-image orderings and
before-image text-prefix warming/reuse across changed images, using 406 cached
tokens in each 750-token prompt. Native loading took 69.050 seconds separately.

The first fresh Calculator trial on `948cbe3c` completed in 660.881 seconds
under the unchanged 900-second limit. Independent accessibility and screenshot
observations confirmed `6 × 7 = 42`; Claude Code exited successfully. The session
completed 12 inference rounds and 1,711 output tokens with no generation failures.
Native averages were 35.114 seconds prefill and 16.905 seconds decode, or 8.434
output tokens/second. Those counters exclude setup probes. Sequential requests
and sampled zero queue depth do not measure queue latency. This is one Calculator
pass, not the required repeated Calculator and website qualification.

The trace exposed a remaining cost: image-spanning client markers retained only
the separately marked 12,485-token tool prefix. Around 14,000–17,000 later tokens
still required prefill on each screenshot round. The subsequent cache correction
retains the causal text before the first image when a marker covers that image,
while leaving every image-derived KV state uncached. The tokenizer regression
covers that partial boundary, TTL selection, unmarked input, and moving a marker
earlier. The native vision harness passed on `be96b32d`, including both image
orderings, covering markers, and explicit text markers. A covering-marker warm
read 3,037 existing text tokens, wrote three additional text tokens, and left
172 tokens uncached in the independently counted 3,212-token prompt. No image
KV was retained. Native loading took 81.623 seconds separately.

The corrected build then passed four fresh Calculator trials using the same
Claude Code/Cua loop and 900-second timeout:

| Calculator expression | First fresh run | Second fresh run | Independent result |
| --- | --- | --- | --- |
| `6 × 7` | 301.905 s | 287.547 s | `42` in both runs |
| `47 × 23` | 318.511 s | 359.599 s | `1,081` in both runs |

Every run launched a new process, inspected an initial display of `0`, cleared
it, and pressed the observed digit, multiply, and equals controls. Independent
accessibility reads and captured images verified each expression and result.
The trusted verifier closed only that trial's Calculator process afterward.
All four clients exited successfully. There were 50 completed inference rounds,
6,880 output tokens, and no generation failures. Native averages were 7.333
seconds prefill and 15.345 seconds decode, with 8.967 output tokens/second.
These are task observations, not a controlled end-to-end comparison. Queue
latency was not instrumented; sampled queue depths were zero.

Both fresh website trials also passed, in 555.736 and 444.731 seconds. Each used
a separate Chrome profile and the unchanged 13-tool reference catalog. The
installed Chrome executable reported `157.0.8103.38`, matching the
[November 3 desktop stable release](https://chromereleases.googleblog.com/2026/11/stable-channel-update-for-desktop.html).
Its normal sandbox remained enabled. Claude Code selected MDN's visible search
control, entered `Promise.all`, and pressed an observed reference-page search
result. Independent window reads confirmed the final URL, title, and rejection
explanation. Captured images and the isolated profiles' histories corroborated
the page and navigation. The trusted verifier ended each owned session and
closed only that trial's Chrome instance.

The first website run recovered from two Cua keyboard-focus errors; Cua reported
that no key had been dispatched. Claude Code refreshed the search results and
used an accessibility link press. The second run used the search-result link
directly. These were client-tool failures, separate from inference or protocol
errors. Across all six app cases, native telemetry recorded 77 completed rounds,
11,241 output tokens, zero generation failures, 10.568 seconds average prefill,
16.185 seconds average decode, and 9.020 output tokens/second. Setup probes are
excluded. The session wall times include client and UI work and exclude model
loading and the trusted browser launch.

Gemma still advertises no effort control and rejects direct requests for effort.
The connection command now configures the documented
[`CLAUDE_CODE_EFFORT_LEVEL=auto`](https://code.claude.com/docs/en/model-config#send-no-effort-level)
setting for that capability. A wire fixture with unchanged Claude Code 2.1.316
observed the default `high` effort, then confirmed effort omission with adaptive
thinking preserved. Omission also held with an explicit CLI effort and an
inherited effort environment value. These are client protocol checks, separate
from native model task qualification. Connection tests cover conflicting caps,
repeat application, and switching back to an effort-capable model.

Native Gemma Turbo retained its empty effort capability list. Four explicit
high-effort requests, covering generation and counting under both API versions,
returned capability errors with HTTP 400 and request IDs. Generation counters
were unchanged. Native loading took 1.144 seconds separately.

With the generated project-local `auto` setting, two unchanged Claude Code coding
sessions completed in 39.983 and 25.121 seconds. Each inspected the fixture,
reproduced failing tests, read and edited the implementation, exercised
`python3 -m unittest missing_test` once, and recovered to passing tests. An
independent run passed both tests afterward, and the original test-file hash
was unchanged. These sessions used Gemma's effective 32,768-token context and
4,096-token output limit, with adaptive thinking preserved. Cache reads and
writes were zero.

Gemma's first AX-only Calculator cases passed: `6 × 7` in 111.699 seconds and
`47 × 23` in 119.413 seconds. Fresh-process action traces, accessibility reads,
and independent captures confirmed both results. Each recovered from two Cua
semantic errors involving conflicting target fields or stale window IDs.

The fresh `6 × 7` repeat failed after 443.267 seconds. An independent screenshot
and accessibility read showed an unfinished `6 × 77`; Claude Code exited with
an error after repeated tool-output validation failures. The server rejected
calls missing a required property before publishing them for execution. Native
telemetry recorded 11 failed generation rounds for this case. No active or
queued inference remained, and a fresh subsequent session was admitted normally.
A fresh subsequent `47 × 23` repeat finished its client session after 434.292
seconds, but independent observations showed `47 × 233 = 10,951`. It was not
counted as a task pass. The client performed three compaction requests and
continued afterward, exercising client-side conversation replacement. Native
telemetry recorded three rejected model-output rounds in that case. These
failures are separate from the resolved effort-field mismatch. Gemma's AX
reliability remains unqualified.

Gemma's first AX-only website trial passed in 254.437 seconds. Independent
observations confirmed the visible MDN search-result action, final URL/title,
page explanation, screenshot, and isolated browser history. The fresh repeat
failed after 499.304 seconds at the client's 40-turn limit. The independent
window read and screenshot showed MDN's home/search dialog with the duplicated
query `Promise.allPromise.all` and an incomplete address-bar entry, rather than
the reference page. Both trials used standard window-scoped Cua permissions.

| Gemma AX task | First fresh run | Second fresh run |
| --- | --- | --- |
| Calculator `6 × 7` | Pass, 111.699 s | Fail, 443.267 s; unfinished `6 × 77` |
| Calculator `47 × 23` | Pass, 119.413 s | Fail, 434.292 s; `47 × 233 = 10,951` |
| Visible MDN search | Pass, 254.437 s | Fail, 499.304 s; turn limit reached |

The effort-field correction is qualified by the native boundary probes and
coding fixtures. These mixed AX outcomes do not qualify Gemma for reliable Cua
operation. Use the recorded Glimmer image mode or Ornith AX mode for the
qualified app cases. Gemma's malformed calls were rejected before execution;
valid calls with stale IDs, conflicting targeting fields, or focus failures
were rejected by Cua and returned as client tool errors. The inference server
executed no desktop actions.

Across the six Gemma AX trials, native telemetry recorded 153 completed inference
rounds and 76,746 output tokens. The 16 rejected model-output rounds are excluded
from completed-round throughput. Native averages were 3.637 seconds prefill,
6.598 seconds decode, and 76.020 output tokens/second. Clients received 41 tool-error
results. Task outcomes are listed separately above. Queue latency was not
instrumented; sampled queue depths were zero.

## Remaining boundaries

Other backends without shared exact preparation reject token counting.
Strict tools, exact reasoning budgets, constrained JSON-schema output, generated
citations, vision PDF rendering, and server-side compaction are unsupported.
PDFs require extractable text; image-only PDFs fail explicitly. Image-derived
prefix caches are unavailable; Glimmer can retain marked text before an image.

The repository gate passes. Glimmer's six listed app cases and Ornith's AX cases
passed. Gemma's coding fixtures passed, but its AX reliability remains unqualified.
Unsupported capabilities are documented and rejected explicitly. The feature is not
qualified for unlisted models, modes, or desktop tasks.
