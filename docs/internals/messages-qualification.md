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

A November 8 tensor-header audit established that the installed Glimmer target
used for the native and Cua trials in this record is the upstream **BF16**
checkpoint. All 1,436 target tensors are BF16; its config has no quantization
field and its installation manifest records `bf16`. The catalog's default Q4
display name does not identify that installed artifact. These task outcomes do
not qualify Q8 or Q4 conversions. Quantized candidates require separate evidence.

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
November 7 and 8, 2026, with macOS unlocked, the official SDK environment, and
installed Ornith and Glimmer tokenizer fixtures enabled. Earlier locked runs
had 30 protected run-record failures: files written with `completeUnlessOpen` could not be reopened
while `IOConsoleLocked = Yes`. Those cases passed in the unlocked run without
changing file protection. A green gate does not establish desktop task success.

## Model and task evidence

| Adapter | Protocol fixture | Native inference | Coding session | Calculator and website | Performance |
| --- | --- | --- | --- | --- | --- |
| Glimmer | Covered | Native count, cache warm/reuse, stream, typed tool and tool-error replay observed | Earlier fixture and two larger coding trials passed; independently verified | Integer Calculator and both MDN search tasks passed twice; extended decimal repeat timed out | Optimized probe and app rounds below |
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

## November 8 extended Glimmer qualification

The optimized binary from `be96b32d` was reused without runtime changes. These
trials used Claude Code 2.1.316 and Anthropic SDK 2.6.0. Desktop trials retained
the Cua 0.38.0 reference loop and unchanged 13-tool catalog. Claude Code tasks
used 65,536 context tokens and 4,096 output tokens, with adaptive thinking and
low effort. Each client task retained the 900-second timeout and 40-turn limit.
Requests ran sequentially with one resident model. Native loading took 80.638
seconds before the trials.

Two fresh coding projects expanded the earlier multiplication fixture into a
JSON Lines reader and a module that builds category summaries. Requirements
covered malformed line recovery, original line numbers, nested Unicode metadata,
validation order, keeping the first valid event for each ID, nonnegative integer
values excluding booleans, and deterministic sorting. Both clients read the
requirements, both source files, and the original tests, reproduced the
failures, and edited both modules. The available coding tools were Read, Edit,
and Bash. Each client exercised the error from a missing test module exactly
once and recovered by running the correct suite successfully.

| Fresh coding trial | Client wall time | Independent original tests | Independent holdout tests |
| --- | --- | --- | --- |
| First | 362.031 s | 10 passed | 3 passed |
| Second | 416.521 s | 10 passed | 3 passed |

The test and requirements hashes were unchanged in both projects. The holdouts
included a 500-record generator with duplicates and nested metadata, whitespace
strings and case-sensitive categories, and validation precedence. Source review
confirmed ordinary implementations of the requirements. Both sessions completed
nine native inference rounds, with 3,008 and 3,603 output tokens respectively,
and zero generation failures. Native average prefill was 2.377 and 2.307
seconds; average decode was 36.315 and 42.653 seconds, or 9.203 and 9.386 output
tokens per second. Client wall time includes tool and client work. Queue latency
was not instrumented; sampled queue depths were zero.

The current binary also passed the combined native harness under both API
versions. This harness used adaptive thinking and the adapter's default effort:

| API version | Local search | Harness checks | Harness wall time | Initial prepared tokens | Search request output budget / actual output |
| --- | --- | --- | --- | --- | --- |
| `2023-06-01` | Regex | 10 passed | 40.471 s | 555 | 512 / 251 |
| `2026-10-21` | BM25 | 10 passed | 42.471 s | 565 | 512 / 274 |

Each combination applied both context-edit strategies, warmed a native prefix
without output, reused that prefix, moved a cache marker earlier, streamed a
server-side catalog search, and returned a validated deferred client-tool call.
The initial cache writes and subsequent reads were 553 and 563 tokens. Moving
the marker earlier retained only 466 and 474 tokens. Actual generation usage
included the internal continuation prompts and matched independent preparation
counts. Signed thinking and a nested PDF tool-error result were accepted on
replay counting. These PDF checks exercised preparation and counting.

A separate real HTTP test disconnected a queued request while an image request
was generating. The queue cleared while the active lease remained held. Closing
the active stream then cleared all active and queued work in an observed 0.263
seconds. Neither cancelled request incremented completed-generation counters. A
fresh image stream returned `READY` in 1.975 seconds with 468 prepared input
tokens and 27 output tokens; usage matched the independent count. These cleanup
timings are observations from one cancellation sequence.

The extended decimal task used observed Calculator buttons to evaluate `(125.5 −
47.25) × 8`, reading the subtraction result before multiplying it.

| Fresh decimal trial | Outcome | Client wall time | Independent observation |
| --- | --- | --- | --- |
| First | Pass | 602.764 s | `78.25 × 8 = 626` |
| Second | Timeout | 900.071 s | Unfinished `125.5 − 47.25`; equals had not been pressed |

The first trial used 16 button actions and three window state reads. The trace
confirmed `78.25` after subtraction; independent accessibility and screenshot
checks confirmed the final expression and result. Calculator used spoken
accessibility labels for decimal values, while the captured image showed the
numeric expression. The raw observations were retained.

The fresh repeat used 12 button actions and 13 window state reads before the
unchanged timeout. Independent capture confirmed the unfinished calculation.
Neither trial had Cua tool errors or malformed tool calls. The repeat reused a
25,163-token text prefix, but uncached history reached 14,427 tokens in the last
completed round. This build retains marked text before the first image;
subsequent history containing images still requires prefill. Additional
full-window observations increased that work.

The first trial completed 23 native rounds and 3,136 output tokens. The repeat
completed 27 rounds and 3,456 output tokens before cancellation. Native average
prefill was 7.932 and 15.143 seconds; average decode was 14.960 and 14.077
seconds, or 9.114 and 9.093 output tokens per second. These counters cover
completed rounds. They describe different action sequences, rather than a
controlled comparison. Admission returned to zero active and queued work after
the timeout. The owned apps and Cua sessions were closed. The extended decimal
workflow remains unqualified because its fresh repeat failed.

Two direct document trials then read validation codes from a two-page text PDF
and repeated the request with the source pages reversed. Both returned the
correct codes in source page order. Each prompt contained 138 prepared tokens. A
cache marker with a `1h` TTL warmed 136 tokens with zero output; generation then
read all 136 tokens from that entry. Changing the document produced a fresh
write with zero cache reads during warming, and the answer changed to the new
page order. Generation took 13.703 and 14.417 seconds, with 129 and 136 output
tokens each with a 256-token output limit. Usage matched independent counts and
both streams carried request IDs. The TTL checks covered acceptance and
immediate reuse. These requests used an ordinary JSON text instruction without
citations or constrained-output enforcement.

The first linked-browser attempt was interrupted after 306.224 seconds because
of a launcher setup error. The trusted launcher had claimed the same public Cua
session name requested by the agent, so Cua refused two otherwise valid calls
from the agent's different transport. This attempt is excluded from model
outcomes. The corrected setup gives the trusted launcher a separate name and
ends that session before Claude Code starts. The agent owns and ends its own
session. The Claude Code loop and 13-tool catalog remain unchanged.

The fresh browser task searches MDN visibly for `URLSearchParams`, opens the
interface reference from the search results, and follows its `getAll()` link to
the method reference. Each trial uses an isolated Chrome profile. The installed
Chrome executable and version are checked before navigation, with the normal
sandbox enabled.

Both corrected fresh browser trials passed. Independent checks verified the
visible MDN search button and query, the interface search-result action, the
method-link action, the final URL and title, the return-value explanation, and
the captured page. After each owned Chrome process exited, its isolated history
confirmed visits to the homepage, interface reference, and method reference.
Both clients ended their own Cua sessions, and the trusted verifier closed only
the corresponding private Chrome instance.

| Fresh linked browser trial | Outcome | Client wall time | Cua tool errors | Native rounds / output tokens |
| --- | --- | --- | --- | --- |
| First | Pass | 684.436 s | 2 keyboard-focus refusals; recovered | 17 / 2,644 |
| Second | Pass | 609.913 s | 0 | 14 / 2,286 |

Cua reported that no key was dispatched for either refused Enter attempt in the
first trial. The client refreshed the search state and used the observed result
link. The second trial used the live results directly. Neither trial typed a
reference URL as a substitute for search and link navigation. Native average
prefill was 20.315 and 22.492 seconds; average decode was 16.876 and 17.999
seconds, or 9.216 and 9.072 output tokens per second. Both used Chrome
157.0.8103.38 with the normal sandbox and separate temporary profiles.

Across the six valid expanded tasks, five passed and one timed out. Native
telemetry recorded 99 completed inference rounds, 18,133 output tokens, and zero
generation failures. Average native prefill was 13.068 seconds and average
decode was 19.937 seconds, with 9.187 output tokens per second. These counters
exclude the API probes and invalid launcher diagnostic and cover completed
rounds only. Model loading and client wall times are reported separately above;
queue latency was not instrumented.

## Remaining boundaries

Other backends without shared exact preparation reject token counting.
Strict tools, exact reasoning budgets, constrained JSON-schema output, generated
citations, vision PDF rendering, and server-side compaction are unsupported.
PDFs require extractable text; image-only PDFs fail explicitly. Image-derived
prefix caches are unavailable; Glimmer can retain marked text before an image.

The repository gate passes. Glimmer's earlier six app cases, both linked-browser
trials, and Ornith's AX cases passed. The extended Glimmer decimal workflow
remains unqualified after a repeat timeout. Gemma's coding fixtures passed, but
its AX reliability remains unqualified.
Unsupported capabilities are documented and rejected explicitly. The feature is not
qualified for unlisted models, modes, or desktop tasks.

## Native Q8 and Q4 comparison on November 8

This comparison uses the 128 GB M5 Max on AC power, macOS 27.2 (26B123), Swift
6.4.2, and pinned mlx-swift revision
`1e077ba968b291e0237d1ab522b271d433fa1713`. Models run one at a time. The API
and desktop trials use the production Release CLI; raw probes use optimized
Release test code with the repository's required test-only compilation hooks.

The conversion comparison preserves the pinned upstream BF16 checkpoint and
creates separate selective affine/group-64 artifacts. The vision tower and token
embedding remain BF16. Q8 uses an exact-zero symmetric encoding with codes fitted
to the stored BF16 scale; Q4 retains the standard converter recipe. Both use the
installed BF16 DFlash2 companion. The conversion receipts include source and
output hashes. These artifacts do not replace the installed BF16 target.

The text diagnostic scores 16 uniformly spaced WikiText-2 test windows, each with
2,048 next-token predictions. Every candidate uses the same actual tokenizer,
token IDs, 128-token scoring blocks, and float32 log-softmax loss calculation.
The token-suite SHA-256 is
`4a7166944d17bfcd382c5fa77d8dd16c60c20b9b6b390cced54be8c55c7821d0`.
The predeclared gate requires both the paired perplexity ratio and its upper 95%
window-bootstrap bound to stay within 5% of BF16.

| Target | Tensor payload | Perplexity | Change from BF16 | 95% paired ratio interval | Text gate |
| --- | --- | --- | --- | --- | --- |
| BF16 | 59.55 GB | 5.9902 | Reference | Reference | Reference |
| Symmetric Q8 | 34.64 GB | 5.9926 | +0.04% | 0.99977–1.00108 | Pass |
| Standard Q4 | 21.35 GB | 6.2016 | +3.53% | 1.02949–1.04170 | Pass |

Q4 cleared the original numerical threshold, but its measurable quality loss is
not accepted for this setup. Further Q4 speed and task qualification was stopped;
Q4 is not selected. A perplexity increase is not an answer-error percentage.
Q8 is the candidate retained for task qualification.

Standard affine Q8 was rejected: perplexity increased 29.3%. Native quantized
prefill reproduced that regression, excluding the new dense-prefill adapter as
its cause. An FP16 Q8 experiment overflowed the feed-forward product at layer 46
and was also rejected. Neither failed candidate is selected for serving. The
converter retains BF16 activation range and selects symmetric Q8 by default
when passed `--bits 8`; the Q4 default is unchanged.

On the M5 Max, the accepted Q8 candidate's repeated 256-token coding-prompt sweep
measured 17.08–17.10 tokens/s serially and 39.19–40.20 tokens/s with three drafts.
Widths two, five, and seven were slower. Every tested width produced the same
256 greedy tokens in both repetitions. The width-three runs were thermally
nominal; some other rows were fair. This single-prompt agreement does not establish
bit-exact speculative equivalence for all inputs. These calibration sweeps disabled
the low-acceptance fallback to compare widths; they are separate from the
production-policy comparison below.

For a 7,173-token prompt, Q8 prefill measured 590.2–694.6 tokens/s serially and
576.4–577.6 tokens/s when also preparing DFlash context. Subsequent speculative
decode measured 29.0–29.6 tokens/s. Those four rows were thermally nominal; peak
active MLX allocation was 39.29 GB. These raw runtime probes exclude model load,
HTTP, admission, image preparation, and app interaction. They do not establish
end-to-end session speed or desktop success.

The matched production-policy comparison keeps the 40% acceptance threshold and
its two-round evaluation enabled. Each row has two 256-token greedy runs, a
2,048-token prefill chunk, and the same BF16 companion. No compiler work runs
during these measurements.

| Target | Prompt tokens | Serial decode (tokens/s) | Three-draft decode (tokens/s) |
| --- | --- | --- | --- |
| BF16 | 86 | 9.71–9.72 | 27.47–27.58 |
| Symmetric Q8 | 86 | 16.94–16.98 | 40.20–41.76 |
| BF16 | 7,173 | 9.15–9.47 | 9.08–9.12 |
| Symmetric Q8 | 7,173 | 16.47–16.57 | 15.63–15.82 |

On the longer prompt, both precisions accept only one of the first six proposals
and switch to serial decoding. The calibration sweep's 29 tokens/s therefore
does not describe this production-policy workload. The short-prompt runs remain
speculative and thermally nominal. Some longer rows reach the fair thermal state;
these sequential runs do not constitute an interleaved thermal comparison.

For the longer prompt, BF16 prefill measures 667.2–685.4 tokens/s serially and
587.5–755.4 when preparing DFlash context. Q8 measures 595.3–630.0 and
618.1–625.8 respectively. Quantization does not improve every prefill measurement.
Peak active MLX allocation is 67.40 GB for BF16 and 39.29 GB for Q8 on these text
probes. These allocations exclude a vision workload's additional activations.

All BF16 serial/speculative greedy sequences match in this comparison, as do
Q8's short-prompt sequences. Q8's longer-prompt sequences differ at token 53 in
both repetitions. Exact greedy equivalence is therefore not established for
Q8; the independent likelihood and task checks remain separate evidence.

Custom artifact paths use distinct served model IDs. Qualification requests must
use the exact selected ID and verify its `installPath` in runtime status. Using the
installed canonical Glimmer ID can select BF16 instead of a separate conversion.
Both discovery routes now obtain custom-model Messages capabilities from the
same runtime resolution used for generation; HTTP fixtures cover this behavior.

The likelihood sample is a bounded English-text diagnostic. Recipe development
included inspecting that diagnostic; its confidence interval is not a claim of
unchanged general capability. Coding, image, tool, and desktop outcomes require
separate matched evidence.

### Q8 protocol and task checks

The fixed 20-case official-SDK suite passes 19 cases for both Q8 and the matched
BF16 baseline. All 20 have exact prepared token-count parity for each precision.
The same failed wording case returns `first\nthen second` instead
of `first\nsecond`; it remains recorded as a failure. A separate explicit
12-character newline fixture passes for both precisions. Its result does not
replace the original
case. These checks cover arithmetic, Unicode, nested JSON types, aggregation,
instruction/data separation, a tool schema containing nested references, and
both API versions.

Both 10-case advanced suites pass: version `2023-06-01` with regex search and
version `2026-10-21` with BM25. The native image suite passes 11 checks, including
image order, changed-image identity, actual prefix reuse, and zero-output
warming. A repeat of the image suite captures an actual HTTP request ID on all
11 responses, including accumulated SDK streams.

The two-page PDF checks preserve page order, reject stale document identity
after reversal, and report real immediate reuse with the accepted one-hour TTL.
Active and queued disconnect recovery pass. A separate 18,786-token request is
disconnected after telemetry reports its first 2,048-token prefill chunk and
before any generated token. Its lease releases in 3.21 seconds; the next request
returns `READY` in 2.41 seconds.

Both fresh coding sessions pass the unchanged public tests, three independent
hidden tests, and manual source review. Each reads the required files, repairs
both modules, and recovers from the deliberate missing-test error. The first
session takes 209.97 seconds across 10 native rounds and measures 18.51 decode
tokens/s. The repeat takes 362.44 seconds, generates more tokens, and overlaps CPU
repository validation; its timing is not an uncontended speed comparison.

All eight Q8 end-to-end trials pass independent verification:

| Task | First fresh run | Second fresh run | Independent evidence |
| --- | --- | --- | --- |
| Coding fixture | 209.97 s | 362.44 s | Public and hidden tests, source review, tool-error recovery |
| Calculator `6 × 7` | 189.61 s | 190.56 s | Initial zero, expression, display `42`, screenshot, process identity |
| Calculator `47 × 23` | 204.63 s | 198.85 s | Initial zero, expression, display `1,081`, screenshot, process identity |
| MDN interface-to-method search | 351.90 s | 365.66 s | Visible search, interface and method clicks, URL, page text, screenshot, browser history |

Each desktop run starts with a fresh app process or Chrome profile. The website
runs search for `URLSearchParams`, open its interface page, and follow the
`getAll()` link. Both identify the returned array and empty-array behavior. One
focus error per website run is recovered without changing permissions or the
reference loop. The trusted verifier closes only owned test instances afterward.

Native decode across these eight sessions measures 14.92–18.51 tokens/s. The
short greedy probe's roughly 40 tokens/s is not representative of every Claude
Code or Cua round. Session times include inference, client work, tools, and app
interaction. Queue wait is not separately instrumented; clients are sequential
and sampled queue depth is zero. The earlier BF16 app runs predate the paired
prefix-cache changes, so their speed difference cannot be attributed solely to
quantization.

Startup warmup is now performed before readiness for custom and managed Glimmer
IDs. The Q8 native probe measures 9.10 seconds of loading and 41.76 seconds of
warmup. The first real request then reports 0.00003 seconds of load time, starts
generating in 0.34 seconds, and completes in 2.09 seconds. Native graph preparation
is reported in the warmup load stage, rather than hidden inside a user's first
request. This startup change does not alter model weights or generation policy.
