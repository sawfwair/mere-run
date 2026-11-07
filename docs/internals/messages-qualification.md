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
SDK against both API versions. The full repository gate was attempted repeatedly;
30 existing protected run-record test cases fail because macOS reports
`IOConsoleLocked = Yes`. Files written with `completeUnlessOpen` cannot be
reopened in that state. The gate is not green, and this is not a release-qualified
change.

## Model and task evidence

| Adapter | Protocol fixture | Native inference | Coding session | Calculator and website | Performance |
| --- | --- | --- | --- | --- | --- |
| Glimmer | Covered | Native count, cache warm/reuse, stream, typed tool and tool-error replay observed | Fixture passed; independently verified | Not qualified | Debug probe below |
| Qwen | Text, counting, streams and typed tools | Ornith 35B text/count/stream/tool and signed replay probes passed | Fixture passed twice after encoding guidance; independently verified | Not qualified | Debug probe below |
| Gemma | Text, counting, streams and typed tools | Gemma Turbo text/count/stream/tool and signed replay probes passed | Client effort request rejected before generation | Not qualified | Debug probe below |
| Other text backends | Buffered route boundary only | Flint 4-bit text probe passed; count rejected explicitly | Unsupported adapter capabilities | Not qualified | Debug probe below |

The unchanged Claude Code/Cua reference loop was not found in this repository.
Its location is needed to repeat the same harness with only the inference
server changed. Calculator qualification requires independent observations of
`6 × 7 = 42` and `47 × 23 = 1081`, repeated from fresh app state. Website and
coding tasks require independent evidence. The coding fixture below passed;
the Calculator, website, and unchanged reference-loop cases remain unqualified.

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

This qualifies the coding fixture only. It does not qualify desktop tasks or
the unavailable unchanged Claude Code/Cua reference harness.

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
Calculator, website, and unchanged reference-harness cases remain unqualified.

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

## Remaining boundaries

Other backends without shared exact preparation reject token counting.
Strict tools, exact reasoning budgets, constrained JSON-schema output, generated
citations, vision PDF rendering, and server-side compaction are unsupported.
PDFs require extractable text; image-only PDFs fail explicitly. Native vision
prefix caches are unavailable and report zero usage.

The repository gate and task qualification remain incomplete. Unsupported
capabilities are documented and rejected explicitly. The feature is not
presented as end-to-end Claude Code support.
