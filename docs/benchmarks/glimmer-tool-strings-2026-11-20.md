# Glimmer Q8 tool-string preservation

Strict Glimmer tools use a schema-constrained JSON envelope inside the native
`to=user` channel. Reasoning remains in `to=self`. This removes the mixed XML and
JSON encoding that produced incorrect nested string values in the
[earlier qualification](./glimmer-constrained-messages-2026-11-20.md).

The earlier failure returned `&lt;` and `&gt;` in a nested JSON string whose
requested value contained literal angle brackets. Extra encoding instructions
passed twice but exhausted the output budget on a repeat. Those failures remain
in the earlier record. This follow-up changes the representation instead of
HTML-decoding returned values or increasing the budget to conceal a failure.

## Representation and validation

All strict arguments use JSON types and escaping. A nested string containing
`</atem:parameter>` remains a string; it cannot close a parameter or create another
call. Literal entity text such as `&amp;` also remains unchanged. The internal
envelope is hidden from visible text, and calls become executable only after the
complete native turn passes schema validation. Automatic text answers stream
without exposing the envelope.

The shared schema compiler enforces named tool choice, parallel-call limits,
closed object arguments, nested schemas, nullable parameters, and local
references. The previous ambiguity between raw strings and JSON no longer
restricts nullable top-level parameters or JSON control-character escapes.
Counting and generation share the prepared schema instructions. Strict prefixes
use a versioned cache identity and boundaries derived from that prepared prompt.

Ordinary tools retain ATEM. Their parser preserves parameter values and names
without HTML-decoding them. This follows the released format's treatment of
strings as literal text and compound values as JSON, described in
[Meta's agentic fundamentals](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/agentic-fundamentals/README.md)
and the [vLLM Muse parser](https://docs.vllm.ai/en/latest/api/vllm/tool_parsers/muse_tool_parser/).

## Qualification method

The checkpoint is the established `q8-symmetric-bf16-g64` Glimmer artifact on an
Apple M5 Max with 128 GiB unified memory. Weights and inference kernels are
unchanged. Requests use adaptive thinking and high effort, with temperature 1.0,
top-p 0.95, and top-k 64. Version `2026-10-21` omits sampling fields and retains
the native recommended defaults. Constrained generation uses final-target
decoding; speculative decoding remains disabled.

Known-answer checks compare decoded argument values with independent expected
values after schema validation. Valid JSON with incorrect escaping fails. The
original failing prompt uses ordinary string schemas, without constants or enums
that reveal the expected values to the grammar. Repeats cover both API versions,
buffered responses, and official SDK stream accumulation. Counts are actual
prepared tokens, and usage must agree with counting. Requests run sequentially.

To repeat the literal-string suite against an already-running local server:

```sh
/path/to/sdk-environment/bin/python scripts/qualify-messages-constrained.py \
  --base-url http://127.0.0.1:18089 \
  --model q8-symmetric-bf16-g64 \
  --literal-tools-only --repeats 3 --max-tokens 4096 \
  --output /private/tmp/glimmer-tool-string-results
```

The command does not install models, start services, or execute returned tools.

## Automated validation

The full `./scripts/check.sh` gate passed with the official Anthropic SDK enabled:
6,386 XCTest cases, 585 skipped, zero failures, and 199 Swift Testing cases.
Both cache-boundary tests ran against the actual checkpoint tokenizer. Six
qualifier-scoring tests passed, including a schema-valid wrong-escaping result
that must fail and a check that the CLI returns a failing status to automation.

The HTTP fixtures exercise buffered and streamed nested arguments, token
counting, locally signed reasoning replay, and recovery from a tool-result error
for both API versions. The loaded synthetic Glimmer adapter and byte-split tests
cover the native generator, complete-turn validation, Unicode, and hidden
internal envelopes. These automated fixtures provide protocol evidence; the
real checkpoint trials provide separate task evidence.

## Original failing prompt

The JSON-envelope revision preserved the expected nested string in all 12
unforced repeats: three buffered and three streamed requests for each API
version. That revision's Release binary SHA-256 is
`6e3f12f3d2342523ce8439015ad84643496d54e9addf9d60b95be048b5953608`.
The schema supplied string types, without forcing the expected values.

The repeats took a median 134.82 seconds, ranging from 67.38 to 289.39 seconds,
and generated 933–2,325 output tokens. The earlier wrong-value trial took 187.96
seconds and generated 2,893 tokens. These are task observations with stochastic
sampling, not evidence of a kernel speedup. Another authenticated CLI server was
present; its inference activity was not inspected. Only this qualification's
server residency and sequential admission were verified, so these timings are
not an isolated host benchmark.

The initial 32-case matrix passed 31 cases. Every named/required tool-value
check passed, including literal entities, signed reasoning replay after a tool
error, nested arrays with local references and nullable values, parallel calls,
and both ordinary-ATEM entity trials. A separate two-case pilot also passed.

One automatic-choice trial under `2023-06-01` returned a tool call when the user
requested a plain answer. The `text` branch was available in the enforced schema;
the API had not forced tool choice. That failure is retained separately from
string preservation. The adapter's automatic-choice instructions explain that
`text` is unwrapped for the user and `tool_calls` requests client execution.

## Final revision

The final revision explains the automatic text branch without changing the
named/required tool instructions. Its Release binary SHA-256 is
`77978a98ab23befa9af6625bac5f3b3024e87b5bf1f5a076780794b4e9ac20e8`.

| Phase | Passed | Result |
| --- | ---: | --- |
| Initial pilot | 2/2 | Original literal prompt and entity text. |
| Initial matrix | 31/32 | One retained automatic-choice failure; all string checks passed. |
| Final selection and strings | 20/20 | Repeated automatic text, actual tool selection, and fresh original-string checks. |
| Final known-answer and typed tools | 28/28 | Both API versions; buffered and streamed requests. |
| Final strict-prefix cache | 3/3 | Zero-output warming, actual reuse, and schema-change invalidation. |

All 12 final automatic-text repeats returned the exact requested text with no
tool call: three buffered and three streamed for each API version. Four automatic
tool-selection checks generated the requested tool with the independently known
argument `1081`; the server did not execute that tool. Four fresh original
unforced-string trials also passed on the final binary. The earlier automatic
failure remains a failure in the initial matrix.

## Cache and coding checks

Zero-output warming wrote 225 actual prefix tokens and generated no tool call.
The next request read 225 cached tokens and preserved the exact arguments. Changing the schema read zero
cached tokens and also preserved the arguments. The marker requested a one-hour
TTL; this follow-up did not wait an hour to test expiration. Deterministic expiry
coverage and the earlier native expiration trials remain separate evidence.

Claude Code `2.1.316` completed a fresh coding fixture through
this native Q8 server. It inspected the files, fixed multiplication and a literal
HTML string, exercised the requested failing test command, and recovered. The
actual `Edit.new_string` and resulting source both preserved the five literal
characters `&amp;`. The test file's hash stayed unchanged, and an independent run
passed all five tests. The client exited with status zero after 141.53 seconds.
The owned server recorded nine native generation rounds and 1,644 generated
tokens. The client ran tools; the inference server handled generation and
protocol validation.

The [machine-readable results](./glimmer-tool-strings-2026-11-20-summary.json)
include per-case values, source and binary identities, usage, startup timing,
native prefill/decode timing, and private-trace hashes. Raw reasoning and
credentials are excluded from the public evidence.

## Qualification boundaries

These outcomes cover the recorded requests and exact Q8 artifact. Other
quantizations and real Qwen/Gemma constrained checkpoints remain unqualified by
this follow-up. Constrained speculative decoding remains disabled. The six
earlier ordinary-tool desktop trials remain in the initial record; this
follow-up does not qualify strict Cua tools. Unsupported combinations retain the
explicit errors described in the [Messages guide](../runtime/messages-api.md).

Both owned server revisions stopped after active and queued requests reached
zero, and their processes were awaited. The qualification listener closed.
Unrelated services were left untouched.
