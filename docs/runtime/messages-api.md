# Local Messages API

`mere.run api serve` exposes native Messages routes alongside its OpenAI routes.
Claude Code owns the agent loop and executes its tools. The server runs local
inference and local tool-catalog searches; it does not execute client shell,
MCP, browser, or desktop tools.

This integration has protocol-fixture coverage. Model task qualification is
separate; see [qualification evidence](../internals/messages-qualification.md).

## Endpoints

| Route | Purpose |
| --- | --- |
| `POST /v1/messages` | Buffered or content-block SSE generation |
| `POST /v1/messages/count_tokens` | Exact prepared prompt tokens without decoding output |
| `GET /v1/models` | OpenAI discovery with display names, effective limits, local capabilities, and pagination |
| `GET /v1/models/:id` | Individual served model metadata |
| `HEAD /api/hello` | Connection probe |

The Messages routes accept `?beta=true`. The `anthropic-version` header accepts
`2023-06-01` and `2026-10-21`; omission selects `2023-06-01`. The latter rejects
`temperature`, `top_p`, and `top_k`, assistant document blocks, and user text
citations, as specified by the [API migration guide](https://platform.claude.com/docs/en/api/migration-2026-10-21).

Use `x-api-key` or `Authorization: Bearer` with the server's configured key.
Both headers can carry the same key. Conflicting or duplicate credentials fail.
Non-loopback serving retains the existing API-key requirement.

Messages errors contain `type`, `error`, and `request_id`. Responses also include
`request-id`. Streaming errors terminate the SSE stream without a successful
terminal event. Messages streams never use the OpenAI completion sentinel.

## Adapter capabilities

Inspect each model's `local_capabilities` before sending advanced options.
Capability metadata describes implemented adapter behavior, not task success.

| Feature | Native Glimmer adapter | Qwen and Gemma adapters | Other native text adapters |
| --- | --- | --- | --- |
| Buffered text | Implemented | Implemented | Implemented |
| Messages SSE text and thinking | Implemented | Implemented | Explicit unsupported error |
| Exact text counting | Implemented | Implemented | Explicit unsupported error |
| Ordered multiple images | Implemented, including image-token expansion | Explicit unsupported error | Explicit unsupported error |
| Custom tools and typed arguments | Implemented | Implemented | Explicit unsupported error |
| Adaptive thinking and effort | Low, medium, high, xhigh | Adaptive; declared Qwen effort presets | Explicit unsupported error |
| Exact thinking-token budgets | Explicit unsupported error | Explicit unsupported error | Explicit unsupported error |
| Local tool search and deferred references | Regex and BM25 | Regex and BM25 | Explicit unsupported error |
| Tool-result and thinking clearing | Implemented | Implemented | Explicit unsupported error |
| Native prefix reuse | Bounded text prefixes | Zero Messages cache accounting | Zero Messages cache accounting |
| Text documents and text PDFs | Implemented | Implemented | Implemented |
| Strict tools, constrained JSON schema, generated citations | Explicit unsupported error | Explicit unsupported error | Explicit unsupported error |

Effort selects the model's coarse reasoning strength. It does not promise an
exact reasoning-token budget. Fixed-reasoning Qwen variants accept their single
`high` effort preset; other effort requests fail explicitly. Switchable Qwen and Gemma adapters also accept
`thinking.type: "disabled"`; fixed-reasoning adapters reject it explicitly.
Glimmer does not advertise a mode that disables
internal reasoning. Returned thinking uses `mere.v1.` continuity tokens,
authenticated with the installation key at `~/.mere/api/reasoning.key`. Replay
must preserve the exact thinking text and token. Foreign or modified tokens
fail explicitly.

Tool schemas retain nested JSON values and local references. Completed calls
are checked before the server releases executable tool blocks. Unsupported
strict enforcement fails before generation. Tool names and IDs exposed to the
client remain unchanged. `local_capabilities.tool_schema_subset` lists supported
schema keywords. It includes local references, nested objects and arrays, enums,
constant values, unions, schema alternatives, bounded value constraints, and
the `uri` string format used by client tools. Unsupported schema keywords and
formats fail before generation.

Regex search accepts patterns up to 200 characters and runs with bounded
matching time. BM25 accepts queries up to 500 characters. Catalogs support up to
10,000 deferred tools, results default to five, and explicit limits range from
1 through 10,000. Results have deterministic ordering. Internal rounds share
one admission lease, cancellation, and output budget.

Context clearing applies before counting and generation. Tool results retain
their matching IDs, exclusions, and recent results. Responses report applied
edits and tokens cleared. Cache markers remain attached to their content after
edits, and internal search rounds do not report the same clearing twice.
Markers inside tool-result content and on non-deferred tools retain their
boundaries when later content or tool definitions change.
Moving a marker earlier also limits reuse of existing cached entries.
Server-side compaction strategies fail explicitly;
client-side conversation replacement remains available.

Text prefix caches retain at most two entries and 32,768 tokens in total, with
five-minute or one-hour expiry, refreshed on actual reuse. Cache identity includes authentication scope,
resident model, template revision, adapter settings, and actual prepared tokens.
Changed tokens invalidate reuse at that boundary. Model unloading releases cache
storage through the residency policy. Glimmer can reuse marked text before the
first image, even when subsequent screenshots change. Image-derived KV is never
retained. A marker spanning an image retains only the preceding text, with
actual partial cache reads and writes reported in usage. Zero-output warming
of such a prompt warms only that text prefix.
Cache warming requires `max_tokens: 0` and a supported cache breakpoint.

Documents accept inline text and base64 text or PDF data. PDFKit extracts PDF
text on macOS, preserving page boundaries. The local limits are 20 MiB and 100
pages. Encrypted, unreadable, image-only, and incompatible PDFs fail explicitly.
Document identity, title, and context remain attached to the canonical input.
Remote document URLs, hosted web search, hosted code execution, remote MCP
hosting, Files, Batches, account services, and billing are outside this API.

Thinking progress updates (`display: "updates"`) are unsupported. Claude Code
2.1.316 retries without that display option; native thinking remains available.
The connection preview reports this boundary. The same client requests
`output_config.effort.high` by default. For adapters without effort control,
including Gemma, the connection command configures
`CLAUDE_CODE_EFFORT_LEVEL=auto` in the selected settings scope. Claude Code omits
the effort field and retains adaptive thinking. This uses the client's
[documented effort omission](https://code.claude.com/docs/en/model-config#send-no-effort-level).
Explicit effort requests to those adapters still fail before generation.
Effort caps in other client settings can force a level even with `auto`; remove
an incompatible cap or choose an adapter supporting effort.

Switching from this omission setting to an effort-capable model removes the
old model's `auto` override, restoring client effort selection. A `null` in the
connection preview's environment patch means that key will be removed. Existing
supported effort preferences are preserved.

Initial Messages usage includes confirmed input and cache accounting. During
prefill, out-of-band ping events can arrive before `message_start`. Text and
thinking stream after prefill; completed tool inputs are published only after
validation.

## Connect Claude Code

Start the server explicitly with an installed model:

```sh
mere.run api serve --engine text-chat-muse-glimmer \
  --model vision-chat-muse-glimmer-30b --context-size 65536
```

Then preview the client configuration:

```sh
mere.run api connect claude-code \
  --base-url http://127.0.0.1:8080 \
  --model vision-chat-muse-glimmer-30b
```

The preview warns about small contexts. The default tool catalog can cause
frequent client compaction at 32k. Use a larger context where the local model
supports it, then rerun the connection command to copy the actual server limit.

The default destination is `.claude/settings.local.json` in the working
project. Add `--write` to apply the merge. Use `--scope user` explicitly for
`~/.claude/settings.json`, or `--json` for a machine-readable preview. The
command preserves unrelated settings and MCP registrations, and refuses to
replace a different credential helper without an explicit settings edit.

The helper reads `MERERUN_API_KEY`; `--api-key-env NAME` selects another variable.
Set that variable in the environment used to launch Claude Code. The command
never writes or prints its value. Existing `ANTHROPIC_API_KEY` or
`ANTHROPIC_AUTH_TOKEN` settings must be resolved explicitly because they can
override the helper. Active Bedrock, Vertex, or Foundry provider selectors also
fail with instructions to remove the conflicting selector.

The qualification candidate is Claude Code **2.1.316**. Qualification applies to
the models, modes, and task cases in the qualification record.
Other installed versions are reported as unqualified. The connection command
does not install models, start services, alter permissions, impersonate Claude
model identities, or disable advanced features to hide adapter gaps.

For manual configuration, merge the following into
`.claude/settings.local.json`. Replace the example limits with the effective
values from `GET /v1/models/:id`; this example uses Glimmer's served limits.
Keep existing permission and MCP settings.

For an adapter with no effort control, add
`"CLAUDE_CODE_EFFORT_LEVEL": "auto"` to that project's `env` object. The generated
connection settings include it automatically for those adapters.

```json
{
  "model": "vision-chat-muse-glimmer-30b",
  "apiKeyHelper": "printenv MERERUN_API_KEY",
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8080",
    "ANTHROPIC_MODEL": "vision-chat-muse-glimmer-30b",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "vision-chat-muse-glimmer-30b",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "vision-chat-muse-glimmer-30b",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "vision-chat-muse-glimmer-30b",
    "ANTHROPIC_DEFAULT_FABLE_MODEL": "vision-chat-muse-glimmer-30b",
    "ANTHROPIC_SMALL_FAST_MODEL": "vision-chat-muse-glimmer-30b",
    "CLAUDE_CODE_SUBAGENT_MODEL": "vision-chat-muse-glimmer-30b",
    "ANTHROPIC_CUSTOM_MODEL_OPTION": "vision-chat-muse-glimmer-30b",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "65536",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096"
  }
}
```

All model inference stays local. Claude Code can independently make optional
network requests, including fast-mode checks, WebFetch safety checks, and
telemetry. See the [Claude Code gateway compatibility guide](https://code.claude.com/docs/en/llm-gateway-protocol)
for client behavior and optional network configuration.

## Qualify advanced combinations

With a capable local model already served and the official Anthropic Python SDK
installed in your test environment, run:

```sh
python scripts/qualify-messages-native.py \
  --base-url http://127.0.0.1:8080 \
  --model vision-chat-muse-glimmer-30b --search regex \
  --output /tmp/messages-native-qualification.json
```

The harness checks real prepared counts, cache warming or explicit rejection,
deferred discovery, streaming, context edits, a synthetic two-page PDF, and
signed reasoning replay. It compares usage with independently counted internal
rounds. Use `--search bm25` for that search variant. It reads credentials from
`MERERUN_API_KEY`, or the variable named by `--api-key-env`, and never includes
their values in the report. Without a value, it uses a placeholder credential for an
unkeyed loopback server; a keyed server rejects that credential.

`--cache-only` selects warming, reuse, and moving-marker checks, using a real
signed-thinking seed. Reports identify the selected checks; a passing report
qualifies that selection.

This opt-in check requires installed assets and makes inference requests. It
does not start services, install models, or execute client tools. The scripted
prior tool exchange and generated lookup request qualify API combinations;
coding and desktop task evidence remains separate. See the
[qualification record](../internals/messages-qualification.md) for tested SDK
versions, native results, and remaining gaps.

For ordered-image checks, supply two labeled PNG or JPEG fixtures with
`--vision-images first.png second.png --vision-labels ALPHA BETA`. The harness
swaps their order and checks each generated first-image label against your
fixture labels, exact input usage, the explicit image-cache boundary, and text
prefix reuse across changed images when the adapter supports caching.
Add `--vision-only` to run those checks. The label task requests low effort
where the adapter advertises it; it keeps adaptive thinking enabled.
