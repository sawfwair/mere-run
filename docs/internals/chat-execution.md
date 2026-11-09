# Shared chat execution

`text chat`, `POST /v1/chat/completions`, and `POST /v1/messages` share request
resolution and native runtime invocation in `MereRunCore`. Start with `ChatRequestResolution.swift`,
then read `NativeChatRuntime.swift` and its `+Selection` and `+Diagnostics`
extensions.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `ChatRequestResolver` | Sampling defaults, shared numeric validation, and request-field preservation |
| `ChatContentBlock` and `ChatGenerationEvent` | Ordered content and semantic generation events, independent of HTTP protocols |
| `ChatPromptPreparation` | Exact adapter tokenization and context edits shared by Messages generation and counting |
| `NativeChatRuntime` | Native generator selection and invocation on macOS and Linux |
| `ChatGenerationOperation` | Request validation, cancellation checks, and runtime lifetime for one generation |
| CLI adapter | Tool authorization, media-input checks, and preflight reporting |
| API adapter | Wire-field validation, model capabilities, aliases, and stored server settings |
| `RuntimeServingServices` and `RuntimeChatSession` | Request admission and model residency for API requests |
| `APIServer.swift` | HTTP response formatting and streaming transport |

## Resolve and validate a request

`ChatRequestResolver` resolves each omitted sampling field independently. Its
`command` and `openAI` policies keep the existing entry-point defaults. For
example, an unrecognized model ID gets temperature `0.7` and top-p `0.9` under
the command policy, and `1.0` and `0.95` under the API policy. This fallback
doesn't make an unknown model available for execution.

Explicit sampling values override their defaults. JSON output suppresses
thinking output. The resolver preserves message media, tools, LoRA, stop
sequences, cache settings, reasoning budgets, and diagnostic options.

Both adapters apply the same numeric checks. These reject empty message lists,
invalid token or context limits, non-finite sampling values, out-of-range
probabilities and penalties, negative top-k values, and repetition penalties
that can't become a finite positive sampler value. CLI preflight reports these
failures as `text_chat_request_invalid`.

Before common resolution, the API adapter validates wire fields and model
capabilities, resolves aliases, and applies stored server settings.

The serving layer accepts `RuntimeChatRequest`, whose preparation operation
captures the typed wire request. Anthropic fields remain in CLI Support.
Messages adapters populate ordered content blocks; existing `ChatMessage`
initializers and stored-history decoding continue to work. Family adapters own
prompt rendering, media expansion, tool syntax, and completed-call validation.

Messages counting and generation consume the same prepared token sequence.
Context edits run before final context validation, and their reports use actual
token differences. Messages generation reserves its requested output budget
after edits; counting can inspect the same input without reserving generation
space. Tool-search continuations prepare the expanded catalog and edited
conversation again before decoding. Native caches match authenticated scope, loaded model,
template settings, media identity, and prepared token prefixes. Unsupported
cache backends report zero Messages cache usage.

Cache breakpoints follow their content when thinking blocks are removed or tool
results are replaced. Tool breakpoints identify the last included tool, so
discovering additional definitions does not extend an earlier marker. Prepared
tool results record clearing state to keep internal continuation edits idempotent.
Search continuations coalesce adjacent streamed fragments without inserting
separators, and retain typed tool references and error state in model history.

## Execution and runtime lifetime

`NativeChatRuntime` includes desktop-only GGUF and process-backed runtimes; the
iOS app selects its on-device generators directly. Commands use the family
selection rules, including the command-only Psi and Inkling runtimes. The API
selects its engine from the serving profile and passes its cache and batching
settings to the same runtime factory.

`ChatGenerationOperation.run` validates the effective request and checks
cancellation before and after generation. A CLI command keeps one runtime for
its whole tool conversation and awaits unload when the conversation completes,
fails, or is cancelled. Family generators own loading and inference.

Native Glimmer prefills text and image-bearing prompts in 512-token chunks.
Image features are embedded once in their ordered prompt positions before the
language-model chunks run. Each chunk checks cancellation and preserves captured
layer states for the speculative adapter. Image requests still report zero
prefix-cache reuse.

`RuntimeServingServices.startChat` acquires API request admission, resolves the
request, and acquires model residency. `RuntimeChatSession` holds both leases.
Response producers keep the session alive through native generation or upstream
proxy consumption. The session's completion task releases model residency
before request admission, and concurrent completion calls await the same
cleanup task.

Native streaming keeps its text, tool-call, usage, and completion events. The
macOS proxy stream forwards upstream bytes and cancels its producer when the
downstream consumer disconnects. The Linux proxy path buffers upstream data and
awaits cleanup before it returns the buffered response.

Messages SSE serializes typed text and reasoning events while withholding tool
inputs until their full validation succeeds. Local tool-search rounds retain
the same admission and model leases, cancellation, and shared output budget.
The server searches catalogs; the client executes coding and desktop tools.
See [Messages API](../runtime/messages-api.md) for adapter capabilities and
unsupported-feature errors.

## Validate a change

`ChatExecutionTests` exercises defaults, request-field preservation, invalid
requests, tool-conversation lifetime, and awaited cancellation cleanup.
`ChatExecutionIntegrationTests` exercises CLI and API resolution, preflight,
server settings and aliases, admission and residency cleanup, and proxy byte
streaming and disconnects.

These fixtures don't load model checkpoints. Use installed-model tests to
assess generation output and performance.
