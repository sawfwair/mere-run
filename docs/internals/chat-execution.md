# Shared chat execution

`text chat` and `POST /v1/chat/completions` share request resolution and native
runtime invocation in `MereRunCore`. Start with `ChatRequestResolution.swift`,
then read `NativeChatRuntime.swift` and its `+Selection` and `+Diagnostics`
extensions.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `ChatRequestResolver` | Sampling defaults, shared numeric validation, and request-field preservation |
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

## Validate a change

`ChatExecutionTests` exercises defaults, request-field preservation, invalid
requests, tool-conversation lifetime, and awaited cancellation cleanup.
`ChatExecutionIntegrationTests` exercises CLI and API resolution, preflight,
server settings and aliases, admission and residency cleanup, and proxy byte
streaming and disconnects.

These fixtures don't load model checkpoints. Use installed-model tests to
assess generation output and performance.
