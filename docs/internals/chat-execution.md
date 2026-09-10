# Shared chat execution

`text chat` and `/v1/chat/completions` share request resolution and native runtime
invocation in `MereRunCore`. Start with `ChatRequestResolution.swift`, then read
`NativeChatRuntime.swift` and its selection and diagnostics extensions.

## Request resolution

`ChatRequestResolver` resolves each omitted sampling field independently. Its
`command` and `openAI` policies preserve the existing entry-point defaults. For
example, an unrecognized model ID receives temperature `0.7` and top-p `0.9`
under the command policy, and `1.0` and `0.95` under the API policy. This fallback
does not make an unknown model available for execution.

Explicit sampling values override their corresponding defaults. JSON output
suppresses thinking output. The resolver preserves message media, tools, LoRA,
stop sequences, cache settings, reasoning budgets, and diagnostic options.

Both adapters use the same numeric checks. These reject empty message lists,
invalid token or context limits, non-finite sampling values, out-of-range
probabilities and penalties, negative top-k values, and repetition penalties
that cannot be represented as a finite positive sampler value. CLI preflight
reports these failures through `text_chat_request_invalid`.

The API adapter still validates wire fields and model capabilities, resolves
aliases, and applies stored server settings before common resolution. CLI tool
authorization and media-input checks remain in the command adapter.

## Generation and cleanup

`NativeChatRuntime` selects and invokes native generators on macOS and Linux.
This command/server adapter includes desktop-only GGUF and process-backed
runtimes; the iOS app continues to select its on-device generators directly.
Commands use the existing family selection rules, including command-only Psi
and Inkling runtimes. The API selects its engine from the serving profile and passes its
cache and batching settings to the same runtime factory.

`ChatGenerationOperation.run` validates the effective request and checks
cancellation before and after generation. A CLI command retains one runtime
through its tool conversation and awaits unload when the conversation completes,
fails, or is cancelled. Family generators continue to own loading and inference.

`RuntimeServingServices.startChat` acquires API request admission, resolves the
request, and acquires model residency. `RuntimeChatSession` owns both leases.
Response producers retain the session through native generation or upstream
proxy consumption. Its completion task releases model residency before request
admission; concurrent completion calls await the same cleanup task.

HTTP response formatting remains in `APIServer.swift`. Native streaming retains
its text, tool-call, usage, and completion events. The macOS proxy stream forwards
upstream bytes and cancels its producer when the downstream consumer disconnects.
The Linux proxy path buffers upstream data and awaits cleanup before returning
that buffered response.

## Validation boundaries

`ChatExecutionTests` exercises defaults, request-field preservation, invalid
requests, tool-conversation lifetime, and awaited cancellation cleanup.
`ChatExecutionIntegrationTests` exercises CLI/API resolution, preflight, server
settings and aliases, admission and residency cleanup, and proxy byte streaming
and disconnects. These fixtures do not load model checkpoints. Use installed
model tests separately to assess generation output and performance.
