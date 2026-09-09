# Shared autoregressive decoding

This library owns device-side token sampling, pipelined decoding, incremental
text emission, and optional logprob diagnostics. It depends only on MLX and
MLXRandom. It does not load models or tokenize prompts.

- `GenerationConfig.swift` defines the sampling policy.
- `Sampling.swift` selects tokens and constructs exact policy distributions.
- `SamplingPenalties.swift` applies repetition, presence, and frequency penalties.
- `SamplingLogprobs.swift` measures raw and policy distributions through opt-in
  host readbacks. `ChatLogprobTypes.swift` defines the diagnostic payloads.
- `AutoregressiveDecodeTypes.swift` defines loop inputs and results.
- `AutoregressiveDecodeEngine.swift` schedules model callbacks and confirms tokens.
- `IncrementalTokenTextDecoder.swift` buffers incomplete UTF-8 text during streaming.

`MereRunCore` re-exports the public types and functions. Model runtimes provide
logits, forward callbacks, token decoding, and cancellation checks. They own
cache creation, MLX stream synchronization, resource cleanup, and request seeds.
The shared sampler retains its existing MLX random-state behavior.

## Invariants

Keep sampling tensors and next-token inputs on the MLX device. Confirm the
first token before opening the deeper pipeline. Preserve steady-state queueing
and avoid an unused forward at the final token budget boundary. EOS and a stop
callback can discard queued work.

`decode` includes a confirmed token before calling `shouldContinue`.
`decodeStateful` calls `didSampleToken`, then `shouldContinue`, then checks EOS
before appending the token. Keep these distinct callback contracts. Stateful
forward work remains one step ahead, including the final sample.

Throw cancellation and forward errors to the caller. The loop does not own or
release a model, stream, cache, or admission lease. Callers must clean up even
when a forward throws with a sample pending.

Preserve penalty history per request, prompt exclusions, ban masks, filtering
order, and exact top-p selection during logprob capture. Diagnostics remain
opt-in, and reasoning tokens retain redacted text. Capture is supported by
`decode`; `decodeStateful` retains its token-only result.

## Validation

Build `MereRunDecode` and `DecodeRuntimeTests` independently. The package
boundary guard rejects dependencies on Core, tokenizers, model families, and
application services. Run the fixture suite with
`swift test --filter DecodeRuntimeTests`, then the repository gate.
