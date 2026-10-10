# AudioSTT

AudioSTT resolves and executes speech-recognition requests over the owned
Parakeet, Qwen3-ASR, and Whistle runtime targets. AudioCore owns portable requests,
results, routing policy, durable operation contracts, and live event types;
AudioCodecs owns file decoding and resampling. Neural arithmetic stays in the
runtime-family targets.

## Entry points

- `SpeechTranscriptionResolver` selects a backend, validates task/language/provider
  combinations, resolves managed/local identity, and builds the execution plan.
- `NativeSpeechTranscriptionExecutor` dispatches the shared operation to generators.
- `Parakeet/`, `Qwen3ASR/`, and `Whistle/` own model preparation, waveform features,
  decoding coordination, and conversion into AudioCore results.
- `ASRUtteranceLiveSession` coordinates VAD, bounded audio queues, partials, commits,
  cancellation, and final events for resident file-oriented models. Parakeet's
  adapter retains overlapped-tail decoding; Whistle re-decodes whole utterances.
- Qwen3-ASR owns its distinct live decoder/session implementation.

## Whistle ownership and lifecycle

`WhistleGenerator` is an actor. `prepare` resolves assets and then installs its
model, vocabulary, and typed decoding controls. `transcribePrepared` uses leased task-local CPU/GPU streams and can be reused
by the live coordinator. Neural graph construction and evaluation run without
further suspension after entering the stream scope. Packed weight matrices,
KV state, and Engram lookups are owned by AudioWhistleModel. AudioSTT computes
16 kHz log-mel features, applies overlapping file windows, estimates word times
from decoder cross-attention DTW, and merges window results.

Whistle live clients disable word alignment because live events carry utterance
times. Keep each utterance within 30 seconds. Cancellation must discard decoded
results that return after cancellation, including during an EOF flush. The live
queue rejects excess audio rather than growing indefinitely.

## Checks

Run `swift test --filter 'Whistle|ParakeetASRLiveSessionTests'` for focused numerical,
routing, and streaming coverage, then `./scripts/check.sh` for the repository gate.
See `docs/runtime/whistle.md` and the SpeechRuntimeTests fixture README for optional
original-checkpoint and packed-checkpoint parity inputs.
