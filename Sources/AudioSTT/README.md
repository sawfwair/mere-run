# AudioSTT

AudioSTT resolves and executes speech-recognition requests over the owned
Parakeet, Qwen3-ASR, Whistle, Nemotron Transcribe, and SOV-Transcribe-1 runtime targets. AudioCore owns portable requests,
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
- `NemotronTranscribe/` owns BF16/local routed-expert INT4 loading and file or per-utterance live transcription/English
  translation over AudioNemotronTranscribeModel, with explicit installation.

Live utterance coordinators share explicit input and terminal phases.
Concurrent finishes join the same EOF flush. Cancellation can interrupt that
flush, closes the event stream with one cancelled final event, and waits for
processing, child-session creation, and native transcription to settle. A flush
failure closes the event stream with the error and remains observable to later
finish calls. A child session returned after cancellation is closed without
receiving audio.

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

## SOV-Transcribe ownership

`SOVTranscribe/` owns original BF16 checkpoint loading, FP32 router preservation,
typed processor checks, full-recording Whisper features, prompt expansion,
the verified Core ML encoder/projector bridge, resident profiling,
and file-transcription lifecycle. `AudioSOVTranscribeModel` owns audio and draft
computation. The optional draft is selected explicitly and loaded atomically
with the target. Translation and live transcription are rejected before loading.
The resident benchmark also exposes decoder expert policies. Q8 packs loaded
expert weights once and replaces their BF16 modules; FP32 routers remain intact.
The generator fixes this policy for its lifetime. Ordinary transcription uses
the original BF16 decoder unless an explicit packed-weight directory selects Q8.
`SOVPackedWeights` installs packed expert slots before streaming conversion, saves
versioned target shards transactionally, and reloads them directly on reuse.
The caller chooses storage, including external SSDs. Encoder/projector and draft
loading retain their original sources. Cache validation covers source identity,
precision, layout version, completeness, and shard metadata.
Q4/Q6 and calibrated per-layer Q4/Q6/Q8 recipes are benchmark-only policies.
Typed recipes are validated before weight loading, and packing occurs once
during preparation. The paired expert benchmark retains a second Q8 Metal
model and reports combined active memory; it does not repack between samples.
With DFlash selected, ordinary transcription pipelines GPU proposal IDs into
target verification and reuses verified next-token predictions. Reference
scheduling remains available in the resident benchmark. Draft-head Q8, shared
sliding KV storage, and allocator retention are separate opt-in experiments.
Checkpoint loading materializes file-backed MLX tensors in bounded batches,
using the original name mapping and dtype checks. This avoids an additional
Foundation `Data` copy of the complete checkpoint during preparation.

The benchmark-only projected-padding policy preserves every encoder input and
projector output, then optionally slices the final window's projected tail.
The released upstream processor retains full padded windows. Trimming therefore
changes decoder inputs and positions and stays off in ordinary transcription.
Per-request measurement records full/retained audio tokens and prompt length.

## Checks

Run `swift test --filter 'Whistle|ParakeetASRLiveSessionTests'` for focused numerical,
routing, and streaming coverage, then `./scripts/check.sh` for the repository gate.
See `docs/runtime/whistle.md` and the SpeechRuntimeTests fixture README for optional
original-checkpoint and packed-checkpoint parity inputs.
