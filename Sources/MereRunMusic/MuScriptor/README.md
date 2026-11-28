# MuScriptor

Native MLX inference for the open MuScriptor audio-to-MIDI checkpoints from
Kyutai and Mirelo.

The runtime consumes mono 16 kHz audio in five-second chunks, reproduces the
published 512-bin HTK mel frontend, prepends the learned dataset/instrument
conditioning tokens, and autoregressively decodes the MT3 event vocabulary.
The decoder carries tied notes across chunk boundaries and emits either note
events or a Standard MIDI File with one track per detected instrument.

For MIDI output, `MuScriptorMusicalContextAnalyzer` adds the score-level
metadata that is intentionally outside the checkpoint output vocabulary. It
combines audio spectral flux with decoded note onsets to estimate tempo, beat
phase, and meter, and applies duration-weighted pitch-class profiles for key.
The MIDI writer embeds standard tempo, time-signature, and key-signature meta
events while retaining the source-relative note timing. Low-evidence fields
remain absent, and the CLI can export confidence scores and beat positions as
JSON.

`chunkBatchSize` is a ceiling on independent five-second chunks, not a forced
allocation. After loading the model, Apple Silicon inference selects an
effective group size from current MLX active/cache headroom, a system reserve,
beam width, and model complexity. The complexity budget limits cross-chunk
grouping; a single chunk always retains the requested beam width while
oversized live-beam forwards are microbatched to the same lane budget. Float32
inference doubles the 16-bit lane-memory estimate. Other hosts skip the
unified-memory clamp when no applicable profile is available. `chunkBatchSize:
1` always keeps the single-chunk path. Persistent beam cache state still scales
with the requested search width, so the grouping policy does not claim that a
beam which cannot fit by itself will be admitted safely.

Published weights are gated on Hugging Face and licensed CC BY-NC 4.0. Users
must accept the model terms and configure a Hugging Face token before pulling.
