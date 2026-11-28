# Native video-depth interchange

This module owns frame-accurate depth-sequence types, preprocessing, windowing,
and atomic artifact export for temporally consistent video depth.

The legacy and streaming exporters write schema-versioned manifests containing
the immutable input video's exact byte count and SHA-256, checkpoint identity,
frame timing, depth semantics, per-frame artifacts, and hashes. Streaming
export keeps bounded state while frames are produced. Shared preprocessing and
windowing reject invalid dimensions, overlap, and frame limits with typed
errors.

The current inference implementation lives in `VideoDepth/VDA`.
