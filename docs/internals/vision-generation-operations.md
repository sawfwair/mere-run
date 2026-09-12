# Shared vision generation operations

This guide is for contributors changing reconstruction and video-depth execution.
Use the Core operation for each family when adding a CLI or API adapter.

The following operations own the shared request and execution decisions:

| Family | Core operation | CLI entry points |
| --- | --- | --- |
| TripoSR | `TripoSRGenerationOperation` | `image reconstruct-3d`, `vision image-to-3d` |
| InstantMesh | `InstantMeshGenerationOperation` | `image reconstruct-3d-multiview`, `vision image-to-3d-multiview` |
| Video Depth Anything | `VideoDepthAnythingGenerationOperation` | `vision depth-video` |
| Depth Anything 3 | `DepthAnything3GenerationOperation` | `vision geometry-multiview` |

For MoGe-2 token-grid planning and execution, see the
[shared single-image geometry operation](./geometry-generation-operation.md).

## Translate requests at the adapter boundary

CLI commands parse flags and camera documents, select output paths, and format
progress and results. The API parses multipart fields and retains its managed
model restrictions. Core requests preserve custom CLI model paths and the API's
managed model IDs; the operation does not expand either adapter's model policy.

Construct the family's immutable settings before execution. TripoSR settings
validate extraction resolution, density threshold, and foreground ratio.
InstantMesh settings validate extraction resolution and supplied camera rows;
preparation checks view and camera counts without changing view order. Video
depth settings resolve the bounded frame-count default through the native limits.
DA3 settings combine native view limits, camera conditioning, and the existing
throwing scene-export configuration.

## Preserve planning and input identity

Image preparation checks current headers and resource limits without loading
weights or creating output. DA3 also compares supplied camera dimensions with
the corresponding image. CLI dry-run continues to verify the checkpoint through
the family's resource resolver.

Video-depth dry-run uses the native bounded snapshot and decode admission. It
hashes the admitted video, validates decoded and network dimensions, verifies
the checkpoint, and removes temporary inputs before returning its plan.

Execution rechecks current input state. Each native generator retains its
immutable snapshots for decoding and provenance; a preceding plan does not
authorize later bytes. Video-depth execution performs its bounded admission
once, without first running a second full decode through dry-run.

## Keep resource and transport ownership explicit

Each operation owns one per-request runtime. It awaits unloading after successful
generation or an error, including scene-export failure. Cancellation is checked
before construction, at native stage boundaries, and before returning success.
DA3 scene export runs inside the operation, using the same validated export
configuration for CLI and API results.

The caller owns admission. HTTP upload cleanup, failed-output cleanup, local
artifact URLs, retention, and response formatting stay in the API handler.
Unload completes within the existing request slot; operations do not acquire
another permit or add a resident model cache.

## Validate behavior at each boundary

Test adapter settings, current-input changes, camera ordering and dimensions,
generation and export errors, cancellation, awaited unloading, and artifact
hashes. Compare representative native CLI and API outputs against the same
checkpoint and input before changing a family. Keep model identities, defaults,
output schemas, and geometry semantics explicit in those comparisons.

Unit lifecycle tests do not establish cancellation recovery during a GPU kernel
or actual HTTP disconnect behavior. Native artifact comparisons do not establish
perceptual quality or performance. A review MP4 can have different container
creation timestamps even when every decoded frame is identical.
