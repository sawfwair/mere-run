# Shared vision generation operations

Use this map when you change 3D reconstruction or video-depth execution, or
when you add a CLI or API adapter for one of these families. Each family has
one Core operation that owns the shared request and execution decisions:

| Family | Core operation | CLI entry points |
| --- | --- | --- |
| TripoSR | `TripoSRGenerationOperation` | `image reconstruct-3d`, `vision image-to-3d` |
| InstantMesh | `InstantMeshGenerationOperation` | `image reconstruct-3d-multiview`, `vision image-to-3d-multiview` |
| Video Depth Anything | `VideoDepthAnythingGenerationOperation` | `vision depth-video` |
| Depth Anything 3 (DA3) | `DepthAnything3GenerationOperation` | `vision geometry-multiview` |

For MoGe-2 single-image geometry, see the
[shared single-image geometry operation](./geometry-generation-operation.md).

## Ownership

CLI commands parse flags and camera documents, choose output paths, and format
progress and results. The API parses multipart fields and keeps its managed
model restrictions. Core requests carry custom CLI model paths and the API's
managed model IDs unchanged; an operation never widens either adapter's model
policy.

The caller owns admission. HTTP upload cleanup, failed-output cleanup, local
artifact URLs, retention, and response formatting stay in the API handler.

## Plan and validate a request

Construct the family's immutable settings before execution:

- TripoSR settings validate extraction resolution, density threshold, and
  foreground ratio. Transparent-foreground preparation also bounds the padded
  canvas by the VFX image budget before it converts the size to an integer or
  allocates it.
- InstantMesh settings validate extraction resolution and supplied camera rows.
  Preparation checks view and camera counts without changing view order.
- Video-depth settings resolve the bounded frame-count default through the
  native limits.
- DA3 settings combine native view limits, camera conditioning, and the
  throwing scene-export configuration.

Image preparation checks current headers and resource limits without loading
weights or creating output. DA3 also compares supplied camera dimensions with
the matching image. CLI `--dry-run` still verifies the checkpoint through the
family's resource resolver.

Video-depth `--dry-run` uses the native bounded snapshot and decode admission.
It hashes the admitted video, validates decoded and network dimensions,
verifies the checkpoint, and removes temporary inputs before it returns its
plan.

## Execution and runtime lifetime

Execution rechecks the current input. Each native generator keeps its immutable
snapshots for decoding and provenance; an earlier plan doesn't authorize bytes
that change later. Video-depth execution performs its bounded admission once,
without first repeating the dry-run decode.

Each operation owns one runtime per request. It awaits unload after success or
an error, including a scene-export failure. Cancellation is checked before
construction, at native stage boundaries, and before success returns. DA3
scene export runs inside the operation with the same validated export
configuration for CLI and API results.

Unload completes within the existing request slot. Operations don't acquire
another permit or add a resident model cache.

## Validate a change

Test adapter settings, current-input changes, camera ordering and dimensions,
generation and export errors, cancellation, awaited unloading, and artifact
hashes. Before you change a family, compare representative native CLI and API
outputs against the same checkpoint and input. Keep model identities, defaults,
output schemas, and geometry semantics explicit in those comparisons.

Unit lifecycle tests don't establish cancellation recovery during a GPU kernel
or real HTTP disconnect behavior. Native artifact comparisons don't establish
perceptual quality or performance. A review MP4 can have different container
creation timestamps even when every decoded frame is identical.
