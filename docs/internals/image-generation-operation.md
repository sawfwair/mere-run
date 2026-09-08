# Shared image operation

CLI image generation, its preflight report, and the API image endpoints share
one Core operation. Model-family runtimes still own numerical inference and
checkpoint loading.

## Ownership

| Component | Responsibility |
| --- | --- |
| `ImageGenerationOptions` | Typed user settings before defaults and adapter recipes apply. |
| `ImageGenerationModelSelection` | Classify local paths and managed IDs; resolve installed model roots without downloading. |
| `ImageGenerationPlan` | Validate options and files, resolve adapters and effective sampling, and build `GenerationRequest`. |
| `ImageGenerationOperation` | Prepare edit inputs, invoke an executor, restore protected pixels, clean up temporary files, and report a typed outcome. |
| `ImageRunSession` | Own an optional versioned record, process lease, input snapshots, and verified output copy. |
| CLI adapter | Parse arguments, expand structured prompts, present progress and installation guidance, and emit existing receipts and run-plan events. |
| API adapter | Decode HTTP requests, choose API v1 compatibility settings, and provide the resident-pool executor. |
| Runtime | Check checkpoint integrity, load weights, run inference, and write the generated image. |

`ImageGenerationSampling` owns manifest defaults and the FLUX.2 Turbo adapter
recipe. `ImageGenerationConditioning` owns input/reference interpretation. Both
preflight and execution use these definitions. The API selects explicit policy
values to preserve its existing Klein and legacy Qwen defaults; see
[API image compatibility](../runtime/api-server.md#openai-imageaudio-compatibility).

## Preflight and execution

Resolving a plan reads metadata and checks input and adapter paths. It does not
create output directories, prepare masks, reserve an inference permit, download
models, or load weights. Preflight presents these results through its existing
versioned report, including actionable installation and file diagnostics.

Execution rechecks mutable input and adapter availability. It scopes temporary
mask and outpaint files to the operation and cleans them up after success,
failure, or cancellation. A caller can inject an executor for a resident runtime
or for tests. The executor owns admission and residency: the shared operation
never acquires another machine permit or unloads a borrowed pool member. Its
default executor creates and releases the native generator for one operation.

Structured prompt expansion remains a CLI preparation step. The image options
are validated before expansion, and the final expanded prompt is resolved into
the execution request. API routing, authentication, request limits, and response
encoding remain at the API boundary.

## Typed results

Swift callers can observe `ImageGenerationEvent`: started, progress, succeeded,
failed, or cancelled. An execution emits one terminal event. Its success outcome
contains the run ID, backend, model ID, image result, and effective request with
the actual seed. Durable input paths remain in that request after temporary edit
files are removed.

These Swift events do not change a CLI wire format. Existing `--receipt`,
`--progress-json`, preflight JSON, run-plan files, and API response shapes retain
their compatibility behavior. Optional `ImageRunSession` recording writes
`image-run.json` atomically. Its held, non-inherited file lock distinguishes
active work from abandoned records after termination or reboot. Inspection can
persist an interrupted state without relying on process IDs or timestamps.
An uncatchable process termination cannot emit a terminal Swift event.

Recording snapshots input images and masks, fingerprints local adapters, and
saves resolved options and the seed before calling the executor. Successful
runs retain a fingerprinted output copy. Retry verifies inputs and the installed
manifest, preserves the API compatibility policy, and starts a new sibling run
with a parent ID. It uses the same machine admission class as image generation.
It does not re-expand structured prompts or resume runtime state. See
[Record and retry an image run](../runtime/image.md#record-and-retry-an-image-run)
for retention and replay limits.

## Validation

`ImageGenerationOperationTests` exercises shared defaults and adapter recipes,
unsupported options, pixel restoration, failure, cancellation, temporary-file
cleanup, and input changes between planning and execution. It injects an
executor and needs no model weights. `ImageGenerationAdapterTests` compares
explicit CLI and API requests across all supported image backends, checks API
compatibility policies, and compares preflight with execution validation.

`ImageRunRecordTests` checks atomic state snapshots, active and abandoned leases,
process termination, input retention, retry validation, and terminal outcomes.
`ImageRunCommandTests` checks opt-in flags, observational preflight, run-plan
round trips, local inspection, legacy JSON compatibility, and admission classes.

These tests establish orchestration behavior. Checkpoint, numerical, GPU-memory,
and platform acceptance still use the existing family tests and runtime gates.
