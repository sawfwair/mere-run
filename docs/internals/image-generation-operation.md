# Shared image generation operation

CLI image generation, its preflight report, and the API image endpoints share
one Core operation. Model-family runtimes still own checkpoint loading and
numerical inference.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `ImageGenerationOptions` | Typed user settings before defaults and adapter recipes apply |
| `ImageGenerationModelSelection` | Classify local paths and managed IDs, and resolve installed model roots without downloading |
| `ImageGenerationPlan` | Validate options and files, resolve adapters and effective sampling, and build `GenerationRequest` |
| `ImageGenerationOperation` | Prepare edit inputs, invoke an executor, restore protected pixels, clean up temporary files, and report a typed outcome |
| `ImageRunSession` | Own an optional versioned record, process lease, input snapshots, and verified output copy |
| CLI adapter | Parse arguments, expand structured prompts, present progress and installation guidance, and emit receipts and run-plan events |
| API adapter | Decode HTTP requests, choose API v1 compatibility settings, and provide the resident-pool executor |
| Runtime | Check checkpoint integrity, load weights, run inference, and write the generated image |

`ImageGenerationSampling` owns manifest defaults and the FLUX.2 Turbo adapter
recipe. `ImageGenerationConditioning` owns input and reference interpretation.
Preflight and execution both use these definitions. The API passes explicit
policy values to keep its Klein and legacy Qwen defaults; see
[API image compatibility](../runtime/api-server.md#openai-image-audio-compatibility).

## Plan and validate a request

Resolving a plan reads metadata and checks input and adapter paths. It doesn't
create output directories, prepare masks, reserve an inference permit, download
models, or load weights. Preflight presents the results in its versioned
report, including installation and file diagnostics.

Structured prompt expansion is a CLI preparation step. The CLI validates the
image options before expansion, then resolves the expanded prompt into the
execution request. API routing, authentication, request limits, and response
encoding stay at the API boundary.

## Execution and runtime lifetime

Execution rechecks mutable input and adapter availability. It scopes temporary
mask and outpaint files to the operation and removes them after success,
failure, or cancellation.

A caller can inject an executor for a resident runtime or for tests. The
executor owns admission and residency: the shared operation never acquires
another machine permit or unloads a borrowed pool member. The default executor
creates and releases the native generator for one operation.

## Results and run records

Swift callers can observe `ImageGenerationEvent`: started, progress,
succeeded, failed, or cancelled. Each execution emits one terminal event. The
success outcome holds the run ID, backend, model ID, image result, and the
effective request with the actual seed. Durable input paths stay in that
request after temporary edit files are removed. An uncatchable process
termination can't emit a terminal event.

These Swift events don't change any CLI wire format. `--receipt`,
`--progress-json`, preflight JSON, run-plan files, and API response shapes keep
their compatibility behavior.

Optional `ImageRunSession` recording writes `image-run.json` atomically, using
`MereRunExecution` storage primitives without changing the image schema. A
held, non-inherited file lock tells active work apart from records abandoned by
termination or reboot, so inspection can record an interrupted state without
relying on process IDs or timestamps.

Before calling the executor, recording snapshots input images and masks,
fingerprints local adapters, and saves resolved options and the seed. A
successful run keeps a fingerprinted output copy. Retry verifies inputs and the
installed manifest, keeps the API compatibility policy, and starts a new
sibling run with a parent ID under the same machine admission class. It doesn't
re-expand structured prompts or resume runtime state. For retention and replay
limits, see
[Record and retry an image run](../runtime/image.md#record-and-retry-an-image-run).

## Validate a change

`ImageGenerationOperationTests` exercises shared defaults and adapter recipes,
unsupported options, pixel restoration, failure, cancellation, temporary-file
cleanup, and input changes between planning and execution. It injects an
executor and needs no model weights. `ImageGenerationAdapterTests` compares
explicit CLI and API requests across all supported image backends, checks API
compatibility policies, and compares preflight with execution validation.

`ImageRunRecordTests` checks atomic state snapshots, active and abandoned
leases, process termination, input retention, retry validation, and terminal
outcomes. `ImageRunCommandTests` checks opt-in flags, observational preflight,
run-plan round trips, local inspection, legacy JSON compatibility, and
admission classes.

These tests establish orchestration behavior. Checkpoint, numerical,
GPU-memory, and platform acceptance still use the family tests and runtime
gates.
