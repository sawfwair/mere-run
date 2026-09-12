# Shared single-image geometry operation

Use `MoGe2GenerationOperation` in `MereRunCore` when running a complete MoGe-2
request. Both `vision geometry` and `POST /v1/vision/geometry` call this operation.

## Settings and planning

`MoGe2GenerationSettings` validates resolution level, target token count, and
maximum point count before constructing `MoGe2InferenceConfiguration`. The
existing configuration initializer retains its compatibility clamping behavior
for direct runtime callers; CLI and API values pass through the validated settings.

`MoGe2GenerationRequest` carries the input, output directory, model selection,
and settings. The CLI accepts managed model IDs and local paths. The API retains
its restriction to `vision-geometry-moge2-small` and uses the default managed
lookup. Both adapters default to resolution level 9 and 3,600 target tokens.

`MoGe2GenerationOperation.prepare` reads input dimensions and returns an
observational `MoGe2GenerationPlan`. It uses `VFXImageInputValidator` and
`MoGe2TokenGrid` to enforce the same dimension and derived-grid bounds as native
execution. It does not load weights, download models, or create output files.
The CLI formats this plan for `--dry-run`.

The derived grid can exceed the target token count because rows and columns
are rounded independently. For example, a 2,000 × 1,080 image at 3,600 target
tokens requires 3,608 grid tokens. Both planning and execution reject this
request before loading a model. Select a lower token count to stay within the
3,600-token workload ceiling.

## Execution and ownership

Execution reads the current input dimensions again before runtime setup. The
native generator then captures a bounded immutable input snapshot and verifies
the pinned checkpoint before inference. Its preprocessing, model computation,
postprocessing, and geometry exporters retain their existing implementations.

The operation creates one runtime for the request and awaits its unload on
success, failure, or cooperative cancellation. It checks cancellation before
accepting a result and after cleanup. The native generator checks cancellation
between preparation, inference, postprocessing, and artifact export. These
checks do not establish recovery from a GPU failure or immediate interruption
of an active GPU kernel.

The caller owns request admission. The operation acquires no additional permit
and does not borrow a runtime from a resident model pool. The API retains upload
validation, temporary-file cleanup, failed-output cleanup, hashed artifact
responses, loopback access, and one-hour artifact retention. Its runtime-setup
callback checks the executable's MLX resources. The CLI retains progress on
stderr and its existing JSON or manifest-path output on stdout.

## Validation boundaries

Fixture tests cover equivalent CLI/API requests, invalid settings, changing
input dimensions, missing inputs, awaited runtime cleanup, cancellation, single
request admission, and artifact hashes. Native compatibility comparisons must
identify their source revision, executable, input, checkpoint, settings, and
output normalization. Fixture results alone do not qualify GPU recovery,
performance, or geometry quality.

Multi-view geometry, TripoSR, InstantMesh, and video depth retain their separate
execution paths. This operation does not change their ownership.
