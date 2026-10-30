# Shared single-image geometry operation

`vision geometry` and `POST /v1/vision/geometry` run a complete MoGe-2 request
through `MoGe2GenerationOperation` in `MereRunCore`. Multi-view geometry,
TripoSR, InstantMesh, and video depth have separate paths; see
[shared vision generation operations](./vision-generation-operations.md).

## Ownership

| Owner | Responsibility |
| --- | --- |
| `MoGe2GenerationSettings` | Validate resolution level, target token count, and maximum point count |
| `MoGe2GenerationRequest` | Carry the input, output directory, model selection, and settings |
| `MoGe2GenerationOperation` | Plan the request, run one native runtime, and await its unload |
| CLI adapter | Accept managed model IDs and local paths, format `--dry-run` plans, report progress on stderr, and write JSON or a manifest path to stdout |
| API adapter | Restrict the model to `vision-geometry-moge2-small`, validate uploads, clean up temporary and failed output, return hashed artifacts, and retain them for one hour on loopback |

## Plan and validate a request

`MoGe2GenerationSettings` validates its values before it constructs
`MoGe2InferenceConfiguration`. Direct runtime callers that use the
configuration initializer keep its clamping behavior; CLI and API values pass
through the validated settings. Both adapters default to resolution level 9 and
3,600 target tokens.

`MoGe2GenerationOperation.prepare` reads the input dimensions and returns an
observational `MoGe2GenerationPlan`. It uses `VFXImageInputValidator` and
`MoGe2TokenGrid` to enforce the same dimension and derived-grid bounds as
native execution. It doesn't load weights, download models, or create output
files.

The derived grid can exceed the target token count because rows and columns
round independently. For example, a 2,000 × 1,080 image at 3,600 target tokens
needs 3,608 grid tokens. Planning and execution both reject that request before
a model loads. Choose a lower token count to stay within the 3,600-token
ceiling.

## Execution and runtime lifetime

Execution reads the input dimensions again before runtime setup. The native
generator then captures a bounded immutable input snapshot and verifies the
pinned checkpoint before inference. Preprocessing, model computation,
postprocessing, and geometry export are unchanged native implementations.

The operation creates one runtime per request and awaits its unload on success,
failure, or cooperative cancellation. It checks cancellation before it accepts
a result and after cleanup. The native generator also checks between
preparation, inference, postprocessing, and artifact export. These checks don't
promise recovery from a GPU failure or immediate interruption of a running GPU
kernel.

The caller owns request admission. The operation acquires no additional permit
and doesn't borrow a runtime from a resident model pool. The API's
runtime-setup callback checks the executable's MLX resources.

## Validate a change

Fixture tests cover equivalent CLI and API requests, invalid settings, changing
input dimensions, missing inputs, awaited runtime cleanup, cancellation,
single-request admission, and artifact hashes.

Native comparisons must name their source revision, executable, input,
checkpoint, settings, and output normalization. Fixture results alone don't
qualify GPU recovery, performance, or geometry quality.
