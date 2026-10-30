# Shared video generation operation

`video generate` and `POST /v1/videos/generations` both call
`MereRunCore.VideoGenerationOperation` with `VideoGenerationOptions`. Use this
map when you change how either interface prepares or runs a video request.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `VideoGenerate` | CLI arguments, output-path defaults, preflight presentation, diagnostics, timing files, and receipts |
| `APIServerContract.VideoGenerationPlan` | API field bounds, API defaults, and protected optional arguments |
| `APIVideoGeneration` | Convert API fields and optional arguments into Core settings, then build the artifact result |
| `VideoGenerationArgumentParser` | Parse compound image, reference-video, frame, and adapter inputs for preflight and execution |
| `VideoGenerationModelResolver` | Resolve installed or explicit checkpoints and validate their layouts |
| `VideoGenerationPlan` | Normalize geometry, frames, seeds, decoder selection, and native recipes |
| `VideoGenerationOperation` | Recheck inputs, prepare requests, dispatch native execution, unload owned models, and write media |
| HTTP route | Authentication, loopback access, request admission, artifact retention, and cleanup after failure |

The API's optional `options` array keeps the CLI argument syntax. Its adapter
uses `VideoGenerate` to parse only that array, assigns JSON fields directly,
and calls Core. Prompt text and model selectors never become command arguments,
and the adapter doesn't run the command or print an output path or receipt.
Core doesn't import ArgumentParser or the CLI.

The other video commands keep small CLI error adapters around the shared
checkpoint resolver. These adapters convert Core input errors into CLI
validation errors; they don't select or load models.

## Plan and validate a request

`VideoGenerationOperation.prepare` validates settings before it resolves a
checkpoint. It then checks source files, resolves the checkpoint, rechecks
model-specific compatibility, and parses conditioning and adapter inputs. LTX
preparation reads adapter metadata to select HDR and reference-video settings.
Optional prompt enhancement runs during preparation.

The result holds one effective plan, a model root, and a typed native input:
LTX, LTX audio-to-video, Wan, or MiniMax-H3. Runtime dispatch uses the plan's
selected LTX route. Duration prediction stays inside the loaded LTX runtime and
can replace the planned frame count when automatic duration is on.

Preflight uses the same validation, compound-input parser, model profiles, and
preparation rules without downloading checkpoints, enhancing prompts, or
loading tensors. Execution checks mutable inputs again.

## Execution and runtime lifetime

The operation acquires no machine permit or API request slot. CLI process
admission and the API's admitted request enclose execution. The caller also
supplies executable-specific Metal resource setup.

Video runtimes belong to one operation; they never join the resident speech or
image pool. LTX keeps its load, generate, await-unload, and media-write order,
and unloads on failure. H3 keeps its wired-memory limit and stream scope; Wan
keeps its native generator lifetime.

The operation checks task cancellation before preparation, before execution,
and before it returns an outcome. Native writers also check cancellation before
they start output. These checks don't promise interruption inside every native
kernel.

Core returns the primary artifact URL, its kind, and optional LTX timing data.
The CLI formats diagnostics and timing reports and emits its receipt. The API
keeps completed MP4 artifacts for one hour and returns their local file URL,
byte count, and SHA-256. A failed API request removes its output directory
before it releases the request slot. Remote clients can't use this route.

### Interface defaults

The API keeps its `video-ltx25-distilled-bf16` default and an explicit 768 × 512
canvas. CLI model and canvas defaults depend on the selected operation.
Request parity means that equivalent inputs produce the same prepared native
request; the two interfaces don't need identical defaults.

## Validate a change

Run the focused contract tests and the repository gate:

```bash
swift test --filter 'VideoGenerationOperationTests|VideoCommandTests|APIServeCommandTests'
./scripts/check.sh
```

For native acceptance, record the source revision, binary hash, checkpoint
metadata, effective settings, CLI receipts, API responses, and decoded media
before and after a change. Compare first and repeated API requests. Keep
fixture cancellation evidence separate from native cancellation recovery, and
keep short render timings separate from performance qualification.
