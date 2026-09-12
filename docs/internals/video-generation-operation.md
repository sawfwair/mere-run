# Shared video generation operation

This guide describes the runtime boundary for contributors working on video
generation. The CLI and `POST /v1/videos/generations` call
`MereRunCore.VideoGenerationOperation` with `VideoGenerationOptions`.

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
and calls Core. Prompt text and model selectors never become command arguments. It does not run the command or print an output path or receipt.
Core does not import ArgumentParser or the CLI.

The other video commands retain small CLI error adapters around the shared
checkpoint resolver. Those adapters convert Core input errors into CLI
validation errors; they do not select or load models.

## Request preparation

`VideoGenerationOperation.prepare` validates settings before checkpoint
resolution. It checks source files, resolves the checkpoint, rechecks
model-specific compatibility, and parses conditioning and adapter inputs.
LTX preparation reads adapter metadata to select HDR and reference-video
settings. Optional prompt enhancement runs during preparation.

The result contains one effective plan, a model root, and a typed native input:
LTX, source-audio LTX, Wan, or MiniMax-H3. Runtime dispatch uses the plan's
selected LTX route. Duration prediction remains inside the loaded LTX runtime;
it can replace the planned frame count when automatic duration is enabled.

Preflight uses the same validation, compound-input parser, model profiles, and
preparation rules without downloading checkpoints, enhancing prompts, or
loading tensors. Execution checks mutable inputs again.

## Admission and runtime lifetime

The operation acquires no machine permit or API request slot. CLI process
admission and the API's admitted request enclose execution. The caller also
supplies executable-specific Metal resource setup.

Video runtimes belong to one operation. They are not added to the resident
speech or image pool. LTX retains its existing load, generate, await-unload,
and media-write order, with unloading on failure. H3 retains its wired-memory
limit and stream scope; Wan retains its native generator lifetime. The
operation checks task cancellation before preparation, before execution, and
before returning an outcome. Native writers also check cancellation before
starting output. These boundaries do not promise interruption during every
native kernel.

Core returns the primary artifact URL, its kind, and optional LTX timing data.
The CLI formats diagnostics and timing reports and emits its receipt. The API
retains completed MP4 artifacts for one hour and returns their local file URL,
byte count, and SHA-256. Failed API requests remove their output directory
before releasing the request slot. Remote clients cannot use this route.

## Compatibility and validation

The API keeps its LTX 2.5 distilled model default and explicit 768 × 512 canvas.
CLI model and canvas defaults continue to depend on its selected operation.
Request parity means equivalent inputs produce the same prepared native
request; it does not require the two interfaces to have identical defaults.

Run the focused contract tests and the repository gate:

```bash
swift test --filter 'VideoGenerationOperationTests|VideoCommandTests|APIServeCommandTests'
./scripts/check.sh
```

For native acceptance, record the source revision, binary hash, checkpoint
metadata, effective settings, CLI receipts, API responses, and decoded media
before and after a change. Compare first and repeated API requests. Keep
fixture cancellation evidence separate from native cancellation recovery,
and keep short render timings separate from performance qualification.
