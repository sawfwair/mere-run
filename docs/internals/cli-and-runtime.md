# CLI and runtime internals

This page shows how the public `mere.run` command tree maps to the package
internals, and where to start reading for a given change. For the full target
list, see the [source layout reference](./source-layout.md).

## Top-level flow

1. `Sources/MereRunCLI/MereRunCLI.swift` defines the public modality-first
   command tree.
2. Each command in `Sources/MereRunCLI/Commands` parses arguments and does a
   small amount of orchestration.
3. Shared helpers in `Sources/MereRunCLI/Support` bootstrap the model store,
   expose the managed model registry, and provide output helpers.
4. Shared operations in `MereRunCore`, `AudioCore`, `AudioSTT`, and `AudioTTS`
   prepare requests and own runtime lifetime.
5. Model runtime targets, such as `MereRunGemmaModel` and `MereRunLTXModel`,
   perform the tensor computation.

Public command concerns stay out of the model code by design.

## Model resolution path

Most commands pass through the same model-resolution layer:

- `MereRunModelKit/MereRunModelPaths.swift`: configured storage paths
- `MereRunModelKit/MereRunModelManifest.swift`: metadata and provenance
- `MereRunModelKit/InstalledModelResolver.swift`: location policy and lookup
- `MereRunCore/ModelResolver.swift`: catalog and runtime-validation adapter

This shared path keeps model handling consistent across every modality. For
the catalog and capability contracts that sit on this layer, see
[catalog and contract ownership](./catalog-contracts.md).

## Resource ownership

[Inference admission](./inference-admission.md) separates a server's machine
reservation from its request queue; `MereRunAdmission` owns these coordinators
and their leases. [Runtime residency](./runtime-residency.md) owns model
generations, model leases, and eviction mechanics. Runtime adapters keep model
construction, cache, and batching policy.

Each modality's shared operation gives the CLI and API adapters one typed
boundary for planning and execution. Start with the operation page for the
area you're changing:

- [Chat execution](./chat-execution.md)
- [Image generation](./image-generation-operation.md)
- [Video generation](./video-generation-operation.md)
- [Speech transcription](./speech-transcription-operation.md) and
  [speech synthesis](./speech-synthesis-operation.md)
- [Single-image geometry](./geometry-generation-operation.md) and
  [vision reconstruction and depth](./vision-generation-operations.md)
- [Image training](./image-training-execution.md) and
  [text training](./text-training-execution.md)
- [Workflow execution](./workflow-execution.md)

## CLI design conventions

### Modality-first commands

Top-level commands group work by user intent: `image`, `text`, `speech`,
`vision`, `geo`, `audio`, `music`, `sfx`, `video`, and `world` for inference;
`graph`, `executor`, `relay`, `run`, and `eval` for workflows and evaluation;
and `model`, `adapter`, `status`, `api`, and the setup commands for management.
Run `mere.run --help` for the complete list.

### Canonical public model IDs

The runtime uses explicit public IDs, such as `image-klein-max`,
`text-chat-gemma4`, `text-chat-q36-nano`, and `speech-tts-qwen3-nano`. Docs,
tests, and examples use the canonical IDs that `mere.run model list` reports.

### Quiet default output

A normal successful run emits only its intended output. Stdout is
machine-readable and stderr is diagnostic. Internal bring-up logging belongs
behind the shared `MereRunRuntimeDebug` helper, not in default stdout.

## Runtime family pattern

Large runtime families split one type across files:

- The root file owns the public type and high-level orchestration.
- Companion `+` extension files own loading, generation, decoding, support, or
  diagnostics.

Examples include `Qwen3TTSGenerator`, `LightOnOCRGenerator`,
`ZImageTurboGenerator`, `QwenImageEditGenerator`, `ACEStepPipeline`, and
`ImageValidateCommand`. The runtime boundary pages list each family's reading
order, for example [Gemma](./gemma-runtime-boundaries.md),
[Qwen](./qwen-runtime-boundaries.md), and [LTX](./ltx-runtime-boundaries.md).

## Choose a starting point

Start in the CLI when you're:

- Changing flags, help text, or output shape
- Changing which runtime family a command dispatches to
- Debugging how user input becomes a runtime request

Start in the runtime when you're:

- Changing inference behavior
- Debugging loading, decoding, or generation internals
- Working inside one modality family

## Contributor reading order

1. Read the [repository tour](../repository-tour.md).
2. Read the [architecture map](../architecture.md).
3. Read one command file in `Sources/MereRunCLI/Commands`.
4. Read the matching shared operation and runtime family entry point.
5. Read the related tests in `Tests/`.
