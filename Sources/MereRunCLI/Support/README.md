# CLI Support

Shared helpers and runtime adapters for the public command surface.

- `APIServerContract.swift`: API wire types, model policy, and request/response contracts.
- `APIServer.swift`: HTTP routing, authentication, transport, and streaming ownership.

- `CLIModelStoreBootstrap.swift`: global model-root handling.
- `CLIOutput.swift` and `CLIStderr.swift`: output channel discipline.
- `TerminalMarkdownPresentation.swift` and `TerminalMarkdownStream.swift`:
  safe, append-only Markdown presentation for interactive token streams while
  preserving raw piped output.
- `BuiltinTools.swift`: local tool authorization and execution policy.
- `BoundedProcessRunner.swift`: concurrent output drainage, bounded capture,
  monotonic deadlines, and cancellation cleanup for approved shell tools and
  code benchmark sandboxes.
- `APIImageGeneration.swift`: API v1 compatibility settings for the Core image
  operation. `ImageGenerationPreflight.swift` presents the same Core resolver's
  diagnostics as an observational report.
- `MachineInferenceAdmission.swift`: CLI workload classification, the shared
  state-directory adapter, and process bootstrap over `MereRunAdmission`.
- `AdmissionExports.swift`: Core progress adapters for admission telemetry.
- `RuntimeModelPool.swift` and `APISidecarModelPool.swift`: loaded-model
  runtime adapters, batching, and caches over `MereRunResidency`.
- `RuntimeServingServices.swift`: text/media composition, pressure coordination,
  request admission, maintenance, and transcription service ownership.
- `APITranscription.swift` and `CLIASRRouting.swift`: API policy and CLI
  presentation adapters over the shared speech resolver, operation, and optional
  durable records.
- `RecordedOperationRun.swift`: typed image/transcription history presentation
  shared by local run inspection and listing.

Keep stdout machine-readable when a command can be scripted; diagnostics and
progress belong on stderr.
