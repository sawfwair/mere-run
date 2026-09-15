# CLI Support

Shared helpers and runtime adapters for the public command surface.

- `APIServerContract.swift`: transport headers, JSON decoding, health, and validation errors.
- `APIServerContract+Models.swift`: discovery and engine capability projections
  from Core model profiles. Modality files own request translation and response
  construction, with each vision response schema beside its constructor.
- `APIServerContract+Fields.swift`: shared optional-field parsing and model-name
  normalization. Modality files retain their aliases, defaults, and diagnostics.
- `APIMultipartFormData.swift`: ordered multipart parsing and shared field checks.
  Each route declares its allowed fields, diagnostics, and text-decoding policy.
- `APIServer.swift`: HTTP routing, authentication, transport, and streaming ownership.
- Geometry, reconstruction, and video-depth routes call their Core generation
  operations for shared settings, execution, and awaited unloading. HTTP handlers
  retain admission, upload and output cleanup, and artifact retention.
- Speech API model aliases and voice descriptions belong to `APIServerContract`.
  `APISidecarModelPool` retains admission and residency around the shared
  `AudioCore.SpeechSynthesisOperation`.

- `CLIModelStoreBootstrap.swift`: global model-root handling.
- `CLIOutput.swift` and `CLIStderr.swift`: output channel discipline.
- `TerminalMarkdownPresentation.swift` and `TerminalMarkdownStream.swift`:
  safe, append-only Markdown presentation for interactive token streams while
  preserving raw piped output.
- `BuiltinTools.swift`: local tool authorization and execution policy.
- `BoundedProcessRunner.swift`: concurrent output drainage, bounded capture,
  monotonic deadlines, and cancellation cleanup for approved shell tools and
  code benchmark sandboxes, and workflow children. Throwing start and output
  callbacks terminate and await the child group before errors propagate.
- `WorkflowRunner.swift`: node scheduling, retry policy, and ordered events.
- `WorkflowRunStore.swift`: exclusive run ownership, resume validation, manifests,
  and synchronized event persistence.
- `WorkflowArtifactStore.swift`: input localization, output verification, cache
  storage, and digest checks for reuse.
- `WorkflowProcessRunner.swift`: child registration, bounded stdout, streaming
  callbacks, and workflow cancellation over the shared process runner.
- `APIVideoGeneration.swift`: API video field and argument translation over the
  shared Core operation. The HTTP route retains admission and artifact cleanup.
  `CLIVideoGenerationPresentation.swift` formats Core video events for the CLI.
- `APIImageGeneration.swift`: API v1 compatibility settings for the Core image
  operation. `ImageGenerationPreflight.swift` presents the same Core resolver's
  diagnostics as an observational report.
- `MachineInferenceAdmission.swift`: CLI workload classification, the shared
  state-directory adapter, and process bootstrap over `MereRunAdmission`.
- `AdmissionExports.swift`: Core progress adapters for admission telemetry.
- `RuntimeModelPool.swift` and `APISidecarModelPool.swift`: loaded-model
  runtime adapters, batching, and caches over `MereRunResidency`.
- `RuntimeServingServices.swift`: text/media composition, pressure coordination,
  request admission, maintenance, and chat/transcription service ownership.
- `RuntimeChatSession.swift`: paired model and admission leases for one chat
  request, with awaited release after generation or proxy consumption.
- `RuntimeChatProxyStream.swift`: upstream byte streaming and cancellation
  when the downstream consumer disconnects.
- `APITranscription.swift` and `CLIASRRouting.swift`: API policy and CLI
  presentation adapters over the shared speech resolver, operation, and optional
  durable records.
- `RecordedOperationRun.swift`: typed image/transcription history presentation
  shared by local run inspection and listing.

Keep stdout machine-readable when a command can be scripted; diagnostics and
progress belong on stderr.

Read [catalog and contract ownership](../../../docs/internals/catalog-contracts.md)
before changing discovery, capability metadata, or API translation.
