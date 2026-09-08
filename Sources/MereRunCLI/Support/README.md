# CLI Support

Shared helpers for the public command surface.

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
- `MachineInferenceAdmission.swift`: crash-safe weighted admission shared by
  heavyweight CLI and API-server processes on one machine.

Keep stdout machine-readable when a command can be scripted; diagnostics and
progress belong on stderr.
