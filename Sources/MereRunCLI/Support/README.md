# CLI Support

Shared helpers for the public command surface.

- `CLIModelStoreBootstrap.swift`: global model-root handling.
- `CLIOutput.swift` and `CLIStderr.swift`: output channel discipline.
- `BuiltinTools.swift`: local tool execution guardrails.
- `MachineInferenceAdmission.swift`: crash-safe weighted admission shared by
  heavyweight CLI and API-server processes on one machine.

Keep stdout machine-readable when a command can be scripted; diagnostics and
progress belong on stderr.
