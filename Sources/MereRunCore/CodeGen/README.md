# CodeGen

Local code-generation support backed by the vendored llama runtime.

- `CodeGenResources.swift`: default model IDs and resource paths.
- `LlamaContext.swift`: low-level llama context bridge.
- `OpenAITypes.swift`: local API-compatible request and response payloads.
- `CodeGenGenerator.swift`: generation orchestration.

Treat the C/C++ runtime as an external boundary. Validate CLI and API contract
changes with `Tests/MereRunCLITests`.
