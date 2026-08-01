# DeepseekV4Flash

DeepSeek V4 Flash agent runtime support.

- `DeepseekV4FlashResources.swift`: managed model id, Hugging Face source, GGUF filename, and user-facing errors.
- `DeepseekV4FlashBinary.swift`: locates the bundled `ds4-server` binaries from installed, SwiftPM, or explicit override layouts.
- `DeepseekV4FlashGenerator.swift`: starts `ds4-server`, waits for readiness, and proxies OpenAI-compatible chat requests over loopback HTTP.

The managed default is the immutable official 0731 pure-Q2 imatrix GGUF. The
runtime keeps one full-resident server, uses a 32K operational context and a
1,024-token prefill chunk, and caps disk KV checkpoints at 8 GiB. SSD streaming
is intentionally not enabled by mere.run.

Keep DS4 process management here. CLI commands should choose the managed model
and serving engine, then let this module own binary discovery, model-file
resolution, server startup, and request translation.
