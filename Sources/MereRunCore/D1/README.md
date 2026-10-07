# D1 decision orchestration

Loads pinned, original LiquidAI safetensors with strict shape and key checks for
the text, head, vision, and FP32 audio modules. Each question runs independently with fresh
state; input token usage counts each executed row. Core owns ordered request
JSON, exact checkpoint prompts, local images/audio, preflight, and typed answers.
See `docs/runtime/d1.md` for limits and qualification.
