# Psi

Psi/GLM-4.7 chat runtime.

- `GLM47FlashConfig.swift`: typed model configuration.
- `GLM47Tokenizer.swift`: tokenizer loading and chat encoding.
- `GLM47ChatTemplate.swift`: prompt rendering.
- `GLM47Flash*.swift`: native model and generation.
- `Psi3ChatGenerator.swift`: `ChatGenerator` wrapper.

Keep prompt templates deterministic and typed. Avoid raw tool dictionaries in
new code.
