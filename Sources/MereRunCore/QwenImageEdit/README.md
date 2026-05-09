# QwenImageEdit

Qwen image-editing runtime.

- `QwenImageEditConfigs.swift`: typed runtime and model configuration.
- `QwenImageEditGenerator*.swift`: loading, encoding, generation.
- `QwenImageEditLatentCreator.swift`: latent setup.
- `FlowMatchEulerScheduler.swift`: scheduler behavior.
- `Model/` and `Tokenizer/`: native components and tokenizer boundary.

Keep scheduler and latent behavior covered by focused tests before changing
generation defaults.
