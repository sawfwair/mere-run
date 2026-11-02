# QwenImageEdit

Qwen text-to-image and image-editing runtime. Input and reference images are
optional; editing accepts up to three ordered pictures.

- `QwenImageEditConfigs.swift`: typed runtime and model configuration.
- `QwenImageEditGenerator*.swift`: loading, encoding, generation.
- `QwenImageEditLatentCreator.swift`: latent setup.
- `FlowMatchEulerScheduler.swift`: scheduler behavior.
- `Model/` and `Tokenizer/`: native components and tokenizer boundary.

Keep scheduler and latent behavior covered by focused tests before changing
generation defaults.
