# HiDreamO1

Native HiDream O1 image generation runtime.

- `HiDreamO1Configs.swift`: typed root model configuration.
- `HiDreamO1Resources.swift`: model-root discovery and required file layout.
- `HiDreamO1TokenizerAndTemplate.swift`: tokenizer and chat-template boundary.
- `HiDreamO1ImagePreprocessor.swift`: reference-image resizing and pixel packing.
- `HiDreamO1SampleBuilder.swift`: text/reference sample metadata and token layout.
- `HiDreamO1Model.swift` and `HiDreamO1Layers.swift`: native MLX model blocks.
- `HiDreamO1Scheduler.swift`: FlowMatch and UniPC timestep behavior.
- `HiDreamO1Generator.swift`: loading, conditioning, denoising, and PNG output.

Keep config/tokenizer parsing typed. Use focused parity tests for scheduler,
sample-building, and manifest defaults before changing generation behavior.
