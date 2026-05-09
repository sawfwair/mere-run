# FalconPerception

Vision grounding and perception runtime.

- `FalconPerceptionConfig.swift`: typed model configuration.
- `FalconPerceptionTokenizer.swift`: tokenizer compatibility boundary.
- `FalconPerceptionModel.swift`: native model layers.
- `FalconPerceptionGrounder.swift`: user-facing grounding pipeline.
- `FalconPerceptionProcessor.swift`: image and prompt preprocessing.

Keep tokenizer/config quirks isolated in boundary files and cover grounding
output contracts with focused tests.
