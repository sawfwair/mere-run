# FalconPerception

Vision grounding and perception runtime.

- `FalconPerceptionConfig.swift`: typed model configuration.
- `FalconPerceptionTokenizer.swift`: tokenizer compatibility boundary.
- `FalconPerceptionModel.swift`: native model layers.
- `FalconPerceptionGrounder.swift`: user-facing grounding pipeline.
- `FalconPerceptionProcessor.swift`: image and prompt preprocessing.

Keep tokenizer/config quirks isolated in boundary files and cover grounding
output contracts with focused tests.

Direct and batched generation stop at either the model-configured EOS or the
`<|end_of_query|>` token resolved from the tokenizer vocabulary. Each batch slot
stops independently; remaining slots continue. The query token ID is not
hardcoded. Tokenizers without that marker retain model-EOS stopping behavior.
