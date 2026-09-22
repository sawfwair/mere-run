# FalconPerception

Vision grounding and perception runtime.

- `FalconPerceptionConfig.swift`: typed model configuration.
- `FalconPerceptionTokenizer.swift`: tokenizer compatibility boundary.
- `FalconPerceptionModel.swift`: native model layers.
- `FalconPerceptionGrounder.swift`: user-facing grounding pipeline.
- `FalconPerceptionProcessor.swift`: image and prompt preprocessing.

Keep tokenizer/config quirks isolated in boundary files and cover grounding
output contracts with focused tests.

## Coordinate token selection

`FalconPerceptionCoordinateDecoder` selects coordinate bins before the selected
coordinate is embedded into the next token. It follows the generation policy in
upstream `modeling_falcon_perception.py` at revision
`54916b3dec58565fafc6d82eb3051fe7246ab666`:

- Keep every coordinate token in the current query's history, including
  coordinates that have no completed detection. Batch slots have separate histories.
- Select the first maximum on each axis. A repeat requires both normalized axes
  to differ from an earlier coordinate by strictly less than 0.01.
- Suppress both selected bins for a repeat and select again. Try at most 100
  candidates, retaining the final candidate even if it repeats.
- Preserve double-precision bin ratios in history before casting the chosen
  coordinate for the model's embedding. Float rounding can change the strict
  threshold decision.

This policy does not deduplicate final boxes or masks. Preprocessing, tensor
numerics, token limits, mask generation, and task-specific detection quality
require separate validation. The pure Swift policy tests do not load weights or
establish full model parity.
