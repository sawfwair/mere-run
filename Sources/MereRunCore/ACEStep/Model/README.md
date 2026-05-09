# ACE-Step Model

Native ACE-Step transformer and conditioning layers.

- `ACEStepConfig.swift`: typed model configuration.
- `ACEStepDiT*.swift`: diffusion transformer blocks.
- `ACEStepAttention*.swift`: attention helpers and masks.
- Encoder/tokenizer files map audio and lyric conditioning into model inputs.

Prefer local shape tests for model edits. Avoid changing generation semantics
from this directory without updating `ACEStepPipeline` coverage.
