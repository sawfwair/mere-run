# QwenImageEdit Transformer

MMDiT transformer layers for image editing.

- `MMDiT.swift`: transformer entry point.
- `MMDiTBlock.swift`: joint and single block behavior.
- `MMDiTAttention.swift`: attention projections.
- `MMDiTRoPE.swift`: rotary position embedding.
- `AdaLNZero.swift`: adaptive normalization.

This directory should stay model-math-only. Loading, CLI, and filesystem logic
belong outside it.
