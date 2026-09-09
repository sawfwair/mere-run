# Shared Qwen vision tower

Vision tower components used by Core's image and vision-language runtimes.

- `QwenVisionTower.swift`: tower entry point.
- `QwenVisionBlock.swift`: block implementation.
- `QwenVisionAttention.swift`: attention behavior.
- Patch, grid, MLP, and merger files own preprocessing/model details.

Keep this directory limited to vision-tower math and tensor preparation.
