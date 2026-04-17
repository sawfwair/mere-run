# LTX Module

This directory owns native video generation for mere.run.

- `LTXDistilledLatentGenerator.swift`: distilled text-to-video and image-to-video path
- `LTXGemmaTextEncoder.swift`: text encoder support for LTX models
- `LTXVideoMP4Writer.swift`: final MP4 assembly

Edit here when you are changing native LTX loading, denoising, decode, or export behavior. Do not mix unrelated CLI concerns into this directory; keep command parsing in `Sources/MereRunCLI/Commands/VideoCommand.swift`.

Before stopping, run `swift test`, `./scripts/check.sh`, and a smoke path if your change affects real model execution.
