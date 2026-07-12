# Vision Flow

## Purpose

Generate dense local optical-flow vectors for motion analysis and VFX passes.

```bash
mere.run vision flow ./frame-001.png ./frame-002.png \
  --output ./frame-001-to-002.flo \
  --json-output ./frame-001-to-002.json \
  --accuracy high
```

Both images must have equal dimensions. The `.flo` artifact uses the standard
Middlebury format with one 32-bit horizontal and vertical vector per pixel.

## Sources

- `Sources/MereRunCLI/Commands/VisionFlowCommand.swift`
- `Sources/MereRunCore/OpticalFlow/NativeOpticalFlowGenerator.swift`
