# Vision Pose

## Purpose

Detect body, hand, and face landmarks locally for motion, compositing, and VFX
workflows.

```bash
mere.run vision pose ./person.png --json-output ./person-pose.json
```

Use `--minimum-confidence` to filter weak points. Disable landmark families with
`--no-body`, `--no-hands`, or `--no-face`. The output coordinates are normalized
with a bottom-left origin so compositing and motion tools can convert them using
the recorded image dimensions.

## Sources

- `Sources/MereRunCLI/Commands/VisionPoseCommand.swift`
- `Sources/MereRunCore/Pose/NativePoseDetector.swift`
