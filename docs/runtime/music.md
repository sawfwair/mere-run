# Music Runtime

This page covers the native music-generation path exposed through
`mere.run music generate`.

## Public surface

- `mere.run music generate`

## Model family

- `music-acestep`

## Typical workflow

```bash
swift run mere.run music generate \
  "upbeat electronic groove" \
  --output ./track.wav
```

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/MusicGenerateCommand.swift`

### Runtime

- `Sources/MereRunCore/ACEStep/ACEStepPipeline.swift`
- `Sources/MereRunCore/ACEStep/ACEStepPipeline+Prompting.swift`
- `Sources/MereRunCore/ACEStep/ACEStepPipeline+Generation.swift`

## Reading order

The ACEStep runtime now follows a clean phase split:

1. `ACEStepPipeline.swift` for the public pipeline and orchestration
2. `ACEStepPipeline+Prompting.swift` for prompt preparation and conditioning
3. `ACEStepPipeline+Generation.swift` for the generation path itself

That makes it much easier to follow than a single pipeline monolith.

## Contributor notes

- this is a native Swift/MLX path, not a Python bridge
- model resolution and storage still follow the same canonical public rules as
  the rest of the repo
