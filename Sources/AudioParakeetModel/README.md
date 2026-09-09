# Parakeet model runtime

Use this library when changing Parakeet tensor computation or decoder behavior.
It depends on MLX, without Core, audio-file decoding, or Core ML frameworks.

- `ParakeetConfig.swift`: typed checkpoint configuration and packaging facts.
- `ParakeetConformer*.swift`: encoder attention, convolution, and subsampling.
- `ParakeetTransducerLayers.swift`: recurrent prediction and joint projection.
- `ParakeetModel.swift`: model composition and the public model factory.
- `ParakeetTDTModel.swift`, `ParakeetRNNTModel.swift`, and `ParakeetCTCModel.swift`:
  token decoding, batched frame readback, and alignment.
- `ParakeetExecution.swift`: interfaces for external encoder and TDT execution.
- `ParakeetAlignment.swift`: aligned tokens, sentences, and overlap merging.
- `ParakeetModelTimings.swift`: decoder timings and the monotonic clock.

Load configuration with `ParakeetModelConfig.load(from:)`, then construct a
native decoder with `ParakeetModelFactory.build(config:)`. The caller loads
weights and supplies mel tensors. The factory does not resolve or download a
checkpoint.

`AudioSTT/Parakeet` owns resource resolution, audio preparation, generators,
Core ML implementations, and conversion to shared ASR results. The package
interfaces let those implementations supply tensors without bringing Core ML
into this library. Provider errors retain their existing types and messages.

Keep recurrent state, batched window order, token timing, and readback cadence
unchanged when splitting decoder code. `SpeechRuntimeTests` exercises these
contracts without importing `AudioSTT` or `MereRunCore`.
