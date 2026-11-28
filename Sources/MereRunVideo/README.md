# MereRunVideo

Video and world-generation runtimes carved out of `MereRunCore`:

- `LTX/`: LTX-2.3 / LTX-2.5 distilled, full, audio-to-video, HDR, retake, and session pipelines.
- `Wan2/`: Wan 2.2 TI2V and DreamX causal video generation.
- `MiniMaxH3/`: MiniMax-H3 FL2VA, Ref2VA, FastH3, sliding-window, and Turbo adapter runtimes.
- `Cosmos3/`: Cosmos 3 world session, edge generator, and Cosmos3-Super image backend adapter (registered into `ImageGenerationOperation` at CLI startup).
- `SCAIL2/`: SCAIL-2 mask-guided video generation (depending on `MereRunVision` for SAM 3.1 mask preparation).
- `VideoGenerationOperation*.swift`, `VideoGenerationPlan+NativeOptions.swift`, `VideoGenerationArgumentParsing.swift`, `VideoGenerationLTX*.swift`, `VideoGenerationH3Preparation.swift`, and `LTXVideoTimingReport.swift`: shared video execution operations.

Lightweight video options, model-family probes, and checkpoint resource/pack metadata remain in `MereRunCore`.
