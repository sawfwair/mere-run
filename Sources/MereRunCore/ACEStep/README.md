# ACE-Step

Music generation pipeline and ACE-Step model resources.

- `ACEStepPipeline*.swift`: prompt preparation and generation orchestration.
- `ACEStepRuntimePreparation.swift`: shared checkpoint discovery and audio decoding.
- `ACEStepAdapterTrainingOptions.swift` and `ACEStepAdapterTrainingPlan.swift`:
  validated inputs and typed JSON/JSONL manifest preparation.
- `ACEStepAdapterTrainingOperation*.swift`: audio preparation, native dispatch,
  progress, and terminal training events. Callers own machine admission.
- `ACEStep5HzLM*.swift`: language-model token path for audio codes.
- `ACEStepTurboScheduler.swift`: scheduler behavior.
- `Model/` and `VAE/`: native model building blocks.

Real checkpoint tests are environment-gated. Keep shape and scheduler behavior
covered with local unit tests when changing this directory.
