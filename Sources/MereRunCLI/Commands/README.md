# mere.run CLI Commands

This directory owns the public CLI surface.

- one command file per modality or command cluster
- parser defaults and public flags are covered in `Tests/MereRunCLITests/`
- stdout should remain machine-readable where possible; diagnostics belong on stderr
- `speech synthesize` owns profiles, optional reference transcription, progress,
  and receipts. AudioTTS resolves models; AudioCore validates synthesis and
  publishes completed audio through the shared operation.
- `video generate` translates flags and compound arguments into Core video
  options. Keep model defaults and compatibility in `VideoGenerationOptions`
  and `VideoGenerationPlan`. `VideoGenerationOperation` owns execution; the CLI
  formats progress, timings, and receipts. Keep preflight JSON and filesystem
  diagnostics in `Support/VideoGenerationPreflight.swift`.
- `vision geometry` translates options into Core MoGe-2 settings. Its dry-run
  presents the shared token-grid plan; execution uses `MoGe2GenerationOperation`.

If you change a flag, subcommand name, or help contract, update the nearest parsing tests and any affected user-facing docs in the same change.
