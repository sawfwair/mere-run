# Shared image training execution

`image train-lora` uses `ImageLoRATrainingOptions`, `ImageLoRATrainingPlan`, and
`ImageLoRATrainingOperation` in `MereRunCore`. Library callers can prepare and
execute the same Krea and Klein training paths without importing the CLI or
starting a dashboard.

## Prepare a plan

Options resolve recipes and explicit overrides, validate training controls,
and select LoRA targets. Plan resolution finds the model manifest, reads image
and caption pairs, and prepares a family-specific native trainer configuration.
It does not load weights or create output directories. Captions are retained in
the plan; native trainers read image bytes during execution.

Krea keeps its synthetic-data and frozen-base quantization options. Klein keeps
its resume checkpoint, target ranks, timestep settings, progressive resolution,
latent-cache settings, preview settings, and benchmark mode. Native trainers
continue to validate model and optimizer checkpoint compatibility.

The CLI performs numeric validation and preflight before checking MLX. For an
actual run, it checks MLX and prepares the plan before starting the dashboard.
Dataset, family, or preview-model validation failures therefore do not start a
viewer. The viewer task is cancelled and awaited when execution finishes or
fails.

## Execute a plan

The operation dispatches the prepared configuration and examples to the
existing native trainer and forwards typed progress. Klein preview generation
uses the prepared prompt, model, seed, and adapter settings. A benchmark returns
a benchmark outcome; a completed training run returns the adapter path.
Cancellation cannot return a successful outcome, but it does not remove
checkpoints or adapters already written by a native trainer.

Callers own machine admission. The CLI retains its process-level reservation;
the operation does not acquire a second permit. Native optimizers, checkpoint
formats, and sampling math remain unchanged.

## Saved plans and validation

The CLI's existing schema-v1 run plan remains the portable replay format. It
now retains optional `base_quantization_bits` when you request Krea frozen-base
quantization. Older plans without this field still decode and use the existing
unquantized default. Output relocation retains the requested quantization.

Command and operation tests cover recipes, explicit overrides, native
configurations, caption snapshots, preview preparation, benchmark outcomes,
cancellation, preflight, and saved-plan replay. Model-manifest fixtures and
injected trainers exercise these boundaries without loading real weights. They
do not establish checkpoint training quality or GPU performance.
