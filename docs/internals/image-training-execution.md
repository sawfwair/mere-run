# Shared image training execution

`image train-lora` uses `ImageLoRATrainingOptions`, `ImageLoRATrainingPlan`,
and `ImageLoRATrainingOperation` in `MereRunCore/LoRA`. Library callers can
prepare and run the same Krea and Klein training paths without importing the
CLI or starting a dashboard.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `ImageLoRATrainingOptions` | Resolve recipes and explicit overrides, validate training controls, and select LoRA targets |
| `ImageLoRATrainingPlan` | Find the model manifest, read image and caption pairs, and prepare a family-specific native trainer configuration |
| `ImageLoRATrainingOperation` | Dispatch the prepared configuration to the native trainer and forward typed progress |
| Native trainers | Validate model and optimizer checkpoint compatibility, read image bytes, and own optimizers, checkpoint formats, and sampling math |
| CLI adapter | Numeric validation, preflight, the MLX check, the optional dashboard, and the saved run plan |

## Plan and validate a request

Plan resolution doesn't load weights or create output directories. The plan
keeps captions; native trainers read image bytes during execution.

Krea keeps its synthetic-data and frozen-base quantization options. Klein keeps
its resume checkpoint, target ranks, timestep settings, progressive resolution,
latent-cache settings, preview settings, and benchmark mode.

The CLI runs numeric validation and `--preflight` before it checks MLX. For a
real run, it checks MLX and prepares the plan before it starts the dashboard, so
dataset, family, or preview-model validation failures never start a viewer.

## Execution and runtime lifetime

The operation sends the prepared configuration and examples to the native
trainer. Klein preview generation uses the prepared prompt, model, seed, and
adapter settings. A benchmark returns a benchmark outcome; a completed training
run returns the adapter path. Cancellation can't return a successful outcome,
but it doesn't remove checkpoints or adapters that a native trainer already
wrote.

Callers own machine admission. The CLI keeps its process-level reservation, and
the operation doesn't acquire a second permit. When `--visualize` is set, the
CLI cancels and awaits the viewer task after execution finishes or fails.

## Saved plans

The CLI's schema-v1 run plan is the portable replay format. It keeps the
optional `base_quantization_bits` field when you request Krea frozen-base
quantization with `--base-quantization-bits`. Older plans without this field
still decode and use the unquantized default. Output relocation keeps the
requested quantization.

## Validate a change

Command and operation tests cover recipes, explicit overrides, native
configurations, caption snapshots, preview preparation, benchmark outcomes,
cancellation, preflight, and saved-plan replay. Model-manifest fixtures and
injected trainers exercise these boundaries without loading real weights. They
don't establish checkpoint training quality or GPU performance.
