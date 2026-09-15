# Shared text training execution

`text train-lora` uses `TextLoRATrainingOptions`, `TextLoRATrainingPlan`, and
`TextLoRATrainingOperation` in `MereRunCore`. Library callers can resolve and run
the same operation without importing ArgumentParser or the dashboard server.

## Resolve a training plan

Options own numeric and resume validation, model-family selection, and default
LoRA targets. Plan resolution reads the training and optional evaluation JSONL
into typed dataset snapshots. Gemma vision training retains its single-image
policy and image digests. Text families reject media. Resolution does not load
weights or create output directories.

The CLI translates option errors to ArgumentParser validation errors and checks
MLX availability before reading datasets for an actual training run. Viewer
flags, stdout formatting, and diagnostic progress remain in the CLI.

## Execute and publish

The operation creates the output directory, invokes the selected native
pipeline, and writes the existing adapter manifest after successful completion.
A dry run writes a `prepared` manifest without invoking a trainer or replacing
an existing adapter. The default model, training settings,
target ordering, and manifest format remain unchanged.

Native pipelines own model loading, tokenization, image integrity checks,
optimizer state, metrics, and tensor artifacts. The shared operation preserves
the training and evaluation examples, image digests, metadata, resume inputs,
and progress callbacks passed to those pipelines. Failure or cancellation does
not publish a new `trained` manifest. Existing checkpoints remain available for
resume; cancellation does not roll back tensor files written by a trainer.

The CLI acquires its machine reservation at process startup. The operation does
not acquire another permit. Library callers own admission around their training
work. The optional CLI dashboard starts after its initial event is written and
its server task is cancelled and awaited when the command exits.

## Resume and validation boundaries

Training continues to use optimizer-bearing checkpoints and existing LoRA
manifests. It does not use the image and transcription `run inspect` or `run
retry` record formats. `--resume-step` retains its global-step meaning for legacy
checkpoints; native trainers validate actual optimizer compatibility.

`TextTrainingCompatibilityTests` covers CLI validation order, empty datasets,
dry-run artifacts, and target handling. `TextLoRATrainingOperationTests` covers
family resolution, dataset snapshots, publication, failure, cancellation, and
resume inputs with injected trainers. Existing vision and optimizer tests cover
their respective contracts. These tests do not qualify real model checkpoints
or measure training quality.
