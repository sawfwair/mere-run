# Shared text training execution

`text train-lora` uses `TextLoRATrainingOptions`, `TextLoRATrainingPlan`, and
`TextLoRATrainingOperation` in `MereRunCore`. Library callers can resolve and
run the same operation without importing ArgumentParser or the dashboard
server.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `TextLoRATrainingOptions` | Numeric and resume validation, model-family selection, and default LoRA targets |
| `TextLoRATrainingPlan` | Typed snapshots of the training and optional evaluation datasets |
| `TextLoRATrainingOperation` | Create the output directory, invoke the native pipeline, and publish the adapter manifest |
| Native pipelines | Model loading, tokenization, image integrity checks, optimizer state, metrics, and tensor artifacts |
| CLI adapter | ArgumentParser errors, the MLX check, viewer flags, stdout formatting, and diagnostic progress |

## Plan and validate a request

Plan resolution reads the training JSONL and the optional `--eval` JSONL into
typed dataset snapshots. Gemma vision training keeps its single-image policy
and image digests; text-only families reject media. Resolution doesn't load
weights or create output directories.

The CLI translates option errors into ArgumentParser validation errors. For a
real training run, it checks MLX availability before it reads datasets.

## Execution and runtime lifetime

The operation creates the output directory, invokes the selected native
pipeline, and writes the adapter manifest after a successful run. `--dry-run`
writes a `prepared` manifest without invoking a trainer or replacing an
existing adapter. The default model, training settings, target ordering, and
manifest format are unchanged by this split.

The operation passes the training and evaluation examples, image digests,
metadata, resume inputs, and progress callbacks to the pipeline unchanged.
Failure or cancellation never publishes a new `trained` manifest. Existing
checkpoints stay available for resume; cancellation doesn't roll back tensor
files that a trainer already wrote.

The CLI acquires its machine reservation at process startup, and the operation
doesn't acquire another permit. Library callers own admission around their
training work. The optional `--visualize` dashboard starts after its initial
event is written; the CLI cancels and awaits its server task when the command
exits.

## Resume

Training resumes from optimizer-bearing checkpoints and LoRA manifests with
`--resume-from`. It doesn't use the `run inspect` or `run retry` record formats
that image and transcription runs use. `--resume-step` keeps its global-step
meaning for legacy checkpoints; native trainers validate actual optimizer
compatibility.

## Validate a change

`TextTrainingCompatibilityTests` covers CLI validation order, empty datasets,
dry-run artifacts, and target handling. `TextLoRATrainingOperationTests` covers
family resolution, dataset snapshots, publication, failure, cancellation, and
resume inputs with injected trainers. Vision and optimizer tests cover their
own contracts. None of these tests qualify real model checkpoints or measure
training quality.
