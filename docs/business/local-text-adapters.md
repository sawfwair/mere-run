# Train and evaluate a local text adapter

Use Dataset Tools when you have reviewed examples of a repeatable text task
and need to test whether a low-rank adaptation (LoRA) adapter improves that
task. The workflow prepares data, trains through the native runtime, and
compares the base model with the adapter on matched behavior prompts.

Start with a task whose expected outputs you can check. Training completion
alone doesn't establish an improvement, and lower held-out loss doesn't prove
that behavior improves.

## Before you begin

Follow [Set up companion workflows](./plugin-workflows.md). Install Dataset
Tools and a managed model supported by native text LoRA training:

```bash
mere.run plugin install mere-dataset-tools --yes
mere.run plugin doctor mere-dataset-tools
mere.run model capabilities
mere.run model info text-chat-laguna-xs-2-1
```

Pull the model if needed after reviewing its terms. Check
[native text training](../runtime/text.md#train-a-lora-adapter) for supported
models and memory requirements. The plugin doesn't make an unsupported model
trainable.

## Separate the examples before training

Prepare three reviewed JSONL files using the
[synthetic example format](https://github.com/sawfwair/mere-run-plugins/tree/main/examples/text-lora):

| File | Purpose |
| --- | --- |
| `train.jsonl` | Chat examples used to update the adapter |
| `held-out.jsonl` | Separate chat examples for assistant-token loss before and after training |
| `behavior.jsonl` | Prompts with case-sensitive expected responses for base and adapter comparison |

Chat examples start with a system message and alternate user and assistant
messages; each message needs at least four characters. Behavior rows contain
`id`, `prompt`, and `expected`. Exact matching ignores surrounding whitespace.

Preparation rejects user prompts repeated across splits after Unicode, case,
and whitespace normalization. Review paraphrases and related records yourself.
Keep examples from the same underlying case together so near-duplicates don't
inflate the result.

Prepare and pin the files in a private directory:

```bash
umask 077
mere.run plugin run mere-dataset-tools -- text-lora prepare   --train-data ./train.jsonl --held-out-data ./held-out.jsonl   --behavior-data ./behavior.jsonl --output-dir ./prepared
```

`dataset.json` records file hashes. The prepared directory contains private
examples.

## Validate and train

Check data through the native runtime without loading a model, then train:

```bash
mere.run plugin run mere-dataset-tools -- text-lora train   --dataset ./prepared/dataset.json --model text-chat-laguna-xs-2-1   --steps 100 --output-dir ./training-preview --dry-run
mere.run plugin run mere-dataset-tools -- text-lora train   --dataset ./prepared/dataset.json --model text-chat-laguna-xs-2-1   --steps 100 --output-dir ./trained
```

A completed run writes `adapter.safetensors`, an adjacent native manifest, and
`training.json` with before-and-after held-out loss and assistant-token counts.
The example step count is a starting configuration, not a quality recommendation.
To continue an optimizer-bearing native checkpoint, use `--resume-from` with
matching data and settings and a fresh output directory.

## Compare the adapter with its base model

Run the paired evaluation:

```bash
mere.run plugin run mere-dataset-tools -- text-lora evaluate   --dataset ./prepared/dataset.json --training ./trained/training.json   --output-dir ./comparison
```

Evaluation verifies recorded hashes and uses the same managed model and greedy
sampling for both arms. Its report lists exact-response pass rates and cases
that improve or regress. Raw prompts and responses are omitted from the report.
Add `--resume` with the same arguments to continue an interrupted comparison.

Read the behavior report alongside held-out loss. Exact-response tests suit
fixed outputs; they don't measure every acceptable paraphrase or broad model
quality. Before adoption, add representative failure cases and separate human
review for outputs that exact matching can't assess. Keep the base-model
identity, adapter, data receipt, and reports together.
