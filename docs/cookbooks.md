# Cookbooks

This page is for mere.run Studio and command-line users. The CLI and the
macOS app bundle two kinds of offline guide: model guides, which describe one
model family, and command cookbooks, which describe one command. Use them to
prepare prompts, choose inputs, and review options without a network
connection. For source research and recipe status, see the
[provider prompting guide tracker](./provider-prompting-guides.md).

## Read a model guide

As of v0.67.1, the model handbook covers 181 managed model IDs in 77 families.
Each guide includes an example, supported controls, variant notes,
review advice, and sources. Guides for models without a text-prompt interface
describe how to prepare inputs instead.

To read a model guide in macOS Studio, follow these steps:

1. Click **Help > mere.run Guide**, or press <kbd>Command+?</kbd>.
2. In **Guide collection**, select **Models**.
3. In the search field, enter a family name or model ID.
4. Select the guide.
5. In **Model**, select a variant.

To list the model families and their managed model IDs, run:

```bash
mere.run guide --list-models
```

To read a model guide in a terminal, pass a model ID:

```bash
mere.run guide --model image-klein-9b
```

The guide text, examples, and index are bundled with the CLI and macOS app.
Reading guides requires neither a network connection nor downloaded model
weights. Source links require a network connection. Before running inference
offline, download the checkpoint and the components required by your workflow.

The recipes are editorial drafts, not records of successful inference for
every checkpoint. When provider guidance is unverified, a guide separates
provider material from local workflow advice.

## Read a command cookbook

The **Commands** collection contains command cookbooks. Each cookbook describes
required models, options, prompting patterns, examples, and troubleshooting.

To read the image generation cookbook in a terminal, run:

```bash
mere.run guide image generate
```

To list the command topics with the models each one covers, run:

```bash
mere.run guide --list
```

To save a Markdown index of command topics, run:

```bash
mere.run guide --list --markdown > guides.md
```

## Available topics

As of v0.67.1, `mere.run guide --list` includes the following topics for
creating and analyzing media:

- `audio edit`
- `image generate`
- `image train-lora`
- `image validate`
- `text chat`
- `text code`
- `text embed`
- `text decide`
- `text classify`
- `text extract`
- `text anonymize`
- `speech synthesize`
- `speech transcribe`
- `speech profile`
- `vision caption`
- `vision embed`
- `vision inspect`
- `vision ground`
- `vision segment`
- `vision track`
- `vision track-live`
- `vision pose`
- `vision flow`
- `vision geometry`
- `vision geometry-multiview`
- `vision image-to-3d`
- `vision image-to-3d-trellis2`
- `vision image-to-3d-multiview`
- `vision depth-video`
- `vision ocr`
- `audio enhance`
- `music generate`
- `music analyze`
- `music transcribe`
- `music separate`
- `sfx generate`
- `video generate`
- `video cosmos3`
- `video export-latents`
- `api serve`
- `open-webui`
- `plugin`
- `status`

The following topics cover model management and setup:

- `model list`
- `model runtime`
- `model benchmark`
- `model capabilities`
- `model info`
- `model pull`
- `model remove`
- `model storage`
- `model repair-manifests`
- `setup`
- `agent onboard`
- `agent status`
- `agent install-pi`
- `agent start`

Commands without a topic, such as `speech diarize`, `geo`, and `graph`, are
covered by their `--help` text and the [CLI reference](./cli.md). Some of their
models have model guides; for example, `mere.run guide --list-models` lists the
Nemotron 3 diarization and TerraMind guides.

## Read a command guide for a model

Some command cookbooks include model-specific advice. To read the image
cookbook for Z-Image nano, run:

```bash
mere.run guide image generate --model image-zimage-nano
```

The CLI checks whether the guide covers the requested model. If the model
doesn't belong to the topic, the command reports the supported model IDs.

## Read guides from an agent

This procedure is for developers who configure agent instructions. Retrieve
only the guide needed for the task to limit the prompt context:

1. To find model guides and their exact IDs, run `mere.run guide --list-models --json`.
2. To retrieve a guide, run `mere.run guide --model image-klein-9b --json`.
   Replace `image-klein-9b` with an ID from the index.

The `--json` option returns topic metadata and Markdown content. For a command
cookbook, use the command path, such as `mere.run guide image generate --json`.
Configure skills to retrieve bundled guides rather than duplicate their content.
