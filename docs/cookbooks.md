# Cookbooks

This page is for mere.run Studio and command-line users. Use the bundled
cookbooks to prepare prompts, choose inputs, and review command options offline.
For source research and recipe status, see the
[provider prompting guide tracker](./provider-prompting-guides.md).

## Read a model guide

The handbook contains original recipes for 139 managed model IDs, organized
into 59 families. Each guide includes an example, supported controls, variant
notes, review advice, and sources. Models without a text-prompt interface have
input-preparation guidance.

To read a model guide in macOS Studio, follow these steps:

1. In **Help**, select **mere.run Guide**.
2. In **Guide collection**, select **Models**.
3. In the search field, enter a family name or model ID.
4. Select the guide.
5. In **Model**, select a variant.

To read a model guide in a terminal, run the following command:

```bash
mere.run guide --model image-klein-9b
```

The guide text, examples, and index are bundled with the CLI and macOS app.
Reading guides requires neither a network connection nor downloaded model
weights. Source links require a network connection. Before running inference
offline, download the checkpoint and the components required by your workflow.

These recipes are editorial drafts. They aren't records of successful
inference for every checkpoint. Guides distinguish provider material from
local workflow advice when provider guidance remains unverified.

## Read a command cookbook

The **Commands** collection contains command cookbooks. Each cookbook describes
required models, options, prompting patterns, examples, and troubleshooting.

To read the image generation cookbook in a terminal, run the following command:

```bash
mere.run guide image generate
```

To list the available command topics, run the following command:

```bash
mere.run guide --list
```

To save a Markdown index of command topics, run the following command:

```bash
mere.run guide --list --markdown > guides.md
```

## Available topics

The following topics cover creative and runtime workflows:

- `image generate`
- `image train-lora`
- `image validate`
- `text chat`
- `text code`
- `text embed`
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
- `music analyze`
- `music generate`
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

The following topics cover operational workflows:

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
- `agent install-pi`
- `agent start`
- `agent status`

## Read a command guide for a model

Some command cookbooks include model-specific advice. To read the image
cookbook for Z-Image nano, run the following command:

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
