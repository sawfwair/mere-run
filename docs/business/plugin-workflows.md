# Set up companion workflows

Use companion plugins to turn local inference into a repeatable task: collect
research, review an invoice, prepare an edit, or compare a trained adapter with
its base model. `mere.run` runs the models. Each plugin is a separate executable
that coordinates commands and records artifacts.

## Check your Mac and workflow

Start with an Apple Silicon Mac and [install mere.run](../getting-started.md).
Check runtime support before downloading a model:

```bash
mere.run model capabilities
mere.run plugin list
mere.run plugin info mere-doc-tools
```

A plugin installation doesn't include every model or external tool its workflows
can use. Read your selected guide for model IDs, memory requirements, FFmpeg,
Pi, service access, or converted checkpoints. Model terms apply separately from
the plugin's license.

## Install and check a plugin

Preview installation, then confirm it and check readiness:

```bash
mere.run plugin install mere-doc-tools
mere.run plugin install mere-doc-tools --yes
mere.run plugin doctor mere-doc-tools
```

On supported Apple Silicon Macs with macOS 15 or later, catalog entries offer
signed, notarized bundles with their own Python runtime. You don't need pipx
for that installation. To choose the source package, install
[pipx](https://pipx.pypa.io/) and add `--source` to the install command.

Launch commands through the managed entrypoint. This works for signed plugins
and source-installed executables on your `PATH`:

```bash
mere.run plugin run mere-doc-tools -- manifest --json
mere.run plugin run mere-doc-tools -- doctor --extractor anydoc --no-redact
```

In Studio, open **Plugins** to inspect catalog entries, install or update a
plugin, and run its doctor. A basic doctor checks dependencies; it doesn't prove
that every model workflow is ready. Use the
[workflow prerequisites](https://plugins-docs.mere.run/guide/check-readiness)
and your plan to check the selected task.

## Keep a small first run

Choose one approved input and a private output directory outside the input
folder. Inspect the plan when the workflow offers one, run the task, and check
the output against the source. Keep `run.json` with its artifacts. Resume
behavior differs by plugin: a command might continue execution, reuse a
collection, or only inspect recorded state.

Before a larger batch, measure elapsed time and disk use. Check missing inputs,
model failures, interruption, and source changes. A valid manifest or hash
proves identity and structure, not the quality of extracted or generated content.

## Update or recover an installation

To install the catalog's signed release again, use the same install command.
Preview a rollback to a retained verified bundle before confirming it:

```bash
mere.run plugin rollback mere-doc-tools
mere.run plugin rollback mere-doc-tools --yes
```

Some commands share a package. Removing Workflow Tools can remove Document
Tools, Transcript Tools, Audio Post, and other commands together. Before
switching from a signed bundle to pipx, follow the
[installation and rollback guide](../plugins.md).

Choose a task from [Business use cases](./index.md). Keep source data, raw
outputs, and run records under your team's access and retention rules.
