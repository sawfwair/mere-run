# Companion Plugins

Extend `mere.run` without letting anything into its process. Official plugins
are separate executables shipped outside the Swift package: they can add
operational integrations and typed workflow-graph nodes, but they run
out-of-process and speak one fixed protocol. A plugin manifest cannot inject a
shell command into your workflow bundle.

## Discover plugins

The CLI reads the public plugin catalog and can return either a human-readable
list or machine-readable JSON:

```bash
mere.run plugin list
mere.run plugin list --json
mere.run plugin info mere-runpod
```

Use `--catalog-url` with a local JSON file when developing or reviewing a
catalog change. The default catalog remains the public release source.

## Install safely

`plugin install` is a dry run unless `--yes` is present. Inspect the exact
package-manager command first, then opt in to execution:

```bash
mere.run plugin install mere-runpod
mere.run plugin install mere-runpod --yes
```

After installation, mere.run verifies the plugin manifest and entrypoint. A
plugin that declares a graph provider is registered only after that manifest
passes validation.

Use `--channel` to select a non-default catalog channel and `--force` only when
you intentionally want to forward a forced reinstall to the package manager.

## Diagnose an installation

```bash
mere.run plugin doctor mere-runpod
```

The doctor command resolves the catalog entry, verifies that its executable is
on `PATH`, and invokes the plugin's fixed `doctor` verb. Plugin stdout and exit
status remain owned by that executable.

## Workflow-provider boundary

A graph-provider plugin exposes a versioned manifest and only three graph
verbs: catalog, preflight, and execute. The core runtime validates typed
requests and events; plugin manifests cannot inject arbitrary shell commands
into workflow bundles.

See [Portable Workflows](./workflows.md#graph-v2-runtime-and-v1-contract) for
the graph ABI and [CLI Reference](./cli.md) for every plugin option.

## Source boundary

- The core repo owns catalog discovery, installation verification, and the
  typed plugin protocol.
- Plugin packages and their implementation dependencies live in the separate
  `mere-run-plugins` repository.
- Hosted-service, billing, and private deployment behavior do not belong in
  this public package.
