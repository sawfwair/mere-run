# Plugin

## Purpose

Discover and install official companion plugins without putting provider-specific
runtime code into the core `mere.run` CLI. Plugins are separate executables that
use the user's own accounts and credentials for workflows such as remote GPU
training.

## Required Models

No model is required to list or install plugins. Individual plugin recipes may
require models, datasets, credentials, or external compute.

## Install And Check

```bash
mere.run plugin list
mere.run plugin info mere-runpod
mere.run plugin install mere-runpod
mere.run plugin install mere-runpod --yes
mere.run plugin doctor mere-runpod
mere.run guide plugin
```

## Parameters

- `--catalog-url`: plugin catalog URL or local JSON path. Defaults to the live
  official catalog in `sawfwair/mere-run-plugins`.
- `--json`: print the catalog or plugin entry as JSON for scripts.
- `--channel`: install channel, defaulting to the catalog's default channel.
- `--yes`: execute the install command. Without it, `install` prints the exact
  command only.
- `--force`: pass `--force` to `pipx install`.

## Usage Patterns

- Use `plugin list` to see the current official catalog and the install command
  each plugin declares.
- Use `plugin info <id>` before installing to inspect repository, package,
  entrypoint, capabilities, and channel-specific install spec.
- Use `plugin install <id>` first when you want a dry-run command preview.
- Add `--yes` when you want `mere.run` to run the catalog's `pipx install`
  command and verify `<plugin> manifest --json`.
- Use `plugin doctor <id>` after install to run the plugin's own environment
  checks.
- Use `--catalog-url ./plugins.v1.json` while developing or testing a catalog
  before it is published.

## Examples

```bash
mere.run plugin list
```

```bash
mere.run plugin info mere-runpod
```

```bash
mere.run plugin install mere-runpod --yes
mere.run plugin doctor mere-runpod
```

```bash
mere.run plugin list \
  --catalog-url https://raw.githubusercontent.com/sawfwair/mere-run-plugins/main/catalog/plugins.v1.json \
  --json
```

## Iteration Tips

- The catalog is data, not dynamic code. `plugin list` and `plugin info` only
  fetch and decode JSON.
- `plugin install` uses `pipx`, so installed plugin CLIs live outside the
  `mere.run` Swift package and can be updated independently.
- A plugin must report the `mere.run/plugin.v1` manifest contract before
  `plugin install --yes` is considered verified.
- Provider credentials stay with the plugin and the user's environment; the core
  CLI only resolves catalog metadata and launches the companion executable.

## Troubleshooting

- `pipx is required`: install it with `brew install pipx`, then retry.
- `pipx cannot force-reinstall the uv environment`: update pipx to 1.16.0 or
  later using its original installer, then retry the same command. For a
  Homebrew installation, run `brew upgrade pipx`. The compatibility check
  leaves the plugin unchanged; switching pipx backends is not required.
- `Unknown plugin`: rerun `mere.run plugin list` or check the selected
  `--catalog-url`.
- `Executable not found on PATH`: ensure `pipx ensurepath` has been run and
  your shell sees the plugin entrypoint.
- `Editable source path no longer exists`: the plugin was installed from a
  local checkout that moved or was removed. Run the exact
  `mere.run plugin install <id> --yes --force` repair command printed by
  `plugin list` or `plugin doctor`.
- Manifest mismatch: the installed command is not the cataloged plugin. Reinstall
  with `--force` or inspect your PATH ordering.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/PluginCommand.swift
- https://github.com/sawfwair/mere-run-plugins/blob/main/catalog/plugins.v1.json

## Signed bundles

Advertised macOS Apple Silicon bundles include their own runtime. Installation
verifies the publisher, artifact hash, Developer ID, and notarization before
activation. Existing pipx installations and PATH remain unchanged.

- Run a managed plugin: `mere.run plugin run mere-doc-tools -- COMMAND ARGS`
- Restore the previous shared package: `mere.run plugin rollback mere-doc-tools --yes`
- Explicitly choose source installation: `mere.run plugin install mere-doc-tools --source --yes`

Invalid signatures stop installation. `--force` does not bypass verification.
Offline installation accepts `--bundle-manifest RELEASE_JSON` and
`--bundle-archive BUNDLE_DMG` with the same verification requirements.
