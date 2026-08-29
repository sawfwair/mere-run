# Companion plugins

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

`plugin list --json` preserves the catalog fields and adds local inspection
state for each entry: whether the entrypoint is installed, whether its manifest
verified, the installed version and path, any verification error, and the
default-channel install command. This is the source of truth for thin clients;
they do not maintain a second plugin registry.

## macOS Studio

Open **Plugins** from the Studio sidebar or **View → Plugins**. The first-class
workspace supports catalog search, installed and verified status, channel
selection, repository access, copyable install commands, confirmed install or
update, and the fixed plugin doctor operation. A custom catalog URL or local
JSON path is available for development.

The UI calls the same `plugin list`, `plugin install`, and `plugin doctor`
commands documented here. Plugins remain separate executables and no plugin
implementation is loaded into the app process.

## Install safely

`plugin install` is a dry run unless `--yes` is present. Inspect the exact
package-manager command first, then opt in to execution:

```bash
mere.run plugin install mere-runpod
mere.run plugin install mere-runpod --yes
```

After installation, `mere.run` verifies the plugin manifest and entrypoint. A
plugin that declares a graph provider is registered only after that manifest
passes validation.

Use `--channel` to select a non-default catalog channel and `--force` only when
you intentionally want to forward a forced reinstall to the package manager.

### Update or repair a plugin

To replace a plugin with the selected catalog version, preview the command and
then confirm it:

```bash
mere.run plugin install mere-doc-tools --force
mere.run plugin install mere-doc-tools --yes --force
```

If the plugin uses a pipx environment managed by uv, forced reinstalls require
pipx 1.16.0 or later. `mere.run` checks that environment's recorded backend and
the pipx version before starting the reinstall. An affected version produces
upgrade instructions and leaves the plugin unchanged. This check does not
require you to switch backends, and it does not block fresh installations or
environments managed by pip.

If you installed pipx with Homebrew, update it and retry:

```bash
brew upgrade pipx
mere.run plugin install mere-doc-tools --yes --force
```

For other installation methods, update pipx with the installer you used to
install it. See the [pipx installation guide](https://pipx.pypa.io/latest/how-to/install-pipx.html).
Pipx 1.16.0 includes the fix for
[forced reinstalls in uv environments](https://github.com/pypa/pipx/issues/1924).

## Diagnose an installation

```bash
mere.run plugin doctor mere-runpod
```

The doctor command resolves the catalog entry, verifies that its executable is
on `PATH`, verifies its manifest, and invokes the plugin's fixed `doctor` verb.
If the entrypoint cannot start, `mere.run` diagnoses the installation before
delegating. A stale editable `pipx` install reports its missing source path and
the exact forced-reinstall command instead of exposing only a Python traceback.
Plugin stdout and exit status remain owned by a verified executable.

## Workflow-provider boundary

A graph-provider plugin exposes a versioned manifest and only three graph
verbs: catalog, preflight, and execute. The core runtime validates typed
requests and events; plugin manifests cannot inject arbitrary shell commands
into workflow bundles.

See [Portable workflows](./workflows.md#graph-v2-runtime-and-v1-contract) for
the graph ABI and [CLI reference](./cli.md) for every plugin option.

## Source boundary

- The core repo owns catalog discovery, installation verification, and the
  typed plugin protocol.
- Plugin packages and their implementation dependencies live in the separate
  `mere-run-plugins` repository.
- Hosted-service, billing, and private deployment behavior do not belong in
  this public package.

## Signed bundle pilot

The signed bundle installer supports official macOS Apple Silicon releases on
macOS 15 or later. The public catalog continues to use pipx until verified
bundle artifacts and a compatible CLI release are published.

When a channel advertises a bundle, `plugin install` verifies the publisher
signature, exact download hash, Developer ID, and notarization. It checks the
plugin manifests and graph providers before switching the active version.
Installation does not require pipx, uv, Homebrew, a user-installed Python, or
Xcode. The CLI verifies the app with macOS's built-in `codesign`,
`syspolicy_check`, and Gatekeeper tools.

```bash
mere.run plugin install mere-doc-tools --yes
mere.run plugin run mere-doc-tools -- process --input source.csv --output-dir output --extractor anydoc --no-redact
mere.run plugin rollback mere-doc-tools --yes
```

Bundles are stored in the MereRun application support directory. A package's
entrypoints share one private runtime. Existing pipx environments and PATH
commands are not removed or rewritten. Use `plugin run` or native graph
workflows to select the managed bundle; a bare PATH command can still refer
to an existing pipx installation. Rollback affects all plugins in the shared
package and revalidates the retained version before activation.

To explicitly select source installation, use `--source`. A signature,
notarization, compatibility, or download failure never falls back to pipx.
`--force` does not bypass bundle verification or permit a downloaded downgrade.
No unsigned bundle mode is provided.

While a signed bundle is active, `plugin install --source` refuses to replace
it. Its existing pipx entrypoints are still available directly on PATH. This
pilot supports rollback between retained bundles, not automatic migration
back to pipx.

For an offline release review, supply the catalog, signed envelope, and DMG:

```bash
mere.run plugin install mere-doc-tools --catalog-url CATALOG_JSON \
  --bundle-manifest RELEASE_JSON --bundle-archive BUNDLE_DMG --yes
```

These options do not change trust: the CLI still requires a known publisher
key, a matching artifact hash, and valid macOS signatures and notarization.
The catalog cannot introduce a new trusted key. Release metadata expires for
new installations; existing installations continue to work offline. Signing
proves publisher identity and integrity, not sandboxing or code safety.
