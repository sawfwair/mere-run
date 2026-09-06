# Models, storage, and plugins

## Separate support, installation, and readiness

Use live inventory instead of a copied model list:

```bash
mere.run model capabilities --all --json
mere.run model list --json
mere.run model location list --json
mere.run model storage --json
```

Hardware support does not prove a checkpoint is installed. A manifest or
installed path does not prove runtime-ready payloads exist. Use the model-pull
preflight, command preflight, and `model info` together. Check conversion
requirements, external bindings, companion models, and component paths when a
model appears installed but cannot load.

Prefer a supported model suitable for the user's latency, quality, and storage
constraints. Treat managed IDs, API engine names, runtime aliases, local roots,
GGUF files, and Hugging Face repository IDs as different values. Read the leaf
command's accepted model input before using one. `--model-root` is not a global
synonym for `--model`.

Some runtime paths auto-download; others require explicit assets. Preflight
and pull deliberately before expensive work. Use `text chat --require-installed`
when its implicit downloads must be prevented. Do not recommend `model pull
--all` as normal setup, or promise every catalog entry has a download source.

## Model terms and authentication

`model pull --preflight --json` reports component `usage_terms` and whether they
are acknowledged. Some gated or restricted models require
`--accept-model-license` (alias `--accept-license-terms`). That flag represents
acceptance of the listed terms; download authorization alone does not establish
acceptance. Reuse explicit acceptance already given for those terms. Otherwise,
show the actual terms and obtain the missing acceptance before passing the flag.

A Hugging Face access token and license acceptance solve different blockers.
Check the reported source and access requirements. `config get` and `config
list` mask secrets; `config set --from-env` avoids putting a token value directly
in argv. `HF_TOKEN`, then `HUGGING_FACE_HUB_TOKEN`, override persisted token
configuration. Do not print token values or record them in a shared report.

`--allow-unsupported` bypasses hardware checks, not missing runtime support or
license terms. Use it only for an explicitly intended unsupported experiment.
`--force` is command-specific: it can re-download, reinstall, or delete. Read
help and the planned effect instead of treating it as a generic repair option.

## Storage and cache ownership

The default writable store is
`~/Library/Application Support/MereRun/models`. `--models-root` or
`MERERUN_MODELS_DIR` selects another store. Registered search roots and explicit
bindings can provide models elsewhere; inspect `model location list --json`.

`MERERUN_HUB_CACHE` selects the Hub cache. For a specific pull, `--cache-dir`
selects its cache and is reflected in the preflight. Model-store entries can
link to cache payloads. Disconnecting the cache volume can make a model
unavailable even while its install entry exists. Changing the store does not
move the cache or repair a disconnected binding.

Inspect disk usage with `model storage --json` before moving or removing data.
`model gc --json` is a read-only plan; `--force` deletes a freshly computed plan.
`model remove --json` requires `--force`; it is **not** a read-only preview.
`--keep-cache` retains unshared Hub payloads when removing model links.

Check sharing, external roots, and active runs before removal. A successful
preflight or an apparently duplicate directory does not establish that files
are disposable. Use documented location/binding commands for existing payloads;
do not manufacture manifests or move caches blindly. Read the help before
repairing manifests; that command can expose a dry-run mode.

## Adapters and training checkpoints

Use `adapter list --json` to select cataloged adapters. Read `adapter pull
--help` for its license flag and source requirements; adapter acceptance flags
are not necessarily spelled like model acceptance flags. Cataloged downloads
verify byte count and SHA-256 before installation.

Match the adapter's base family, quantization, target modules, and required
sampling schedule. A turbo adapter can require a specific step count or CFG.
A `.safetensors` extension alone does not prove compatibility. Use the matching
model handbook and cookbook before adding `--lora` or a family-specific adapter
option. Preserve the adapter ID or path and scale in preflight and execution.

## Companion plugins

These are out-of-process `mere.run` executables, separate from Codex skills and
Codex plugins. Discover the catalog and local verification state:

```bash
mere.run plugin list --json
mere.run plugin info mere-doc-tools --json
```

`plugin install ID` previews installation; `--yes` executes it. Inspect the
selected channel, artifact, entrypoint, and install mode before installation.
When the channel advertises a signed macOS bundle, the CLI verifies signatures,
hashes, notarization, manifests, and provider contracts. A source channel can
require pipx and Python. `--source` deliberately chooses that lane; do not fall
back to it to bypass a failed bundle verification.

After installation, use `plugin doctor ID`. For managed bundles, invoke
`plugin run ENTRYPOINT -- ...` or a registered graph node; a bare PATH command
can still select an older source installation. Plugin arguments after `--`
belong to the plugin, so inspect its own help rather than assuming core flags.

A bundle rollback can affect multiple entrypoints in the same package. Inspect
`plugin rollback --help` and the shared package before applying it. Preserve
unrelated PATH installs and use the reported repair procedure for a broken
source environment instead of repeatedly reinstalling it.
