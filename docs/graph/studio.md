# Graph Studio

Build workflows on a canvas instead of by hand in JSON. Mere Graph Studio is
the visual authoring and operations surface for the current `mere.run` Graph v2
runtime — either the hosted app at
[studio.mere.run](https://studio.mere.run/) or the cross-platform Tauri 2
desktop build for macOS, Windows, and Linux.

Studio is a client, not a second engine. It authors exactly the typed graph,
inputs, and immutable job bundle that the CLI validates and executes locally,
over SSH, or through Relay — so anything you draw there runs headless
everywhere else, unchanged.

## Graph v2 versus `schema_version: 1`

Two version labels describe different layers:

| Label | Meaning |
| --- | --- |
| Graph v2 | The current runtime generation: catalog-driven nodes, immutable bundles, preflight, parallel and resumable execution, providers, Relay placement, and typed artifacts. |
| `schema_version: 1` | The stable serialized `mere.run/workflow-graph` ABI consumed by every current executor. |

Do not change a workflow to `schema_version: 2`. Graph v2 intentionally keeps
the version-1 document contract so the CLI, desktop Studio, hosted Studio,
Relay, and paired Nodes execute the same immutable bundle bytes.

## Choose a Studio surface

### Hosted Studio

[Open Graph Studio](https://studio.mere.run/app) in a modern browser and sign in
with Mere World. Projects are private to the signed-in account. The site loads
the live node catalog reported by the user's paired fleet, performs placement
preflight, creates an immutable job, and submits it through Relay.

The hosted site does not run models. Relay owns admission, scheduling, leases,
retry state, and artifact delivery; the paired Node owns model execution. A
stale worker cannot settle a newer attempt.

Hosted execution currently accepts graphs whose inputs are JSON values. Use the
desktop app for workflows that bind local files or directories; browser asset
upload is a separate transport boundary and is not inferred from a machine-local
path.

### Desktop Studio

The desktop app packages the React/XYFlow editor in a Rust and Tauri 2 shell.
It is designed for:

- macOS, Windows, and Linux native installation;
- local file and directory inputs;
- local, SSH, and Relay execution;
- offline authoring and local execution;
- native project and run workspaces.

Desktop Studio requires a compatible `mere.run` executable for local execution.
Optional workflow tools add templates, programs, and conservative ComfyUI
prompt import; core graph authoring remains available without them.

The desktop product has one native backend: the frontend invokes Rust through
Tauri IPC, and Rust invokes the installed CLI. It does not ship a Python
runtime, browser compatibility server, or loopback HTTP API.

Native packaging is implemented for each platform. Public installer links are
published from [studio.mere.run](https://studio.mere.run/) only after the
matching platform artifact has passed its signing and package-verification
gate; the hosted app remains available without a desktop install.

## Authentication boundary

Local Studio never requires Mere World sign-in. Launch, authoring, project
save and load, catalog discovery, validation, preflight, and local execution
remain available without an account or network connection when the required
CLI and model assets are already installed.

Identity is introduced only when Studio crosses an account or fleet boundary:

| Surface | Authentication |
| --- | --- |
| Desktop local | None. Tauri IPC stays on the machine. |
| Desktop SSH | The configured SSH credentials for that target. |
| Desktop Relay | Relay credentials, requested only after Relay is selected. |
| Hosted Studio | Mere World OAuth Authorization Code with PKCE. |
| Mere Node joining Relay | OAuth device grant completed by the operator at `mere.world/device`. |

The Node device flow does not open a browser callback or local web server. The
Node requests a device code, shows the verification URL and user code, polls
Relay for completion, stores the resulting rotating credentials in its
platform configuration directory, and authenticates its Relay WebSocket with
a bearer token. Desktop local mode does none of this.

## The project model

Studio keeps editor state outside the executable graph:

```text
workflow graph   typed nodes, references, execution policy, outputs
inputs           values supplied to declared graph inputs
editor sidecar   node positions, groups, notes, selections, view state
program          optional higher-level map and branch authoring document
```

Import and export use a `.meregraph.json` project package to carry those
documents together. Execution still materializes the canonical
`mere.run/job-bundle.v1` files described in [Portable Workflows](../workflows.md).

## Run from the CLI

A Studio-authored workflow remains a normal CLI workflow:

```bash
mere.run graph validate workflow.json --inputs-json inputs.json --json
mere.run graph preflight workflow.json --inputs-json inputs.json --executor local --json
mere.run graph run workflow.json --inputs-json inputs.json --run-dir ./runs/studio --json
```

For a paired fleet, select the Relay executor in Studio or submit the same
materialized bundle with `mere.run graph submit-job`. Run reports, events, and
artifact hashes use the same contract in either client.

## Compatibility and drift prevention

Studio does not keep a private hard-coded execution schema. Desktop Studio
reads `mere.run graph catalog --json`; hosted Studio reads catalogs reported by
connected Nodes through Relay. The catalog carries node fields, requirements,
provider versions, and provider catalog digests, so an unavailable or outdated
node is rejected during authoring or preflight instead of failing after queueing.

The `mere.run` repository keeps its CLI and documentation synchronized through
`DocumentationContractTests`. Graph Studio, Relay, and Node maintain shared
contract fixtures and run their own local gates before release. Studio's gate
also rejects a returning Python host, browser proxy, or authentication
dependency in the native runtime.
