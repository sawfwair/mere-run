# Portable Workflows

`mere.run` workflows are immutable, headless job graphs that use the same
contract locally, over SSH, and through a relay-managed GPU fleet. A future
canvas can author this document, but it does not own execution behavior.

## Graph V1

Every graph uses `schema_version: 1` and `kind: "mere.run/workflow-graph"`.
Identifiers match `[a-z][a-z0-9-]{0,63}`. Supported graph input types are
`string`, `integer`, `number`, `boolean`, `enum`, `asset`, and
`asset_directory`.

References are complete JSON values:

```json
{
  "$ref": "nodes.train-style.outputs.adapter"
}
```

References add dependency edges automatically. `depends_on` adds ordering-only
edges. Validation blocks cycles, duplicate IDs, missing references, unknown node
kinds, type mismatches, and incompatible artifact content types.

The V1 node catalog contains:

- `image.train-lora`
- `image.generate`
- `video.generate`

Inspect the exact typed fields and output descriptors with:

```bash
mere.run graph catalog --json
```

Canvas coordinates and editor state do not belong in the execution graph. Keep
them in a separate sidecar keyed by graph and node IDs.

## Validate and preflight

Point discovery at a broad project data root to rank image-caption directories
without loading a model or hashing the entire tree:

```bash
mere.run graph dataset discover /path/to/project/data --json
```

```bash
mere.run graph validate workflow.json --inputs-json inputs.json --json
mere.run graph preflight workflow.json --inputs-json inputs.json --executor local --json
mere.run graph preflight workflow.json --inputs-json inputs.json --executor ssh:gpu-box --json
mere.run graph preflight workflow.json --inputs-json inputs.json --executor relay:fleet --json
```

Validation is executor-independent. Preflight combines graph diagnostics with a
worker capability probe, including contract version, node kinds, installed
models, accelerator backend, memory, and available disk. Missing models produce
declarative pull actions; V1 never pulls models automatically on a worker.

## Portable job bundles

Materialization resolves defaults and random seeds, fingerprints the canonical
graph and inputs, and copies declared file inputs into a content-addressed store.

```bash
mere.run graph materialize workflow.json \
  --inputs-json inputs.json \
  --run-dir ./runs/job \
  --json

mere.run graph export-job workflow.json \
  --inputs-json inputs.json \
  --output ./job-bundle \
  --json
```

Both commands produce:

```text
job.json
graph.json
inputs.json
assets.json
assets/sha256/<digest>
```

The bundle contains relative paths only. Directory entries are ordered and
content-addressed. Symlinks, device files, path traversal, and files escaping a
declared directory root are rejected. Executor profiles, tokens, URLs, and
machine paths never enter the bundle.

## Local execution

```bash
mere.run graph run workflow.json \
  --inputs-json inputs.json \
  --run-dir ./runs/job \
  --json

mere.run graph run workflow.json \
  --inputs-json inputs.json \
  --run-dir ./runs/job \
  --resume \
  --json
```

Nodes run sequentially in stable topological order and fail fast. Every node is
adapted to deterministic public CLI arguments and launched as an isolated
`mere.run` child process. The runner captures structured stdout, preserves
diagnostic stderr, records exit status, verifies every declared artifact, and
honors a cooperative cancellation marker.

`--resume` reuses a finished node only when its fingerprint and every artifact
size and SHA-256 still match.

## Run directories

Local, SSH, and relay execution converge on:

```text
run.json
graph.json
inputs.json
job.json
assets.json
actions.json
events.jsonl
nodes/<ordinal>-<node-id>/
outputs/
```

Run and node states are `planned`, `preflighting`, `queued`, `assigned`,
`running`, `finished`, `failed`, and `cancelled`. Final outputs are hard-linked
into `outputs/` when possible and copied otherwise. Artifact paths in reports
are relative to the run directory.

```bash
mere.run run inspect ./runs/job --json
```

## Worker protocol

The machine protocol is public and is shared by SSH and relay nodes:

```bash
mere.run graph worker probe --json
mere.run graph worker execute --bundle PATH --run-dir PATH --json-stream
mere.run graph worker inspect --run-dir PATH --json
mere.run graph worker cancel --run-dir PATH --json
```

`execute --json-stream` emits the run event model as NDJSON on stdout. Progress
and diagnostics stay on stderr.

The versioned contracts are published under `docs/public/schemas`, including the
[Graph Event V1 schema](./public/schemas/graph-event-v1.schema.json), workflow
graph, inputs, asset manifest, job bundle, worker probe, and graph run schemas.
Relay and node implementations consume the same canonical fixtures used by the
Swift tests.

## Executor profiles

Profiles are stored in the application-support directory as `executors.json`
with mode `0600`.

```bash
mere.run executor add ssh gpu-box \
  --destination user@host \
  --remote-root ~/.cache/mere.run

mere.run executor add relay fleet \
  --url https://relay.example.com

mere.run executor list --json
mere.run executor probe ssh:gpu-box --json
mere.run executor inspect relay:fleet --json
mere.run executor login relay:fleet --json
mere.run executor auth-status relay:fleet --json
mere.run executor fleet relay:fleet --json
mere.run executor node-refresh relay:fleet <device-id> --json
mere.run executor node-configure relay:fleet <device-id> --drain --json
mere.run executor logout relay:fleet --json
mere.run executor remove gpu-box --json
```

Relay profiles require an explicit URL. `executor login` discovers the relay's
mere.world device-authorization contract, prints the approval URL and code to
stderr, and stores the resulting access and refresh token set in a `0600` file.
Relay requests refresh expiring credentials automatically, serialize refreshes
across concurrent CLI processes, persist rotated refresh tokens atomically, and
retry once after an authenticated `401`. Raw token files and
`MERERUN_RELAY_TOKEN` remain supported for automation.

`executor fleet` reports stable device identity, scheduling status, graph-worker
capability, installed models, and recent graph and modality history.
`node-refresh` asks a connected node to rescan hardware, runtime, models,
plugins, disk, and graph support without reconnecting. `node-configure` exposes
headless enable, disable, drain, undrain, priority, and display-name controls.

## SSH execution

```bash
mere.run graph submit workflow.json \
  --inputs-json inputs.json \
  --executor ssh:gpu-box \
  --run-dir ./runs/job \
  --json
```

SSH uses the system `ssh` and `scp`, `BatchMode=yes`, and normal host-key
verification. It probes the remote worker, checks contract and model support,
queries a separate content-addressed cache, uploads only missing hashes, verifies
size and SHA-256 remotely, links cached assets into the immutable job directory,
and launches a fixed worker script detached.

## Relay execution

```bash
mere.run graph submit workflow.json \
  --inputs-json inputs.json \
  --executor relay:fleet \
  --run-dir ./runs/job \
  --json
```

Relay creation is two-phase: create the manifest, upload only missing hashes,
then commit. The relay verifies each digest before queueing and places the whole
graph on one compatible worker. Intermediate artifacts remain on that worker.
The node forwards the public worker NDJSON rather than creating another event
format. Large result artifacts are transferred as independently verified parts,
stored by full content digest, and streamed back as one artifact; the local
fetch still verifies the complete size and SHA-256 before materializing it.
Interrupted multipart uploads resume from relay-verified parts after retry.
Each job reports download, execution, upload, total timing, transferred bytes,
and reused bytes and parts.

## Remote lifecycle

```bash
mere.run run inspect relay://fleet/<job-id> --json
mere.run run watch relay://fleet/<job-id> --json-stream
mere.run run fetch relay://fleet/<job-id> --into ./runs/job --json
mere.run run fetch relay://fleet/<job-id> --into ./runs/job --all-artifacts --json
mere.run run cancel relay://fleet/<job-id> --json
mere.run run retry relay://fleet/<job-id> --json
mere.run run list --executor relay:fleet --limit 50 --json
```

Fetch verifies artifact size and SHA-256 and materializes the same local
run-directory shape. The default fetch includes reports, manifests, and final
outputs; `--all-artifacts` also includes node artifacts. Explicit relay retry
uses the same immutable bundle and resolved seeds.

Queued relay jobs include a typed placement report in `run inspect --json`.
It identifies every connected device and reports concrete blockers such as a
legacy worker, fleet policy, busy state, contract or worker version, missing
node kind or model, accelerator backend or memory, and available disk.

## Schemas

- [Workflow Graph V1](/schemas/workflow-graph-v1.schema.json)
- [Workflow Inputs V1](/schemas/workflow-inputs-v1.schema.json)
- [Asset Manifest V1](/schemas/asset-manifest-v1.schema.json)
- [Job Bundle V1](/schemas/job-bundle-v1.schema.json)
- [Graph Run V1](/schemas/graph-run-v1.schema.json)
- [Worker Probe V1](/schemas/worker-probe-v1.schema.json)
