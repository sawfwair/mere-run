# Portable workflows

Describe a job once as a typed graph. The same bundle then runs on your laptop,
over Secure Shell (SSH) on a graphics processing unit (GPU) host, or across a
relay fleet with the same resolved seeds, fingerprints, and execution contract.

Workflows are immutable and headless. Execution-location details do not enter
the bundle: executor profiles, tokens, URLs, and machine paths remain external.
bundle. Visual editors such as [Graph Studio](./graph/studio.md) author these
documents, but execution behavior is defined here, not on the canvas.

## Graph v2 runtime and v1 contract

The runtime and serialized workflow contract use separate version numbers.

**Graph v2** is the runtime generation with immutable job bundles, typed
provider catalogs, executor preflight, resumable runs, parallel scheduling, and
one execution contract shared by local, SSH, and Relay.

The serialized workflow ABI is a separate thing, and it stays deliberately
pinned at version 1. Graph v2 still reads and writes `mere.run/workflow-graph`
documents with `schema_version: 1`. The runtime name is not a request to emit a
`schema_version: 2` document.

Every graph uses `schema_version: 1` and `kind: "mere.run/workflow-graph"`.
Identifiers match `[a-z][a-z0-9-]{0,63}`. Supported graph input types are
`string`, `integer`, `number`, `boolean`, `enum`, `json`, `asset`,
`asset_directory`, and `asset_collection`.

References are complete JSON values:

```json
{
  "$ref": "nodes.train-style.outputs.adapter"
}
```

References add dependency edges automatically. `depends_on` adds ordering-only
edges. Validation blocks cycles, duplicate IDs, missing references, unknown node
kinds, type mismatches, and incompatible artifact content types.

Independent ready nodes may run concurrently when the graph declares
`execution.max_parallel_nodes` from 1 through 32. The default remains one, and
all scheduling, event commits, and final outputs retain stable topological order.
Each node may independently set retry, timeout, and cache policy.

The version 1 node catalog contains:

- `text.value`, `integer.value`, `number.value`, `boolean.value`, `json.value`
- `seed.value`, `choice.value`
- `text.join`, `text.template`
- `text.enhance`, `image.describe`
- `vision.ground`, `vision.segment`, `vision.track`
- `image.train-lora`
- `image.generate`
- `video.generate`
- `music.generate`, `sfx.generate`, `music.separate`, `music.transcribe`
- `speech.synthesize`, `speech.transcribe`, `speech.diarize`
- `vision.caption`, `vision.ocr`, `vision.geometry`, `vision.image-to-3d`
- `audio.enhance`
- `text.generate`, `text.embed`, `text.anonymize`

The value and text nodes make authored creative material reusable without
turning it into a runtime graph input. Literal values, seeds, joins, and
templates execute as native deterministic intrinsics. `text.template` accepts
strict `{{name}}` identifiers, preserves escaped `\{{name}}` text, and blocks
missing variables. `text.enhance` and `image.describe` require an explicit
installed model and never trigger a hidden pull.

`vision.ground` uses the managed `vision-ground-falcon-perception` model by
default. It emits an annotated image, structured detections JSON, and a
portable directory of per-detection PNG masks as verified artifacts. Grounding
output is candidate geometry; graph execution does not promote it into evidence
or findings.

`vision.segment` and `vision.track` use the managed `vision-segment-sam31`
model by default. Both emit an annotated media artifact, structured JSON, and a
portable directory of PNG masks. They refine or propagate candidate geometry;
their masks remain review candidates until a mission workflow corroborates and
accepts them.

Catalog value schemas recursively describe array items, object properties, and
additional properties. Editors may expose those nested slots as typed ports
while the serialized graph continues to use ordinary nested JSON references.

Companion plugins may add typed nodes without adding arbitrary command or
Python execution to the graph ABI. Plugin nodes name their provider explicitly:

```json
{
  "id": "prepare",
  "kind": "dataset.prepare",
  "provider": "mere-dataset-tools",
  "arguments": {
    "data": {"$ref": "inputs.data"}
  }
}
```

Providers use one fixed machine protocol:

```bash
PLUGIN graph catalog --json
PLUGIN graph preflight --request invocation.json --run-dir PATH --json
PLUGIN graph execute --request invocation.json --run-dir PATH --json-stream
```

The catalog describes `string`, `integer`, `number`, `boolean`, `enum`,
`json`, `asset`, `asset_directory`, and `asset_collection` ports, including
defaults, ranges, content types, model and accelerator requirements, progress,
preview, determinism, caching, and side-effect traits. `mere.run` validates the
catalog and invokes only these fixed verbs. A plugin manifest cannot inject a
command template.

Providers may also declare minimum accelerator memory, system memory, free disk,
CPU cores, and network access. Secret fields accept only a named binding such as
`{"$secret":"hugging-face-token"}`. Values are resolved from
`MERERUN_SECRET_HUGGING_FACE_TOKEN` only inside the executing worker process;
secret values never enter graphs, bundles, fingerprints, reports, events, or
Relay payloads. Secret-bound nodes must disable caching.

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
models, exact provider versions and catalog digests, accelerator backend,
accelerator and system memory, CPU capacity, available disk, network access, and
named secret availability. Missing models produce declarative pull actions; V1
never pulls models automatically on a worker.

## Portable job bundles

Materialization records source graph and input fingerprints, resolves defaults
and random seeds, fingerprints the portable graph and inputs, and copies
declared file inputs into a content-addressed store. The optional source
fingerprints also flow into `run.json`, allowing authoring clients to display
results only when they match the source documents open in the editor.

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

An exported bundle can be executed directly without materializing the source
graph again. The bundle remains immutable and the mutable run record must use a
separate, non-nested directory:

```bash
mere.run graph run-job \
  --bundle ./job-bundle \
  --run-dir ./runs/local \
  --json

mere.run graph submit-job \
  --bundle ./job-bundle \
  --executor ssh:gpu-box \
  --run-dir ./runs/ssh \
  --json

mere.run graph submit-job \
  --bundle ./job-bundle \
  --executor relay:fleet \
  --run-dir ./runs/relay \
  --json
```

All four portable documents and every content-addressed asset are verified
before execution or transport. This is the acceptance path for proving that
local, SSH, and Relay consume the same resolved seeds, fingerprints, manifests,
and asset bytes.

The job pins each companion provider by ID, semantic version, catalog SHA-256,
and node kinds. It also records managed model repository, revision, and catalog
identity. At execution time the run record adds the installed model manifest
digest when one is available. Adapter and other asset inputs retain their full
content digests.

The bundle contains relative paths only. Directory entries are ordered and
content-addressed. Symlinks, device files, path traversal, and files escaping a
declared directory root are rejected. Executor profiles, tokens, URLs, and
machine paths never enter the bundle.

Bundles pin their minimum compatible `mere.run` worker version. Preflight and
submission reject older SSH or Relay workers before transfer or queueing; the
worker repeats the check before execution. Graphs using creative material nodes
require `mere.run` 0.24.0 or newer.

## Local execution

```bash
mere.run graph run workflow.json \
  --inputs-json inputs.json \
  --run-dir ./runs/job \
  --json

mere.run graph run-job \
  --bundle ./job-bundle \
  --run-dir ./runs/exported-job \
  --resume \
  --json

mere.run graph run workflow.json \
  --inputs-json inputs.json \
  --run-dir ./runs/job \
  --resume \
  --json
```

Nodes run in stable topological order and fail fast. Graphs are sequential by
default; an explicit graph parallelism limit allows independent ready nodes to
overlap without changing dependency semantics. Every node is
adapted to deterministic public CLI arguments and launched as an isolated
`mere.run` child process. The runner captures structured stdout, preserves
diagnostic stderr, records exit status, verifies every declared artifact, and
honors a cooperative cancellation marker.

Node execution policy supports bounded retries, hard subprocess timeouts, and a
cross-run content-addressed cache. Cache keys include normalized arguments,
provider identity, model provenance, and directly referenced artifact digests.
`cache: refresh` recomputes and replaces an entry; `cache: never` bypasses it.

`--resume` reuses a finished node only when its provider pin, normalized
arguments, exact model provenance, directly referenced upstream outputs, and
every output digest still match. Unrelated branches do not invalidate each
other. File, directory, collection, scalar, and JSON outputs retain their typed
identity in the node run record.

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

`execute --json-stream` emits the run event model as newline-delimited JSON
(NDJSON) on standard output. It
preserves node progress, preview, artifact, diagnostic, metric, heartbeat, and
lifecycle events. Diagnostics produced outside the machine protocol stay on
stderr.

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

mere.run graph submit-job \
  --bundle ./job-bundle \
  --executor ssh:gpu-box \
  --run-dir ./runs/exported-ssh-job \
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

mere.run graph submit-job \
  --bundle ./job-bundle \
  --executor relay:fleet \
  --run-dir ./runs/exported-relay-job \
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

## Direct relay (`relay serve`)

`mere.run relay serve` hosts the same client-facing graph-job HTTP surface
directly on a machine and runs submitted jobs there without a broker or cloud
hop. It is the single-machine, self-sovereign lane. Reach it over the local area network (LAN) or
a tailnet (Tailscale users can front it with `tailscale serve` for HTTPS),
and prompts, inputs, and outputs never leave your network.

```bash
# On the machine that will run the work:
mere.run relay serve
# The command prints the address and a six-digit pairing code.

# On any other machine, exchange the code for a bearer token:
curl -X POST http://192.0.2.10:6373/api/pair \
  -H 'Content-Type: application/json' \
  -d '{"code": "123-456", "device_name": "example-laptop"}'

# Save the token and add a normal relay executor profile:
mere.run executor add relay lab --url http://192.0.2.10:6373 --token-file ./lab-token
mere.run graph submit workflow.json --executor relay:lab --run-dir ./runs/job
```

The iOS app pairs through **Connect to a machine** on its pairing screen.
Because the API is byte-identical to the hosted relay's, every client —
`graph submit`, fetch, events polling, the phone — works unchanged against
either.

Pairing accepts new devices for a bounded window after startup (default 15
minutes, `--pairing-window`) and locks after ten failed codes; issued tokens
are stored hashed in the spool directory and survive restarts. Jobs are
validated against the machine's live capability probe at commit, execute
strictly serially through the same `WorkflowRunner` as every other lane
(events stream through `events.jsonl`, including `node_output_delta` for
`text.generate`), and artifacts are served with digest verification from the
run directory. Jobs interrupted by a shutdown come back as failed with an
explanatory error and can be retried.

## Remote lifecycle

```bash
mere.run run inspect relay://fleet/<job-id> --json
mere.run run watch relay://fleet/<job-id> --json-stream
mere.run run fetch relay://fleet/<job-id> --into ./runs/job --json
mere.run run fetch relay://fleet/<job-id> --into ./runs/job --artifact sample --artifact adapter --json
mere.run run fetch relay://fleet/<job-id> --into ./runs/job --all-artifacts --json
mere.run run cancel relay://fleet/<job-id> --json
mere.run run retry relay://fleet/<job-id> --json
mere.run run list --executor relay:fleet --limit 50 --json
```

Fetch verifies artifact size and SHA-256 and materializes the same local
run-directory shape. The default fetch includes reports, manifests, and final
outputs; `--all-artifacts` also includes node artifacts. Explicit relay retry
uses the same immutable bundle and resolved seeds. Repeat `--artifact` to fetch
only named artifacts; an already-present file with the declared size and
SHA-256 is reused, so retrying an interrupted fetch continues from verified
local results. `--artifact` and `--all-artifacts` are mutually exclusive.

Queued relay jobs include a typed placement report in `run inspect --json`.
It identifies every connected device and reports concrete blockers such as a
legacy worker, fleet policy, busy state, contract or worker version, missing
node kind or model, accelerator backend or memory, and available disk.

The macOS Studio app exposes a top-level **Runs & Operations** workspace. It
discovers local run folders and structured reports, lists recent Relay jobs,
polls inspection state, shows artifact inventories, reveals local runs, and
offers verified fetch, cancellation, and immutable Relay retry. The Advanced
forms remain available for raw arguments and JSON streaming.

Relay profile setup and device authorization also stay on the public
`executor add relay`, `executor auth-status`, and `executor login` contracts.
Studio streams the CLI's approval URL, opens it after you choose **Sign
In**, and never reads or stores bearer credentials itself.

The ownership boundary is deliberate:

| Studio owns | Relay and its Node app own |
| --- | --- |
| Creator-facing local and Relay run inbox | Account-scoped scheduling and durable control-plane state |
| Run inspection, artifact fetch, cancel, retry | Node identity, eligibility, leases, placement, and retries |
| Local executor profile selection and run destinations | Worker lifecycle, hardware/runtime telemetry, model inventory and distribution |
| Links into the fleet console | Fleet policy, draining, priority, device revocation, and inventory refresh |

Graph authoring opens the canonical Graph Studio product, while device pairing
and fleet policy open the Relay console. Studio consumes public CLI contracts
and does not keep parallel graph, fleet, node, or scheduling schemas.

## Schemas

- [Workflow graph V1](/schemas/workflow-graph-v1.schema.json)
- [Workflow inputs V1](/schemas/workflow-inputs-v1.schema.json)
- [Asset manifest V1](/schemas/asset-manifest-v1.schema.json)
- [Job bundle V1](/schemas/job-bundle-v1.schema.json)
- [Graph run V1](/schemas/graph-run-v1.schema.json)
- [Worker probe V1](/schemas/worker-probe-v1.schema.json)
- [Plugin graph provider V1](/schemas/graph-node-provider.v1.schema.json)
- [Plugin graph invocation V1](/schemas/graph-node-invocation.v1.schema.json)
- [Plugin graph preflight V1](/schemas/graph-node-preflight.v1.schema.json)
- [Plugin graph event V1](/schemas/graph-node-event.v1.schema.json)
