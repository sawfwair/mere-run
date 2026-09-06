# Workflows and services

## Author a portable graph

Read `graph catalog --json` for node kinds, typed inputs/outputs, model
requirements, and provider identity. CLI capability IDs and graph node kinds
are separate catalogs. Graph v2 is the runtime name; its serialized graph still
uses `schema_version: 1` and `kind: "mere.run/workflow-graph"`.

Use complete JSON references such as `{"$ref":"nodes.message.outputs.text"}`,
not shell substitutions or interpolated reference strings. Node references add
dependencies. `depends_on` adds ordering-only edges. Pin explicit model IDs and
use the catalog's input types for files, directories, and collections.

For a model-free first workflow, save this document as `workflow.json` in a
fresh task directory:

```json
{
  "schema_version": 1,
  "kind": "mere.run/workflow-graph",
  "name": "message-check",
  "inputs": {},
  "nodes": [
    {
      "id": "message",
      "kind": "text.value",
      "arguments": {"value": "Workflow execution verified."}
    }
  ],
  "outputs": {"message": {"$ref": "nodes.message.outputs.text"}}
}
```

Inspect the catalog, validate, and check the selected executor:

```bash
mere.run graph catalog --json
mere.run graph validate ./workflow.json --json
mere.run graph preflight ./workflow.json --executor local --json
```

After the report permits execution, run and inspect the result:

```bash
mere.run graph run ./workflow.json --run-dir ./runs/message-check --json
mere.run run inspect ./runs/message-check --json
```

For graphs with required inputs, provide the corresponding `--inputs-json`
file consistently at validation, preflight, and execution. Discover dataset
candidates with `graph dataset discover` when relevant. Worker preflight checks
installed models, node/provider contracts, accelerator resources, storage,
network requirements, and secret availability. Graph workers do not implicitly
pull missing models; provision them on the selected worker and repeat preflight.

## Bundles, resume, and output identity

`graph materialize` creates an immutable job bundle in a run directory.
`graph export-job` exports one for later `graph run-job` or `graph submit-job`.
The bundle includes `job.json`, `graph.json`, `inputs.json`, `assets.json`, and
content-addressed assets. Executor profiles and credentials remain external.

For an exported bundle, use a separate, non-nested mutable run directory.
Do not edit a bundle after hashing or submission. `--resume` on local
`graph run`/`run-job` reuses finished nodes only when arguments, provider/model
provenance, referenced inputs, and output digests still match. Reusing a folder
alone is not proof that work can resume.

A run directory contains `run.json`, `events.jsonl`, node records, and
`outputs/`. Preserve it for inspection and recovery. Check final state and
verify the declared outputs. For a local graph, `run inspect --json` returns
the run record: read `contract_version`, `state`, `job_id`, `nodes`, and
`outputs`, rather than expecting a preflight-style `status` envelope. Resolve
relative artifact paths against the run directory. Parallelism is explicit in
the graph and should fit the executor's memory budget; more independent nodes do not imply spare GPU
capacity.

## SSH and relay execution

Use `executor list --json`, `executor inspect`, `executor probe`, and, for relay,
`executor auth-status` before submission. Use `executor login` when the selected
relay profile needs its supported authorization flow. Do not print credentials.
SSH uses system SSH configuration, batch authentication, and host-key checks.

After configuring and preflighting the requested executor, submit with
`graph submit` or `graph submit-job`. Preserve the returned job reference.
Profile selection uses `ssh:PROFILE` or `relay:PROFILE`; remote job references
use `ssh://PROFILE/JOB_ID` or `relay://PROFILE/JOB_ID`.

| Operation | Contract |
| --- | --- |
| Inspect | `run inspect` supports local directories/reports and remote references. |
| Watch | `run watch` accepts SSH/relay references. `--json-stream` emits events; `--json` emits a final job object. |
| Fetch | `run fetch` verifies sizes and SHA-256. An interrupted fetch can reuse verified local files. |
| Cancel | `run cancel` targets SSH/relay jobs. Inspect afterward to confirm terminal cancellation. |
| Retry | `run retry` accepts relay references and reuses the immutable bundle. Track the returned retry identity. |
| Resume local | Use the graph command's `--resume`, not `run retry`. |

A successful submit means accepted or queued, not finished. If a relay job
stays queued, inspect its placement report for concrete worker/model/resource
blockers instead of submitting duplicates. After a lost response, inspect/list
existing work before deciding to resubmit.

Fetch defaults to reports, manifests, and final outputs. Use `--artifact` for
selected artifacts or `--all-artifacts` for node artifacts; do not combine
those two modes. Check that the completed result matches the graph and inputs
the user actually requested, especially after editing the source workflow.

`relay serve` hosts the graph-job protocol on a machine. It is different from
`api serve` and from a persistent `world` session. Use its help for pairing,
binding, spool, and lifecycle options; preserve its authentication boundary.

## Serve and verify an API

Choose an engine compatible with the selected model. First inspect its pull:

```bash
mere.run model pull text-chat-gemma4 --preflight --json
```

After resolving blockers, pull and inspect the serving request:

```bash
mere.run model pull text-chat-gemma4
mere.run api serve --engine text-chat-gemma4 --model text-chat-gemma4 \
    --host 127.0.0.1 --port 8080 --preflight --json
```

After reading the report, start the same configuration as a retained service
process, then check it from another terminal:

```bash
mere.run api serve --engine text-chat-gemma4 --model text-chat-gemma4 \
    --host 127.0.0.1 --port 8080
mere.run status --host 127.0.0.1 --port 8080 --json
```

API preflight does not bind a port, load a model, or verify an HTTP request.
It can report that runtime auto-download is allowed even with no model
installed. Inspect model state before starting. Resolve port conflicts by
checking the existing service; do not stop an unrelated server.

Use a loopback bind for local work. Non-loopback binds require `--api-key` or
`MERERUN_API_KEY`; reuse the authorized credential without logging it. A bind
address such as `0.0.0.0` is not the remote client's destination. Use the actual
reachable host and port, and `/v1` where the client's base-URL setting expects it.

After startup, check health, `/v1/models`, and a small request to the intended
endpoint/model. OpenAI compatibility varies by engine: inspect tools,
structured-output, vision, sampling, and other capability fields before using
them. A healthy server or model listing alone does not prove a request works.

`/runtime/status` exposes loaded runtime state. `model runtime get/set` manages
persistent per-model aliases, pinning, TTL, and supported generation defaults.
Inspect before changing settings; request concurrency and large contexts can
raise memory demand. Stop only the service process owned by this task.
