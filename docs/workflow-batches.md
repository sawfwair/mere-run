# Fleet workflow batches

Use a workflow batch to run independent inputs across a relay fleet. Each item
executes its complete workflow graph on one worker, preserving intermediate files
and dependencies on that machine.

Create `shots.jsonl` with one typed input object per line:

```json
{"key":"scene-001","inputs":{"prompt":"A lighthouse in morning fog"}}
{"key":"scene-002","inputs":{"prompt":"A quiet harbor at dusk"}}
```

Submit a workflow whose `prompt` input drives its generation node:

```bash
mere.run graph batch submit workflow.json \
  --items-jsonl shots.jsonl \
  --executor relay:fleet \
  --run-dir ./runs/lighthouse-batch \
  --max-in-flight 8 \
  --json
```

`--items-jsonl` also accepts a JSON array. Item keys must be unique and contain
letters, numbers, dots, underscores, colons, or hyphens. Models and providers use
the ordinary portable workflow contract.

The command persists `batch.json` with child IDs and seeds before materializing
the bundles. Repeat the same command and directory to resume submission. Changed
rows, graphs, or admission settings cannot reuse that directory. Interrupted
preparation is preserved before reconstruction. Shared input assets upload once
per account digest. The committed batch continues after the CLI exits.

The in-flight window shares the account's active graph quota. It includes queued
and running children. Workers receive another complete graph as they become
available; the batch does not distribute individual graph nodes or model tensors.
The relay schedules batches fairly alongside ordinary work and requires Node
agents that advertise assignment-token support.

## Inspect and retrieve results

Copy the `batch_id` from the submission response:

```bash
mere.run graph batch inspect <batch-id> --executor relay:fleet --json
mere.run graph batch watch <batch-id> --executor relay:fleet --json
mere.run graph batch fetch <batch-id> --executor relay:fleet --into ./outputs --json
```

Inspection returns counts, ordered item keys, child job IDs, attempts, errors,
placement diagnostics, and verified output metadata. Watch emits NDJSON changes
until the batch finishes, fails, or is cancelled. `--timeout-seconds` limits the
watch without stopping the batch.

Fetch writes completed items into ordinal directories and saves
`batch-results.json` with the item-key mapping. It uses ordinary run fetching to
verify artifact size and SHA-256. Repeating fetch reuses verified local files.

## Retry and cancel

Infrastructure retries are disabled by default. For replay-safe workflows, add
`--max-attempts 2` to submission. This includes the initial attempt. A disconnect
or stale assignment can restart that item's entire graph on another compatible
worker. Model failures do not automatically rerun successful siblings. Old
attempts cannot overwrite the active assignment's outputs. External effects must
be idempotent because a disconnected attempt might still execute.

```bash
mere.run graph batch retry <batch-id> --executor relay:fleet --json
mere.run graph batch cancel <batch-id> --executor relay:fleet --json
```

Retry reactivates settled failed items using their original bundles and seeds.
Completed outputs remain available. Cancel first stops admission, then cancels
unfinished children. Cancelled batches remain terminal.

Batch v1 accepts 1–1000 portable items, a window of 1–20 children, and 1–4 attempts.
The Relay API also bounds request and descriptor sizes, final output indexes,
account batch metadata, and graph storage. Local-custody batches require a separate
contract and are rejected. Real checkpoint performance and physical fleet
throughput require qualification beyond the local protocol tests.

## Local qualification

The October 14, 2026 Apple Silicon qualification passed `./scripts/check.sh`:
5,462 XCTest cases (451 skipped) and 193 Swift Testing cases, plus lint, build,
command help, package policy, and repository hygiene checks. `pnpm docs:build`
also passed. Batch-specific tests cover immutable child IDs and seeds, replay,
interrupted materialization, canonical 64-bit documents, HTTP submission and
commit, input validation, and CLI parsing.

The companion Relay gate qualifies simulated fleet dispatch and token fencing.
Deploy its Worker and rebuild the desktop Node before using batches against a
real fleet. Real checkpoint execution and physical fleet throughput remain
separate qualification steps.
