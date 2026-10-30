# Fleet workflow batches

Use a workflow batch to run one workflow graph over many independent inputs on
a relay fleet. Each item runs its complete graph on one worker, so its
intermediate files and dependencies stay on that machine. Batches need a relay
executor profile; see [Executor profiles](./workflows.md#executor-profiles).

## Submit a batch

1. Create `shots.jsonl` with one item per line. Each item has a unique `key`
   and an `inputs` object for the graph's declared inputs:

   ```json
   {"key":"scene-001","inputs":{"prompt":"A lighthouse in morning fog"}}
   {"key":"scene-002","inputs":{"prompt":"A quiet harbor at dusk"}}
   ```

2. Submit a workflow whose `prompt` input drives its generation node:

   ```bash
   mere.run graph batch submit workflow.json \
     --items-jsonl shots.jsonl \
     --executor relay:fleet \
     --run-dir ./runs/lighthouse-batch \
     --max-in-flight 8 \
     --json
   ```

3. Copy the `batch_id` from the response. You need it for every other batch
   command.

`--items-jsonl` also accepts a file that holds a JSON array of items. Item keys
are 1–160 characters long and can contain letters, numbers, dots, underscores,
colons, and hyphens. Models and providers follow the ordinary
[portable workflow](./workflows.md) contract.

Submission writes `batch.json`, with each item's child job ID and seed, before
it materializes any bundle. To resume an interrupted submission, run the same
command with the same `--run-dir`. A run directory can't be reused with changed
items, a changed graph, or changed admission settings. Shared input assets
upload once for each digest in the account. After the relay commits the batch,
it keeps running when the CLI exits.

### Admission and scheduling

`--max-in-flight` sets how many children can be queued or running at once, from
1 through 20. The default is 8, and the window counts against the account's
active graph quota. As workers become free, the relay assigns each one a
complete graph. It doesn't split individual nodes or model tensors across
workers. The relay schedules batches fairly alongside ordinary jobs and
requires Node agents that advertise assignment-token support.

## Inspect and retrieve results

The following commands use the batch ID from the submission response:

```bash
mere.run graph batch list --executor relay:fleet --json
mere.run graph batch inspect <batch-id> --executor relay:fleet --json
mere.run graph batch watch <batch-id> --executor relay:fleet --json
mere.run graph batch fetch <batch-id> --executor relay:fleet --into ./outputs --json
```

- `list` returns the durable batches in the relay account.
- `inspect` returns counts, ordered item keys, child job IDs, attempts, errors,
  placement diagnostics, and verified output metadata.
- `watch` emits changes as newline-delimited JSON (NDJSON) until the batch
  finishes, fails, or is cancelled. `--timeout-seconds` stops watching after
  that many seconds without stopping the batch. The default, `0`, waits
  indefinitely.
- `fetch` writes each completed item into its own ordinal run directory and
  saves `batch-results.json`, which maps directories to item keys. It uses the
  ordinary run fetch to verify artifact size and SHA-256, so a repeated fetch
  reuses verified local files.

Batch commands always write machine-readable JSON to stdout.

## Retry and cancel

Infrastructure retries are off by default. For a replay-safe workflow, add
`--max-attempts 2` to `graph batch submit`. The count includes the first
attempt, and the limit is 4. With retries on, a disconnect or a stale
assignment can restart that item's entire graph on another compatible worker.
A model failure doesn't rerun successful sibling items, and an old attempt
can't overwrite the outputs of the active assignment. A disconnected attempt
might still finish, so any external effects in the graph must be idempotent.

To retry failed items or cancel the batch, run:

```bash
mere.run graph batch retry <batch-id> --executor relay:fleet --json
mere.run graph batch cancel <batch-id> --executor relay:fleet --json
```

`retry` reactivates settled failed items with their original bundles and seeds.
Completed outputs stay available. `cancel` stops admission first, then cancels
unfinished children. A cancelled batch stays terminal.

## Limits

A version 1 batch accepts 1–1000 items, an in-flight window of 1–20 children,
and 1–4 attempts per item. The Relay API also bounds request and descriptor
sizes, final output indexes, account batch metadata, and graph storage.
Local-custody batches need a separate contract, and the relay rejects them.

## Local qualification

The October 14, 2026 Apple Silicon qualification passed `./scripts/check.sh`:
5,462 XCTest cases (451 skipped) and 193 Swift Testing cases, plus lint, build,
command help, package policy, and repository hygiene checks. `pnpm docs:build`
also passed. Batch-specific tests cover immutable child IDs and seeds, replay,
interrupted materialization, canonical 64-bit documents, HTTP submission and
commit, input validation, and CLI parsing.

The companion Relay gate qualifies simulated fleet dispatch and token fencing.
Before you use batches against a real fleet, deploy that gate's Worker and
rebuild the desktop Node. Real checkpoint execution and physical fleet
throughput remain separate qualification steps.
