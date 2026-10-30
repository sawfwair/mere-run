# Inference admission

Use `MereRunAdmission` to coordinate inference work without importing the CLI,
HTTP server, or tensor runtimes. The library uses Foundation and platform
system APIs and has no Swift package dependencies.

## Ownership

Admission has three scopes:

| Scope | Owner | Lifetime |
| --- | --- | --- |
| Machine reservation | `MachineInferenceCoordinator` | One command, or the full server lifetime |
| Request slot | `RuntimeRequestAdmission` | One request or model-maintenance operation |
| Loaded-model lease | Runtime pool (see [runtime residency](./runtime-residency.md)) | Runtime use, including streaming production |

A server holds one weighted machine reservation. Its request queue enforces
`--max-active-requests` inside that reservation, so acquiring a request slot
never acquires another machine ticket.

The machine coordinator keeps weighted FIFO ordering, disk and memory headroom
checks, file locking, atomic state updates, and recovery from dead processes and
previous boots. CLI adapters select the shared admission state directory and
classify command arguments. Typed callers provide a `MachineInferenceRequest`.
A caller with a resolved model-size estimate can use its `estimatedModelBytes`
initializer with a modality cost floor, or request exclusive admission.

`MereRunResidency` owns model generations, leases, and eviction mechanics.
`RuntimeModelPool` and `APISidecarModelPool` supply runtime factories, settings,
and pressure samples, and their runtime owners keep batching and KV caches. The
admission library never loads checkpoints.

`RuntimeRequestProgress` carries admission telemetry. CLI adapters translate
Core's `ChatProgress` into that type, and the status field names stay
compatible.

## Release ownership

Choose the helper that matches how long the work lives:

- Use `withMachineInferenceAdmission` when the operation's lifetime matches the
  machine reservation. API serving uses this scope around server creation and
  execution, so startup failure and shutdown release the same reservation.
- Use `withRuntimeRequestAdmission` when the operation finishes before its
  caller returns. The helper releases the slot on success, failure, or
  cancellation, and scoped cancellations increment the cancellation counter.
- For a streaming response whose producer outlives its handler, acquire an
  explicit request lease and release it when production ends. The streaming
  owner holds both its request slot and its loaded-model lease. Don't wrap
  response construction in a scope that releases before streaming finishes.

Request admission checks cancellation before it acquires a slot and after
suspended memory-pressure sampling. It removes cancelled queued requests and
releases a lease if cancellation races with a queued grant. Its pressure and
capacity checks keep FIFO ordering across actor suspension.

## Validate a change

Compile the library and its test module without the inference targets:

```bash
swift build --target MereRunAdmissionTests
```

This command only compiles the tests. Run `swift test` or `./scripts/check.sh`
to run them. Coverage includes machine FIFO and crash recovery, request
cancellation, scoped cleanup, concurrent release, and the distinction between
server reservations and request concurrency. Runtime-pool and API integration
tests live in `MereRunCLITests`.
