# Inference admission

Use `MereRunAdmission` to coordinate inference work without importing the CLI,
HTTP server, or tensor runtimes. It uses Foundation and platform system APIs
and has no Swift package dependencies.

## Two admission scopes

| Scope | Owner | Lifetime |
| --- | --- | --- |
| Machine reservation | `MachineInferenceCoordinator` | One command or the full server lifetime |
| Request slot | `RuntimeRequestAdmission` | One request or model-maintenance operation |
| Loaded-model lease | Runtime pool | Runtime use, including streaming production |

A server holds one weighted machine reservation. Its request queue enforces
`--max-active-requests` inside that reservation. Acquiring a request slot does
not acquire another machine ticket.

The machine coordinator preserves weighted FIFO ordering, disk and memory
headroom checks, file locking, atomic state updates, and dead-process and
previous-boot recovery. CLI adapters select the existing shared admission
state directory and classify command arguments. Typed callers provide a
`MachineInferenceRequest`; resolved model-size estimates can use its
`estimatedModelBytes` initializer with a modality cost floor or explicit
exclusive admission.

## Release ownership

Use `withMachineInferenceAdmission` when the operation's lifetime matches the
machine reservation. API serving uses this scope around server creation and
execution, so startup failure and shutdown release the same reservation.

Use `withRuntimeRequestAdmission` when the operation finishes before its
caller returns. The helper releases the slot on success, failure, or
cancellation. Scoped cancellations increment the cancellation counter.

For a streaming response whose producer outlives its handler, acquire an
explicit request lease and release it when production ends. The streaming
owner retains both its request slot and its loaded-model lease. Do not wrap
response construction in a scope that releases before streaming finishes.

Request admission checks cancellation before acquiring a slot and after
suspended memory-pressure sampling. It removes cancelled queued requests and
releases a lease if cancellation races with a queued grant. Its pressure and
capacity checks retain FIFO ordering across actor suspension.

## Runtime boundaries

`RuntimeRequestProgress` carries admission telemetry. CLI adapters translate
Core's `ChatProgress` into that type. The existing status field names remain
compatible.

`RuntimeModelPool` and `APISidecarModelPool` own loaded models, model leases,
pinning, TTL, eviction, readiness, batching, and KV caches. They supply pressure
samples to request admission. The admission library does not load checkpoints
or change those runtime policies.

## Validation

To compile the library and its test module without the inference targets, run:

```bash
swift build --target MereRunAdmissionTests
```

This command compiles the tests. Run `swift test` or `./scripts/check.sh` to
execute the package tests. Coverage includes machine FIFO and crash recovery,
request cancellation, scoped cleanup, concurrent release, and the distinction
between server reservations and request concurrency. Runtime-pool and API
integration tests remain in `MereRunCLITests`.
