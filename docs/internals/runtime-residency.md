# Runtime residency and serving services

Use `MereRunResidency` when an executor needs to keep models warm between
operations. The library uses Foundation and `MereRunAdmission`; it does not
import the CLI, HTTP server, or tensor runtimes.

## Choose the residency pattern

| Runtime behavior | Owner | Concurrency |
| --- | --- | --- |
| Supports concurrent requests | `ResidentRuntimeCache` | Multiple leases for one prepared generation |
| Has mutable state that requires exclusive use | `ResidentRuntimeSlot` | One operation at a time per slot |
| Shares cold-load headroom across media lanes | `RuntimeOperationCoordinator` | Warm lanes can overlap; cold operations are exclusive |

The text pool uses `ResidentRuntimeCache`. It supplies model factories and
preparation and unload callbacks. The media pool uses a separate
`ResidentRuntimeSlot` for image generation, speech synthesis, transcription,
and embeddings. Each slot retains one model identity and unloads a replacement's
predecessor before constructing the replacement.

The runtime adapters continue to own checkpoint loading, model settings,
batching, and KV caches. The residency library owns lifecycle mechanics.

## Keep a generation alive

Acquire a model lease before inference. A concurrent cold request joins the
same preparation. Cancelling one waiter preserves preparation for the remaining
waiters. When the last waiter cancels, cleanup waits for preparation to settle
before unloading its generation.

A generation identifier prevents a cancelled load from returning or unloading
a replacement model. Active leases block explicit unload and idle eviction.
Release a lease explicitly when streaming production ends. For an operation
that finishes before its caller returns, use `withResidentRuntimeLease` to
release on success, failure, or cancellation.

## Apply eviction policy

`RuntimeEvictionPlanner` selects expired or least-recently-used residents. It
excludes active, queued, preparing, pinned, and explicitly protected models.
Elevated pressure selects one eligible resident; critical pressure can select
all eligible residents. Text and media adapters preserve their existing default
model protections and pressure resampling.

Revalidate each decision before unloading. The concurrent cache checks both the
model generation and its access generation, so a warm request invalidates an
earlier idle decision. The exclusive slot uses its idle-eviction generation and
execution gate. Autonomous media TTL checks continue to reread pin and deadline
settings.

## Connect serving owners

`RuntimeServingServices` in CLI Support composes the text pool, media pool, and
request admission. It connects their pressure callbacks and owns request scopes
for model maintenance and transcription. It has no HTTP routing or authentication
code.

The server retains authentication, rate limits, multipart parsing, response
formatting, and streaming transport. A server holds one machine reservation for
its full lifetime. Its request slots and model leases operate inside that
reservation; acquiring a model does not acquire another machine ticket.

## Validate without checkpoints

Compile the residency test target without inference dependencies:

```bash
swift build --target MereRunResidencyTests
```

Run the repository gate to execute the library and integration tests:

```bash
./scripts/check.sh
```

Tests cover shared cold loads, warm reuse, cancellation, replacement generations,
active-lease protection, TTL, pressure eviction, and service request cleanup.
These lifecycle tests do not measure throughput or checkpoint inference quality.
