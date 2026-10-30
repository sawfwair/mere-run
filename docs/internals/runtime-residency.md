# Runtime residency and serving services

Use `MereRunResidency` when an executor needs to keep models warm between
operations. The library depends only on Foundation and `MereRunAdmission`; it
doesn't import the CLI, HTTP server, or tensor runtimes.

## Ownership

Choose the residency pattern that matches the runtime:

| Runtime behavior | Owner | Concurrency |
| --- | --- | --- |
| Supports concurrent requests | `ResidentRuntimeCache` | Multiple leases for one prepared generation |
| Has mutable state that needs exclusive use | `ResidentRuntimeSlot` | One operation at a time per slot |
| Shares cold-load headroom across media lanes | `RuntimeOperationCoordinator` | Warm lanes can overlap; cold operations are exclusive |

The text pool uses `ResidentRuntimeCache` and supplies model factories and
preparation and unload callbacks. The media pool, `APISidecarModelPool`, uses a
separate `ResidentRuntimeSlot` for each of image generation, speech synthesis,
transcription, embeddings, and decisions. Each slot holds one model identity and unloads
the current model before it constructs a replacement.

The runtime adapters keep checkpoint loading, model settings, batching, and KV
caches. The residency library owns lifecycle mechanics. For request slots and
machine reservations, see [inference admission](./inference-admission.md).

## Keep a generation alive

Acquire a model lease before inference. A concurrent cold request joins the
same preparation. Cancelling one waiter keeps preparation going for the others.
When the last waiter cancels, cleanup waits for preparation to settle before it
unloads the generation.

A generation identifier keeps a cancelled load from returning or unloading a
replacement model. Active leases block explicit unload and idle eviction.
Release a lease explicitly when streaming production ends. For an operation
that finishes before its caller returns, use `withResidentRuntimeLease` to
release on success, failure, or cancellation.

## Apply eviction policy

`RuntimeEvictionPlanner` selects expired or least-recently-used residents. It
excludes active, queued, preparing, pinned, and explicitly protected models.
Elevated pressure selects one eligible resident; critical pressure can select
all of them. The text and media adapters keep their default model protections
and pressure resampling.

Revalidate each decision before you unload. The concurrent cache checks both
the model generation and its access generation, so a warm request invalidates an
earlier idle decision. The exclusive slot uses its idle-eviction generation and
execution gate. Autonomous media TTL checks reread pin and deadline settings.

## Connect serving owners

`RuntimeServingServices` in `Sources/MereRunCLI/Support` composes the text
pool, media pool, and request admission. It connects their pressure callbacks
and owns request scopes for model maintenance and transcription. It has no HTTP
routing or authentication code.

The server keeps authentication, rate limits, multipart parsing, response
formatting, and streaming transport. It holds one machine reservation for its
full lifetime. Request slots and model leases operate inside that reservation;
acquiring a model never acquires another machine ticket.

## Validate a change

Compile the residency test target without inference dependencies:

```bash
swift build --target MereRunResidencyTests
```

The repository gate runs the library and integration tests:

```bash
./scripts/check.sh
```

Tests cover shared cold loads, warm reuse, cancellation, replacement
generations, active-lease protection, TTL, pressure eviction, and service
request cleanup. These lifecycle tests don't measure throughput or checkpoint
inference quality.
