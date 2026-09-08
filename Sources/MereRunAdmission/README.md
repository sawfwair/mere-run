# MereRunAdmission

Reusable machine-wide and in-process inference admission with Foundation and
platform system APIs. The target has no package dependencies and does not load
models or import CLI parsing, HTTP, Core, or MLX.

- `MachineInferenceAdmission.swift`: weighted, persistent FIFO tickets, host
  resource checks, crash recovery, and idempotent machine leases. Callers select
  the state directory; CLI adapters retain the existing shared location.
- `RuntimeMemoryPressure.swift`: host samples and the existing memory-guard policy.
- `InferenceResourceClassification.swift`: resolved model estimates and modality
  cost floors, with explicit exclusive admission.
- `ScopedAdmission.swift`: release on success, failure, and cancellation.
- `RuntimeRequestAdmission.swift`: fair request admission, pressure-aware
  concurrency, cancellation, leases, and typed lifecycle telemetry.

A server holds one machine reservation for its lifetime. Its request queue
operates inside that reservation. Request admission does not acquire another
machine ticket. Loaded-model leases, pinning, TTL, eviction, batching, and KV
caches remain in their runtime owners.
