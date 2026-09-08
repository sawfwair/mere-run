# MereRunResidency

Use this library to keep inference runtimes resident without importing CLI, HTTP,
or tensor implementations. It depends on `MereRunAdmission`.

- `ResidentRuntimeSlot.swift`: exclusive mutable-runtime reuse, replacement, and idle eviction.
- `RuntimeOperationCoordinator.swift`: concurrent warm lanes and exclusive cold operations.
- `ResidentRuntimeCache.swift`: shared cold preparation, concurrent leases, generation checks, and cleanup.
- `ScopedResidency.swift`: operation-scoped model lease ownership.
- `RuntimeEvictionPlanner.swift` and `RuntimeEvictionReason.swift`: shared TTL and pressure decisions.

Runtime adapters supply construction, preparation, execution, unloading, and settings.
