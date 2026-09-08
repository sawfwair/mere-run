@_exported import MereRunResidency

typealias APISidecarResidentSlotState<Key: Equatable & Sendable> = ResidentRuntimeSlotState<Key>
typealias APISidecarResidentIdlePolicy = ResidentIdlePolicy
typealias APISidecarOperationMode = RuntimeOperationMode
typealias APISidecarOperationCoordinator = RuntimeOperationCoordinator
typealias APISidecarOperationLease = RuntimeOperationLease
typealias APISidecarResidentSlot<Key: Equatable & Sendable, Value: Sendable> = ResidentRuntimeSlot<Key, Value>
typealias RuntimeSidecarEvictionReason = RuntimeEvictionReason
