import Foundation
#if os(Linux)
import Glibc
#else
import Darwin
#endif

public enum MachineInferenceClass: String, Codable, CaseIterable, Sendable {
    case small
    case standard
    case large

    public func permits(capacity: Int) -> Int {
        switch self {
        case .small:
            return 1
        case .standard:
            return min(2, capacity)
        case .large:
            return capacity
        }
    }

    public var minimumAvailableBytes: UInt64 {
        let gibibyte = UInt64(1_073_741_824)
        switch self {
        case .small:
            return 6 * gibibyte
        case .standard:
            return 16 * gibibyte
        case .large:
            return 32 * gibibyte
        }
    }
}

public struct MachineInferenceRequest: Equatable, Sendable {
    public let label: String
    public let resourceClass: MachineInferenceClass

    public init(label: String, resourceClass: MachineInferenceClass) {
        self.label = label
        self.resourceClass = resourceClass
    }
}

public struct MachineInferenceAdmissionEntrySnapshot: Codable, Equatable, Sendable {
    public let id: UUID
    public let processID: Int32
    public let label: String
    public let resourceClass: MachineInferenceClass
    public let permits: Int
    public let queuedAt: Date
    public let admittedAt: Date?

    public init(
        id: UUID,
        processID: Int32,
        label: String,
        resourceClass: MachineInferenceClass,
        permits: Int,
        queuedAt: Date,
        admittedAt: Date?
    ) {
        self.id = id
        self.processID = processID
        self.label = label
        self.resourceClass = resourceClass
        self.permits = permits
        self.queuedAt = queuedAt
        self.admittedAt = admittedAt
    }
}

public struct MachineInferenceAdmissionSnapshot: Codable, Equatable, Sendable {
    public let capacityPermits: Int
    public let activePermits: Int
    public let active: [MachineInferenceAdmissionEntrySnapshot]
    public let queued: [MachineInferenceAdmissionEntrySnapshot]
    public let memoryPressure: RuntimeMemoryPressureLevel
    public let availableMemoryBytes: UInt64?
    public let availableDiskBytes: UInt64?
    public let minimumDiskBytes: UInt64

    public init(
        capacityPermits: Int,
        activePermits: Int,
        active: [MachineInferenceAdmissionEntrySnapshot],
        queued: [MachineInferenceAdmissionEntrySnapshot],
        memoryPressure: RuntimeMemoryPressureLevel,
        availableMemoryBytes: UInt64?,
        availableDiskBytes: UInt64?,
        minimumDiskBytes: UInt64
    ) {
        self.capacityPermits = capacityPermits
        self.activePermits = activePermits
        self.active = active
        self.queued = queued
        self.memoryPressure = memoryPressure
        self.availableMemoryBytes = availableMemoryBytes
        self.availableDiskBytes = availableDiskBytes
        self.minimumDiskBytes = minimumDiskBytes
    }
}

public struct MachineInferenceHostSnapshot: Equatable, Sendable {
    public let physicalMemoryBytes: UInt64
    public let availableMemoryBytes: UInt64?
    public let memoryPressure: RuntimeMemoryPressureLevel
    public let availableDiskBytes: UInt64?

    public init(physicalMemoryBytes: UInt64, availableMemoryBytes: UInt64?,
                memoryPressure: RuntimeMemoryPressureLevel, availableDiskBytes: UInt64?) {
        self.physicalMemoryBytes = physicalMemoryBytes
        self.availableMemoryBytes = availableMemoryBytes
        self.memoryPressure = memoryPressure
        self.availableDiskBytes = availableDiskBytes
    }
}

public enum MachineInferenceAdmissionError: LocalizedError, Equatable {
    case corruptState(String)
    case missingTicket(UUID)
    case insufficientDisk(available: UInt64, required: UInt64)
    case insufficientMemory(available: UInt64, required: UInt64)
    case criticalMemoryPressure
    case systemCall(operation: String, code: Int32)

    public var errorDescription: String? {
        switch self {
        case .corruptState(let detail):
            return "Machine inference admission state is invalid: \(detail)"
        case .missingTicket(let id):
            return "Machine inference admission ticket \(id.uuidString) disappeared while waiting."
        case .insufficientDisk(let available, let required):
            return "Inference was not started because only \(Self.bytes(available)) of disk space is available; "
                + "at least \(Self.bytes(required)) is reserved for macOS swap and temporary files."
        case .insufficientMemory(let available, let required):
            return "Inference was not started because only \(Self.bytes(available)) of reclaimable memory is available; "
                + "this workload requires at least \(Self.bytes(required)) of admission headroom."
        case .criticalMemoryPressure:
            return "Inference was not started because the machine is already under critical memory pressure."
        case .systemCall(let operation, let code):
            return "Machine inference admission could not \(operation) (errno \(code))."
        }
    }

    private static func bytes(_ value: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .memory)
    }
}

private enum MachineInferenceTicketState: String, Codable, Sendable {
    case queued
    case active
}

private struct MachineInferenceTicket: Codable, Equatable, Sendable {
    let id: UUID
    let processID: Int32
    let bootSessionID: String
    let label: String
    let resourceClass: MachineInferenceClass
    let permits: Int
    let queuedAt: Date
    var admittedAt: Date?
    var state: MachineInferenceTicketState

    var snapshot: MachineInferenceAdmissionEntrySnapshot {
        MachineInferenceAdmissionEntrySnapshot(
            id: id,
            processID: processID,
            label: label,
            resourceClass: resourceClass,
            permits: permits,
            queuedAt: queuedAt,
            admittedAt: admittedAt
        )
    }
}

private struct MachineInferenceAdmissionState: Codable, Equatable, Sendable {
    static let currentVersion = 1

    var version = currentVersion
    var tickets: [MachineInferenceTicket] = []
}

public struct MachineInferenceCoordinator: Sendable {
    public static let pollNanoseconds: UInt64 = 250_000_000

    public let stateDirectory: URL
    private let processID: Int32
    private let bootSessionID: String
    private let currentDate: @Sendable () -> Date
    private let hostSnapshot: @Sendable () -> MachineInferenceHostSnapshot
    private let processIsAlive: @Sendable (Int32) -> Bool
    private let sleeper: @Sendable (UInt64) async throws -> Void

    public init(
        stateDirectory: URL,
        processID: Int32 = getpid(),
        bootSessionID: String = MachineInferenceCoordinator.currentBootSessionID(),
        currentDate: @escaping @Sendable () -> Date = Date.init,
        hostSnapshot: (@Sendable () -> MachineInferenceHostSnapshot)? = nil,
        processIsAlive: (@Sendable (Int32) -> Bool)? = nil,
        sleeper: @escaping @Sendable (UInt64) async throws -> Void = {
            try await Task.sleep(nanoseconds: $0)
        }
    ) {
        self.stateDirectory = stateDirectory
        self.processID = processID
        self.bootSessionID = bootSessionID
        self.currentDate = currentDate
        self.hostSnapshot = hostSnapshot ?? {
            Self.currentHostSnapshot(diskURL: stateDirectory.deletingLastPathComponent())
        }
        self.processIsAlive = processIsAlive ?? { Self.isProcessAlive($0) }
        self.sleeper = sleeper
    }

    public func acquire(
        _ request: MachineInferenceRequest,
        onWait: (@Sendable (MachineInferenceAdmissionSnapshot) -> Void)? = nil
    ) async throws -> MachineInferenceLease {
        let ticketID = try register(request)
        var announcedWait = false
        do {
            while true {
                try Task.checkCancellation()
                let result = try attemptAdmission(ticketID: ticketID)
                if result.admitted {
                    return MachineInferenceLease(coordinator: self, ticketID: ticketID)
                }
                if !announcedWait {
                    onWait?(result.snapshot)
                    announcedWait = true
                }
                try await sleeper(Self.pollNanoseconds)
            }
        } catch {
            try? removeTicket(ticketID)
            throw error
        }
    }

    public func acquireBlocking(
        _ request: MachineInferenceRequest,
        onWait: ((MachineInferenceAdmissionSnapshot) -> Void)? = nil
    ) throws -> MachineInferenceLease {
        let ticketID = try register(request)
        var announcedWait = false
        do {
            while true {
                let result = try attemptAdmission(ticketID: ticketID)
                if result.admitted {
                    return MachineInferenceLease(coordinator: self, ticketID: ticketID)
                }
                if !announcedWait {
                    onWait?(result.snapshot)
                    announcedWait = true
                }
                usleep(useconds_t(Self.pollNanoseconds / 1_000))
            }
        } catch {
            try? removeTicket(ticketID)
            throw error
        }
    }

    public func snapshot() throws -> MachineInferenceAdmissionSnapshot {
        try withLockedState { state in
            pruneDeadTickets(&state)
            return makeSnapshot(state: state, host: hostSnapshot())
        }
    }

    fileprivate func removeTicket(_ ticketID: UUID) throws {
        try withLockedState { state in
            state.tickets.removeAll { $0.id == ticketID }
        }
    }

    private func register(_ request: MachineInferenceRequest) throws -> UUID {
        let host = hostSnapshot()
        try validateDisk(host)
        let capacity = Self.capacityPermits(physicalMemoryBytes: host.physicalMemoryBytes)
        let ticket = MachineInferenceTicket(
            id: UUID(),
            processID: processID,
            bootSessionID: bootSessionID,
            label: request.label,
            resourceClass: request.resourceClass,
            permits: request.resourceClass.permits(capacity: capacity),
            queuedAt: currentDate(),
            admittedAt: nil,
            state: .queued
        )
        try withLockedState { state in
            pruneDeadTickets(&state)
            state.tickets.append(ticket)
        }
        return ticket.id
    }

    private func attemptAdmission(
        ticketID: UUID
    ) throws -> (admitted: Bool, snapshot: MachineInferenceAdmissionSnapshot) {
        try withLockedState { state in
            pruneDeadTickets(&state)
            guard let ticketIndex = state.tickets.firstIndex(where: { $0.id == ticketID }) else {
                throw MachineInferenceAdmissionError.missingTicket(ticketID)
            }
            let host = hostSnapshot()
            try validateDisk(host)
            if state.tickets[ticketIndex].state == .active {
                return (true, makeSnapshot(state: state, host: host))
            }

            let orderedQueued = state.tickets
                .filter { $0.state == .queued }
                .sorted(by: Self.ticketOrder)
            guard orderedQueued.first?.id == ticketID else {
                return (false, makeSnapshot(state: state, host: host))
            }

            let capacity = Self.capacityPermits(physicalMemoryBytes: host.physicalMemoryBytes)
            let activePermits = state.tickets
                .filter { $0.state == .active }
                .reduce(0) { $0 + $1.permits }
            let requestedPermits = state.tickets[ticketIndex].permits
            guard activePermits + requestedPermits <= capacity else {
                return (false, makeSnapshot(state: state, host: host))
            }

            switch host.memoryPressure {
            case .critical:
                throw MachineInferenceAdmissionError.criticalMemoryPressure
            case .elevated where activePermits > 0:
                return (false, makeSnapshot(state: state, host: host))
            case .disabled, .unknown, .nominal, .elevated:
                break
            }

            let minimumAvailable = state.tickets[ticketIndex].resourceClass.minimumAvailableBytes
            if let available = host.availableMemoryBytes, available < minimumAvailable {
                if activePermits > 0 {
                    return (false, makeSnapshot(state: state, host: host))
                }
                throw MachineInferenceAdmissionError.insufficientMemory(
                    available: available,
                    required: minimumAvailable
                )
            }

            state.tickets[ticketIndex].state = .active
            state.tickets[ticketIndex].admittedAt = currentDate()
            return (true, makeSnapshot(state: state, host: host))
        }
    }

    private func validateDisk(_ host: MachineInferenceHostSnapshot) throws {
        let minimum = Self.minimumDiskBytes(physicalMemoryBytes: host.physicalMemoryBytes)
        if let available = host.availableDiskBytes, available < minimum {
            throw MachineInferenceAdmissionError.insufficientDisk(
                available: available,
                required: minimum
            )
        }
    }

    private func pruneDeadTickets(_ state: inout MachineInferenceAdmissionState) {
        state.tickets.removeAll {
            $0.bootSessionID != bootSessionID || !processIsAlive($0.processID)
        }
    }

    private func makeSnapshot(
        state: MachineInferenceAdmissionState,
        host: MachineInferenceHostSnapshot
    ) -> MachineInferenceAdmissionSnapshot {
        let active = state.tickets
            .filter { $0.state == .active }
            .sorted(by: Self.ticketOrder)
        let queued = state.tickets
            .filter { $0.state == .queued }
            .sorted(by: Self.ticketOrder)
        return MachineInferenceAdmissionSnapshot(
            capacityPermits: Self.capacityPermits(physicalMemoryBytes: host.physicalMemoryBytes),
            activePermits: active.reduce(0) { $0 + $1.permits },
            active: active.map(\.snapshot),
            queued: queued.map(\.snapshot),
            memoryPressure: host.memoryPressure,
            availableMemoryBytes: host.availableMemoryBytes,
            availableDiskBytes: host.availableDiskBytes,
            minimumDiskBytes: Self.minimumDiskBytes(physicalMemoryBytes: host.physicalMemoryBytes)
        )
    }

    private func withLockedState<T>(
        _ operation: (inout MachineInferenceAdmissionState) throws -> T
    ) throws -> T {
        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: stateDirectory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        let lockURL = stateDirectory.appendingPathComponent("state.lock", isDirectory: false)
        let descriptor = open(lockURL.path, O_CREAT | O_RDWR, mode_t(0o600))
        guard descriptor >= 0 else {
            throw MachineInferenceAdmissionError.systemCall(operation: "open its lock", code: errno)
        }
        defer { close(descriptor) }
        while flock(descriptor, LOCK_EX) != 0 {
            guard errno == EINTR else {
                throw MachineInferenceAdmissionError.systemCall(operation: "lock its state", code: errno)
            }
        }
        defer { flock(descriptor, LOCK_UN) }

        var state = try loadState()
        let original = state
        let result = try operation(&state)
        if state != original {
            try saveState(state)
        }
        return result
    }

    private func loadState() throws -> MachineInferenceAdmissionState {
        let url = stateDirectory.appendingPathComponent("state.json", isDirectory: false)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return MachineInferenceAdmissionState()
        }
        do {
            let state = try JSONDecoder().decode(
                MachineInferenceAdmissionState.self,
                from: Data(contentsOf: url)
            )
            guard state.version == MachineInferenceAdmissionState.currentVersion else {
                throw MachineInferenceAdmissionError.corruptState(
                    "unsupported version \(state.version)"
                )
            }
            return state
        } catch let error as MachineInferenceAdmissionError {
            throw error
        } catch {
            throw MachineInferenceAdmissionError.corruptState(error.localizedDescription)
        }
    }

    private func saveState(_ state: MachineInferenceAdmissionState) throws {
        let url = stateDirectory.appendingPathComponent("state.json", isDirectory: false)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(state)
        try data.write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    public static func capacityPermits(physicalMemoryBytes: UInt64) -> Int {
        let gibibyte = UInt64(1_073_741_824)
        switch physicalMemoryBytes {
        case ..<(48 * gibibyte):
            return 1
        case ..<(96 * gibibyte):
            return 2
        case ..<(192 * gibibyte):
            return 4
        default:
            return 6
        }
    }

    public static func minimumDiskBytes(physicalMemoryBytes: UInt64) -> UInt64 {
        let gibibyte = UInt64(1_073_741_824)
        return min(32 * gibibyte, max(8 * gibibyte, physicalMemoryBytes / 8))
    }

    private static func ticketOrder(_ lhs: MachineInferenceTicket, _ rhs: MachineInferenceTicket) -> Bool {
        if lhs.queuedAt != rhs.queuedAt {
            return lhs.queuedAt < rhs.queuedAt
        }
        return lhs.id.uuidString < rhs.id.uuidString
    }

    public static func currentHostSnapshot(diskURL: URL) -> MachineInferenceHostSnapshot {
        let memory = RuntimeMemorySample.current()
        let disk = availableDiskBytes(
            at: diskURL
        )
        return MachineInferenceHostSnapshot(
            physicalMemoryBytes: memory.physicalBytes,
            availableMemoryBytes: memory.availableBytes,
            memoryPressure: RuntimeMemoryPressurePolicy.default.pressure(for: memory),
            availableDiskBytes: disk
        )
    }

    private static func availableDiskBytes(at url: URL) -> UInt64? {
#if os(Linux)
        return fileSystemFreeBytes(at: url)
#else
        let importantUsageCapacity = try? url.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        ).volumeAvailableCapacityForImportantUsage
        return reconciledAvailableDiskBytes(
            importantUsageCapacity: importantUsageCapacity,
            fileSystemFreeBytes: fileSystemFreeBytes(at: url)
        )
#endif
    }

    /// The important-usage capacity probe can return zero when macOS cannot
    /// reach `sysmond`, including inside a restricted process sandbox. A
    /// positive filesystem value proves that the volume is not actually full.
    /// When both probes succeed, use the smaller value for admission safety.
    public static func reconciledAvailableDiskBytes(
        importantUsageCapacity: Int64?,
        fileSystemFreeBytes: UInt64?
    ) -> UInt64? {
        let positiveFileSystemBytes = fileSystemFreeBytes.flatMap { $0 > 0 ? $0 : nil }
        guard let importantUsageCapacity else {
            return fileSystemFreeBytes
        }
        guard importantUsageCapacity > 0 else {
            if importantUsageCapacity == 0 {
                return positiveFileSystemBytes ?? 0
            }
            return fileSystemFreeBytes
        }

        let importantBytes = UInt64(importantUsageCapacity)
        guard let positiveFileSystemBytes else {
            return importantBytes
        }
        return min(importantBytes, positiveFileSystemBytes)
    }

    private static func fileSystemFreeBytes(at url: URL) -> UInt64? {
        guard let attributes = try? FileManager.default.attributesOfFileSystem(forPath: url.path),
              let freeBytes = attributes[.systemFreeSize] as? NSNumber else {
            return nil
        }
        return freeBytes.uint64Value
    }

    private static func isProcessAlive(_ processID: Int32) -> Bool {
        guard processID > 0 else { return false }
        if kill(processID, 0) == 0 {
            return true
        }
        return errno == EPERM
    }

    public static func currentBootSessionID() -> String {
#if os(Linux)
        if let value = try? String(contentsOfFile: "/proc/sys/kernel/random/boot_id", encoding: .utf8) {
            return value.trimmingCharacters(in: .whitespacesAndNewlines)
        }
#else
        var bootTime = timeval()
        var size = MemoryLayout<timeval>.size
        if sysctlbyname("kern.boottime", &bootTime, &size, nil, 0) == 0 {
            return "\(bootTime.tv_sec).\(bootTime.tv_usec)"
        }
#endif
        let estimatedBootDate = Date().timeIntervalSince1970 - ProcessInfo.processInfo.systemUptime
        return String(Int64(estimatedBootDate.rounded()))
    }
}

public final class MachineInferenceLease: @unchecked Sendable {
    private let coordinator: MachineInferenceCoordinator
    private let ticketID: UUID
    private let lock = NSLock()
    private var released = false

    fileprivate init(coordinator: MachineInferenceCoordinator, ticketID: UUID) {
        self.coordinator = coordinator
        self.ticketID = ticketID
    }

    deinit {
        release()
    }

    public func release() {
        lock.lock()
        guard !released else {
            lock.unlock()
            return
        }
        released = true
        lock.unlock()
        try? coordinator.removeTicket(ticketID)
    }
}
