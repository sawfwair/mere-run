import Foundation
import MereRunCore
#if os(Linux)
import Glibc
#else
import Darwin
#endif

enum MachineInferenceClass: String, Codable, CaseIterable, Sendable {
    case small
    case standard
    case large

    func permits(capacity: Int) -> Int {
        switch self {
        case .small:
            return 1
        case .standard:
            return min(2, capacity)
        case .large:
            return capacity
        }
    }

    var minimumAvailableBytes: UInt64 {
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

struct MachineInferenceRequest: Equatable, Sendable {
    let label: String
    let resourceClass: MachineInferenceClass
}

struct MachineInferenceAdmissionEntrySnapshot: Codable, Equatable, Sendable {
    let id: UUID
    let processID: Int32
    let label: String
    let resourceClass: MachineInferenceClass
    let permits: Int
    let queuedAt: Date
    let admittedAt: Date?
}

struct MachineInferenceAdmissionSnapshot: Codable, Equatable, Sendable {
    let capacityPermits: Int
    let activePermits: Int
    let active: [MachineInferenceAdmissionEntrySnapshot]
    let queued: [MachineInferenceAdmissionEntrySnapshot]
    let memoryPressure: RuntimeMemoryPressureLevel
    let availableMemoryBytes: UInt64?
    let availableDiskBytes: UInt64?
    let minimumDiskBytes: UInt64
}

struct MachineInferenceHostSnapshot: Equatable, Sendable {
    let physicalMemoryBytes: UInt64
    let availableMemoryBytes: UInt64?
    let memoryPressure: RuntimeMemoryPressureLevel
    let availableDiskBytes: UInt64?
}

enum MachineInferenceAdmissionError: LocalizedError, Equatable {
    case corruptState(String)
    case missingTicket(UUID)
    case insufficientDisk(available: UInt64, required: UInt64)
    case insufficientMemory(available: UInt64, required: UInt64)
    case criticalMemoryPressure
    case systemCall(operation: String, code: Int32)

    var errorDescription: String? {
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

struct MachineInferenceCoordinator: Sendable {
    static let pollNanoseconds: UInt64 = 250_000_000

    let stateDirectory: URL
    private let processID: Int32
    private let bootSessionID: String
    private let currentDate: @Sendable () -> Date
    private let hostSnapshot: @Sendable () -> MachineInferenceHostSnapshot
    private let processIsAlive: @Sendable (Int32) -> Bool
    private let sleeper: @Sendable (UInt64) async throws -> Void

    static var shared: MachineInferenceCoordinator {
        MachineInferenceCoordinator()
    }

    init(
        stateDirectory: URL = MereRunModelPaths.applicationSupportBase
            .appendingPathComponent("admission", isDirectory: true),
        processID: Int32 = getpid(),
        bootSessionID: String = MachineInferenceCoordinator.currentBootSessionID(),
        currentDate: @escaping @Sendable () -> Date = Date.init,
        hostSnapshot: @escaping @Sendable () -> MachineInferenceHostSnapshot = {
            MachineInferenceCoordinator.currentHostSnapshot()
        },
        processIsAlive: @escaping @Sendable (Int32) -> Bool = {
            MachineInferenceCoordinator.isProcessAlive($0)
        },
        sleeper: @escaping @Sendable (UInt64) async throws -> Void = {
            try await Task.sleep(nanoseconds: $0)
        }
    ) {
        self.stateDirectory = stateDirectory
        self.processID = processID
        self.bootSessionID = bootSessionID
        self.currentDate = currentDate
        self.hostSnapshot = hostSnapshot
        self.processIsAlive = processIsAlive
        self.sleeper = sleeper
    }

    func acquire(
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

    func acquireBlocking(
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

    func snapshot() throws -> MachineInferenceAdmissionSnapshot {
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

    static func capacityPermits(physicalMemoryBytes: UInt64) -> Int {
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

    static func minimumDiskBytes(physicalMemoryBytes: UInt64) -> UInt64 {
        let gibibyte = UInt64(1_073_741_824)
        return min(32 * gibibyte, max(8 * gibibyte, physicalMemoryBytes / 8))
    }

    private static func ticketOrder(_ lhs: MachineInferenceTicket, _ rhs: MachineInferenceTicket) -> Bool {
        if lhs.queuedAt != rhs.queuedAt {
            return lhs.queuedAt < rhs.queuedAt
        }
        return lhs.id.uuidString < rhs.id.uuidString
    }

    private static func currentHostSnapshot() -> MachineInferenceHostSnapshot {
        let memory = RuntimeMemorySample.current()
        let disk = availableDiskBytes(
            at: MereRunModelPaths.applicationSupportBase.deletingLastPathComponent()
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
    static func reconciledAvailableDiskBytes(
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

    static func currentBootSessionID() -> String {
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

final class MachineInferenceLease: @unchecked Sendable {
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

    func release() {
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

enum CLIInferenceAdmissionClassifier {
    private static let largeModelBytes = Int64(48 * 1_073_741_824)

    static func request(arguments: [String]) -> MachineInferenceRequest? {
        let tokens = Array(arguments.dropFirst())
        guard !tokens.contains("--help"),
              !tokens.contains("-h"),
              !tokens.contains("--version"),
              !tokens.contains("--dry-run") else {
            return nil
        }
        let commandTokens = commandPath(tokens)
        guard let topLevel = commandTokens.first else { return nil }
        let subcommand = commandTokens.dropFirst().first
        let label = [topLevel, subcommand].compactMap { $0 }.joined(separator: " ")

        // These orchestration commands either acquire a workload-specific
        // lease internally or spawn a child process that does. Classifying
        // their --model argument here would make the parent hold permits while
        // waiting for the child to acquire the same permits.
        if ["api", "open-webui", "run", "graph", "executor", "gate", "setup", "agent", "plugin"]
            .contains(topLevel) {
            return nil
        }

        if tokens.contains(where: { token in
            let normalized = token.lowercased()
            return normalized.contains("deepseek-v4") || normalized.contains("ds4")
        }) {
            return MachineInferenceRequest(label: label, resourceClass: .large)
        }
        if modelIdentifiers(in: tokens).contains(where: isLargeModel) {
            return MachineInferenceRequest(label: label, resourceClass: .large)
        }

        switch topLevel {
        case "speech", "audio":
            return MachineInferenceRequest(label: label, resourceClass: .small)
        case "image":
            let large = [
                "reconstruct-3d",
                "reconstruct-3d-multiview",
                "reconstruct-3d-trellis2",
                "image-to-3d",
                "image-to-3d-multiview",
                "train-lora",
            ].contains(subcommand)
            return MachineInferenceRequest(label: label, resourceClass: large ? .large : .standard)
        case "text":
            let large = subcommand == "train-lora"
            return MachineInferenceRequest(label: label, resourceClass: large ? .large : .standard)
        case "vision":
            let large = [
                "geometry",
                "geometry-multiview",
                "image-to-3d",
                "image-to-3d-trellis2",
                "depth-video",
            ].contains(subcommand)
            return MachineInferenceRequest(label: label, resourceClass: large ? .large : .standard)
        case "music", "sfx":
            let large = ["train-adapter"].contains(subcommand)
            return MachineInferenceRequest(label: label, resourceClass: large ? .large : .standard)
        case "video", "world":
            return MachineInferenceRequest(label: label, resourceClass: .large)
        case "geo":
            return MachineInferenceRequest(label: label, resourceClass: .large)
        case "model" where subcommand == "benchmark":
            return MachineInferenceRequest(label: label, resourceClass: .large)
        case "adapter" where subcommand != "list" && subcommand != "pull":
            return MachineInferenceRequest(label: label, resourceClass: .standard)
        default:
            return nil
        }
    }

    static func apiServerRequest(engine: APIEngine, modelID: String? = nil) -> MachineInferenceRequest {
        let resourceClass: MachineInferenceClass = engine == .textChatDeepseekV4Flash || isLargeModel(modelID)
            ? .large
            : .standard
        return MachineInferenceRequest(
            label: "api serve \(engine.rawValue)",
            resourceClass: resourceClass
        )
    }

    private static func modelIdentifiers(in tokens: [String]) -> [String] {
        var identifiers: [String] = []
        for (index, token) in tokens.enumerated() {
            if token == "--model" || token == "-m" {
                if tokens.indices.contains(index + 1) {
                    identifiers.append(tokens[index + 1])
                }
            } else if token.hasPrefix("--model=") {
                identifiers.append(String(token.dropFirst("--model=".count)))
            }
        }
        return identifiers
    }

    private static func isLargeModel(_ identifier: String?) -> Bool {
        guard let identifier, !identifier.isEmpty else { return false }
        if let estimatedBytes = ManagedModelCatalog.spec(for: identifier)?.estimatedDownloadBytes {
            return estimatedBytes >= largeModelBytes
        }
        let fileURL = URL(fileURLWithPath: identifier)
        guard let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
              values.isRegularFile == true,
              let fileSize = values.fileSize else {
            return false
        }
        return Int64(fileSize) >= largeModelBytes
    }

    private static func commandPath(_ tokens: [String]) -> [String] {
        var result: [String] = []
        var skipNext = false
        for token in tokens {
            if skipNext {
                skipNext = false
                continue
            }
            if token == "--models-root" {
                skipNext = true
                continue
            }
            if token.hasPrefix("--models-root=") || token.hasPrefix("-") {
                continue
            }
            result.append(token)
            if result.count == 2 {
                break
            }
        }
        return result
    }
}

private func releaseCLIProcessAdmission() {
    CLIProcessAdmissionBootstrap.release()
}

enum CLIProcessAdmissionBootstrap {
    private final class Storage: @unchecked Sendable {
        let lock = NSLock()
        var lease: MachineInferenceLease?
        var registeredExitHandler = false
    }

    private static let storage = Storage()

    static func acquireIfNeeded(arguments: [String]) throws {
        guard let request = CLIInferenceAdmissionClassifier.request(arguments: arguments) else {
            return
        }
        storage.lock.lock()
        defer { storage.lock.unlock() }
        guard storage.lease == nil else { return }
        var waited = false
        let lease = try MachineInferenceCoordinator.shared.acquireBlocking(request) { snapshot in
            waited = true
            CLIStderr.write(
                "Queued by machine admission: \(request.label) "
                    + "(\(snapshot.activePermits)/\(snapshot.capacityPermits) permits active, "
                    + "\(snapshot.queued.count) queued).\n"
            )
        }
        if waited {
            CLIStderr.write("Machine admission granted: \(request.label).\n")
        }
        storage.lease = lease
        if !storage.registeredExitHandler {
            atexit(releaseCLIProcessAdmission)
            storage.registeredExitHandler = true
        }
    }

    static func release() {
        storage.lock.lock()
        let lease = storage.lease
        storage.lease = nil
        storage.lock.unlock()
        lease?.release()
    }
}
