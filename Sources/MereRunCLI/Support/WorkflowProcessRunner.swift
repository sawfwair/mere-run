import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

struct WorkflowProcessResult: Equatable {
    let status: Int32
    let stdout: String
    let stderr: String
    let terminationReason: WorkflowProcessTerminationReason

    init(
        status: Int32,
        stdout: String,
        stderr: String = "",
        terminationReason: WorkflowProcessTerminationReason = .exit
    ) {
        self.status = status
        self.stdout = stdout
        self.stderr = stderr
        self.terminationReason = terminationReason
    }

    var failureSummary: String {
        var summary = terminationReason == .uncaughtSignal
            ? "terminated by signal \(status)"
            : "exited with status \(status)"
        if !stderr.isEmpty {
            summary += ". stderr: \(stderr)"
        } else if !stdout.isEmpty {
            summary += ". stdout: \(stdout.suffix(16 * 1_024))"
        }
        return summary
    }
}

enum WorkflowProcessTerminationReason: String, Equatable {
    case exit
    case uncaughtSignal = "uncaught_signal"

    init(_ reason: Process.TerminationReason) {
        switch reason {
        case .exit:
            self = .exit
        case .uncaughtSignal:
            self = .uncaughtSignal
        @unknown default:
            self = .exit
        }
    }
}

private struct WorkflowProcessTimeoutError: LocalizedError {
    let seconds: Int

    var errorDescription: String? {
        "Process timed out after \(seconds) seconds."
    }
}

/// Identifies a process beyond its pid, which the kernel reuses after exit.
enum WorkflowChildProcessIdentity {
    /// Start time of a running process in host-specific units that stay fixed for
    /// its lifetime. Nil when the process is gone or belongs to another user.
    static func startTime(of processID: Int32) -> UInt64? {
#if os(Linux)
        guard let stat = try? String(contentsOfFile: "/proc/\(processID)/stat", encoding: .utf8),
              let commandEnd = stat.lastIndex(of: ")") else {
            return nil
        }
        // Fields after the parenthesised command start at field 3; starttime is field 22.
        let fields = stat[stat.index(after: commandEnd)...].split(separator: " ")
        guard fields.count > 19 else { return nil }
        return UInt64(fields[19])
#else
        var info = proc_bsdinfo()
        let size = Int32(MemoryLayout<proc_bsdinfo>.size)
        guard proc_pidinfo(processID, PROC_PIDTBSDINFO, 0, &info, size) == size else { return nil }
        return info.pbi_start_tvsec * 1_000_000 + info.pbi_start_tvusec
#endif
    }
}

/// One `<pid>.pid` entry. Entries written before identities were recorded carry
/// no start time and are stale by definition.
struct WorkflowChildProcessRegistration: Equatable {
    let processID: Int32
    let startTime: UInt64?
}

/// Tracks worker children so cancellation and recovery can find them after the
/// worker dies. Liveness requires the recorded identity to match, so a pid reused
/// by an unrelated process is neither signalled nor treated as an active child.
enum WorkflowChildProcessRegistry {
    static let directoryName = "worker-child-pids"
    static let legacyFilename = "worker-child.pid"
    private static let lock = NSLock()

    static func register(
        _ processID: Int32,
        in runDirectory: URL,
        fileManager: FileManager = .default
    ) throws {
        try register(
            processID, startTime: WorkflowChildProcessIdentity.startTime(of: processID),
            in: runDirectory, fileManager: fileManager
        )
    }

    static func register(
        _ processID: Int32,
        startTime: UInt64?,
        in runDirectory: URL,
        fileManager: FileManager = .default
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        let directory = runDirectory.appendingPathComponent(directoryName, isDirectory: true)
        let entry = directory.appendingPathComponent("\(processID).pid")
        let lines = [String(processID)] + (startTime.map { [String($0)] } ?? [])
        do {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data(lines.joined(separator: "\n").utf8).write(to: entry, options: .atomic)
            try Data(String(processID).utf8).write(
                to: runDirectory.appendingPathComponent(legacyFilename),
                options: .atomic
            )
        } catch {
            try? fileManager.removeItem(at: entry)
            throw error
        }
    }

    static func unregister(
        _ processID: Int32,
        in runDirectory: URL,
        fileManager: FileManager = .default
    ) {
        lock.lock()
        defer { lock.unlock() }
        let entry = runDirectory
            .appendingPathComponent(directoryName, isDirectory: true)
            .appendingPathComponent("\(processID).pid")
        try? fileManager.removeItem(at: entry)
        let legacy = runDirectory.appendingPathComponent(legacyFilename)
        if readRegistration(at: legacy, fileManager: fileManager)?.processID == processID {
            try? fileManager.removeItem(at: legacy)
        }
    }

    static func processIDs(
        in runDirectory: URL,
        fileManager: FileManager = .default
    ) -> [Int32] {
        registrations(in: runDirectory, fileManager: fileManager).map(\.processID)
    }

    static func registrations(
        in runDirectory: URL,
        fileManager: FileManager = .default
    ) -> [WorkflowChildProcessRegistration] {
        let directory = runDirectory.appendingPathComponent(directoryName, isDirectory: true)
        let entries = (try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey]
        )) ?? []
        var registrations: [Int32: WorkflowChildProcessRegistration] = [:]
        for entry in entries {
            guard entry.pathExtension == "pid",
                  let values = try? entry.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey]),
                  values.isRegularFile == true,
                  values.isSymbolicLink != true,
                  let registration = readRegistration(at: entry, fileManager: fileManager) else {
                continue
            }
            registrations[registration.processID] = registration
        }
        if let legacy = readRegistration(
            at: runDirectory.appendingPathComponent(legacyFilename),
            fileManager: fileManager
        ), registrations[legacy.processID] == nil {
            registrations[legacy.processID] = legacy
        }
        return registrations.values.sorted { $0.processID < $1.processID }
    }

    /// Children still running under the identity recorded at registration. Callers
    /// hold the run lease, so no worker is registering while stale entries are pruned.
    static func activeProcessIDs(in runDirectory: URL, fileManager: FileManager = .default) -> [Int32] {
        var active: [Int32] = []
        for registration in registrations(in: runDirectory, fileManager: fileManager) {
            if isLive(registration) {
                active.append(registration.processID)
            } else {
                unregister(registration.processID, in: runDirectory, fileManager: fileManager)
            }
        }
        return active
    }

    /// Signals only children whose recorded identity still matches. A reused pid or
    /// an entry without an identity belongs to someone else and is left alone.
    @discardableResult
    static func terminateAll(
        in runDirectory: URL,
        fileManager: FileManager = .default
    ) -> [Int32] {
        let live = registrations(in: runDirectory, fileManager: fileManager).filter(isLive)
        for registration in live {
            _ = kill(registration.processID, SIGTERM)
        }
        return live.map(\.processID)
    }

    private static func isLive(_ registration: WorkflowChildProcessRegistration) -> Bool {
        guard let startTime = registration.startTime, kill(registration.processID, 0) == 0 else { return false }
        return WorkflowChildProcessIdentity.startTime(of: registration.processID) == startTime
    }

    static func clear(
        in runDirectory: URL,
        fileManager: FileManager = .default
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        let legacy = runDirectory.appendingPathComponent(legacyFilename)
        if fileManager.fileExists(atPath: legacy.path) {
            try fileManager.removeItem(at: legacy)
        }
        let directory = runDirectory.appendingPathComponent(directoryName, isDirectory: true)
        if fileManager.fileExists(atPath: directory.path) {
            try fileManager.removeItem(at: directory)
        }
    }

    /// Line one is the pid; line two, when present, is the start time recorded at registration.
    private static func readRegistration(at url: URL, fileManager: FileManager) -> WorkflowChildProcessRegistration? {
        guard fileManager.fileExists(atPath: url.path),
              let raw = try? String(contentsOf: url, encoding: .utf8) else {
            return nil
        }
        let lines = raw.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }
        guard let first = lines.first, let processID = Int32(first), processID > 1 else { return nil }
        return WorkflowChildProcessRegistration(processID: processID, startTime: lines.dropFirst().first.flatMap { UInt64($0) })
    }
}

private final class WorkflowProcessStderrTail: @unchecked Sendable {
    private static let maximumBytes = 16 * 1_024
    private let lock = NSLock()
    private var data = Data()

    func append(_ newData: Data) {
        lock.lock()
        defer { lock.unlock() }
        if newData.count >= Self.maximumBytes {
            data = Data(newData.suffix(Self.maximumBytes))
            return
        }
        data.append(newData)
        if data.count > Self.maximumBytes {
            data.removeFirst(data.count - Self.maximumBytes)
        }
    }

    var string: String {
        lock.lock()
        defer { lock.unlock() }
        return String(decoding: data, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

protocol WorkflowProcessRunning {
    func run(arguments: [String], currentDirectory: URL) throws -> WorkflowProcessResult
}

protocol WorkflowStreamingProcessRunning {
    func run(
        executable: URL,
        arguments: [String],
        currentDirectory: URL,
        timeoutSeconds: Int?,
        stdoutLineHandler: ((String) throws -> Void)?
    ) throws -> WorkflowProcessResult

    func run(
        executable: URL,
        arguments: [String],
        currentDirectory: URL,
        timeoutSeconds: Int?,
        stdoutLineHandler: ((String) throws -> Void)?,
        stdoutChunkHandler: ((Data) throws -> Void)?
    ) throws -> WorkflowProcessResult
}

/// A runner that can stop native children when its caller executes on a worker queue.
protocol WorkflowCancellableProcessRunning {
    func run(
        executable: URL, arguments: [String], currentDirectory: URL,
        timeoutSeconds: Int?, stdoutLineHandler: ((String) throws -> Void)?,
        stdoutChunkHandler: ((Data) throws -> Void)?, isCancelled: () -> Bool
    ) throws -> WorkflowProcessResult
}

extension WorkflowStreamingProcessRunning {
    // Chunk delivery is opt-in for runners that support it; others keep
    // line-based behavior and ignore raw chunks.
    func run(
        executable: URL,
        arguments: [String],
        currentDirectory: URL,
        timeoutSeconds: Int?,
        stdoutLineHandler: ((String) throws -> Void)?,
        stdoutChunkHandler: ((Data) throws -> Void)?
    ) throws -> WorkflowProcessResult {
        try run(
            executable: executable,
            arguments: arguments,
            currentDirectory: currentDirectory,
            timeoutSeconds: timeoutSeconds,
            stdoutLineHandler: stdoutLineHandler
        )
    }
}

struct WorkflowProcessRunner: WorkflowProcessRunning, WorkflowStreamingProcessRunning, WorkflowCancellableProcessRunning {
    let maximumStdoutBytes: Int

    init(maximumStdoutBytes: Int = 16 * 1024 * 1024) {
        self.maximumStdoutBytes = maximumStdoutBytes
    }

    static func stdoutCaptureURL(in directory: URL) -> URL {
        directory.appendingPathComponent(".workflow-stdout-\(UUID().uuidString)")
    }

    func run(arguments: [String], currentDirectory: URL) throws -> WorkflowProcessResult {
        try run(
            executable: CurrentExecutable.url(),
            arguments: arguments,
            currentDirectory: currentDirectory,
            timeoutSeconds: nil,
            stdoutLineHandler: nil
        )
    }

    func run(
        executable: URL,
        arguments: [String],
        currentDirectory: URL,
        timeoutSeconds: Int?,
        stdoutLineHandler: ((String) throws -> Void)?
    ) throws -> WorkflowProcessResult {
        try run(
            executable: executable,
            arguments: arguments,
            currentDirectory: currentDirectory,
            timeoutSeconds: timeoutSeconds,
            stdoutLineHandler: stdoutLineHandler,
            stdoutChunkHandler: nil
        )
    }

    func run(
        executable: URL,
        arguments: [String],
        currentDirectory: URL,
        timeoutSeconds: Int?,
        stdoutLineHandler: ((String) throws -> Void)?,
        stdoutChunkHandler: ((Data) throws -> Void)?
    ) throws -> WorkflowProcessResult {
        try run(
            executable: executable, arguments: arguments, currentDirectory: currentDirectory,
            timeoutSeconds: timeoutSeconds, stdoutLineHandler: stdoutLineHandler,
            stdoutChunkHandler: stdoutChunkHandler, isCancelled: { Task.isCancelled }
        )
    }

    func run(
        executable: URL, arguments: [String], currentDirectory: URL,
        timeoutSeconds: Int?, stdoutLineHandler: ((String) throws -> Void)?,
        stdoutChunkHandler: ((Data) throws -> Void)?, isCancelled: () -> Bool
    ) throws -> WorkflowProcessResult {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.currentDirectoryURL = currentDirectory
        process.standardInput = FileHandle.nullDevice
        let runDirectory = currentDirectory.deletingLastPathComponent().deletingLastPathComponent()
        let stderrTail = WorkflowProcessStderrTail()
        var captured = Data()
        var pending = Data()
        defer {
            if process.processIdentifier > 0 {
                WorkflowChildProcessRegistry.unregister(process.processIdentifier, in: runDirectory)
            }
        }
        let result: BoundedProcessRunner.Result
        do {
            result = try BoundedProcessRunner.run(
                process, timeout: timeoutSeconds.map(TimeInterval.init), outputLimitBytes: 0,
                isCancelled: {
                    isCancelled() || FileManager.default.fileExists(atPath: runDirectory.appendingPathComponent("cancel.request").path)
                },
                onStarted: { try WorkflowChildProcessRegistry.register($0, in: runDirectory) },
                stdoutHandler: { data in
                    guard data.count <= maximumStdoutBytes - captured.count else {
                        throw WorkflowProcessOutputLimitError(limit: maximumStdoutBytes)
                    }
                    captured.append(data)
                    try stdoutChunkHandler?(data)
                    guard stdoutLineHandler != nil else { return }
                    pending.append(data)
                    while let newline = pending.firstIndex(of: 0x0A) {
                        let lineData = pending.prefix(upTo: newline)
                        pending.removeSubrange(...newline)
                        let line = String(decoding: lineData, as: UTF8.self)
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        if !line.isEmpty { try stdoutLineHandler?(line) }
                    }
                },
                stderrHandler: { data in
                    stderrTail.append(data)
                    try? FileHandle.standardError.write(contentsOf: data)
                }
            )
        } catch is CancellationError {
            throw WorkflowCancellationError()
        }
        if !pending.isEmpty {
            let line = String(decoding: pending, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            if !line.isEmpty { try stdoutLineHandler?(line) }
        }
        if result.completion == .cancelled
            || FileManager.default.fileExists(atPath: runDirectory.appendingPathComponent("cancel.request").path) {
            throw WorkflowCancellationError()
        }
        if result.completion == .timedOut, let timeoutSeconds {
            throw WorkflowProcessTimeoutError(seconds: timeoutSeconds)
        }
        return WorkflowProcessResult(
            status: result.status,
            stdout: String(decoding: captured, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines),
            stderr: stderrTail.string,
            terminationReason: WorkflowProcessTerminationReason(process.terminationReason)
        )
    }
}

struct WorkflowProcessOutputLimitError: LocalizedError {
    let limit: Int

    var errorDescription: String? {
        "Workflow child stdout exceeded the \(limit)-byte limit. Write large outputs as declared artifacts."
    }
}
