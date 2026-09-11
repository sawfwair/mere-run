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

enum WorkflowChildProcessRegistry {
    static let directoryName = "worker-child-pids"
    static let legacyFilename = "worker-child.pid"
    private static let lock = NSLock()

    static func register(
        _ processID: Int32,
        in runDirectory: URL,
        fileManager: FileManager = .default
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        let directory = runDirectory.appendingPathComponent(directoryName, isDirectory: true)
        let entry = directory.appendingPathComponent("\(processID).pid")
        do {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data(String(processID).utf8).write(to: entry, options: .atomic)
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
        if readProcessID(at: legacy, fileManager: fileManager) == processID {
            try? fileManager.removeItem(at: legacy)
        }
    }

    static func processIDs(
        in runDirectory: URL,
        fileManager: FileManager = .default
    ) -> [Int32] {
        let directory = runDirectory.appendingPathComponent(directoryName, isDirectory: true)
        let entries = (try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey]
        )) ?? []
        var processIDs = Set(entries.compactMap { entry -> Int32? in
            guard entry.pathExtension == "pid",
                  let values = try? entry.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey]),
                  values.isRegularFile == true,
                  values.isSymbolicLink != true else {
                return nil
            }
            return readProcessID(at: entry, fileManager: fileManager)
        })
        if let legacy = readProcessID(
            at: runDirectory.appendingPathComponent(legacyFilename),
            fileManager: fileManager
        ) {
            processIDs.insert(legacy)
        }
        return processIDs.sorted()
    }

    static func activeProcessIDs(in runDirectory: URL, fileManager: FileManager = .default) -> [Int32] {
        processIDs(in: runDirectory, fileManager: fileManager).filter { pid in
            kill(pid, 0) == 0 || errno == EPERM
        }
    }

    @discardableResult
    static func terminateAll(
        in runDirectory: URL,
        fileManager: FileManager = .default
    ) -> [Int32] {
        let processIDs = processIDs(in: runDirectory, fileManager: fileManager)
        for processID in processIDs {
            _ = kill(processID, SIGTERM)
        }
        return processIDs
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

    private static func readProcessID(at url: URL, fileManager: FileManager) -> Int32? {
        guard fileManager.fileExists(atPath: url.path),
              let raw = try? String(contentsOf: url, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines),
              let processID = Int32(raw),
              processID > 1 else {
            return nil
        }
        return processID
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

struct WorkflowProcessRunner: WorkflowProcessRunning, WorkflowStreamingProcessRunning {
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
                    Task.isCancelled || FileManager.default.fileExists(atPath: runDirectory.appendingPathComponent("cancel.request").path)
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
