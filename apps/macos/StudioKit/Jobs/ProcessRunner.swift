import Foundation
import Darwin

/// Decodes a byte stream into UTF-8 incrementally, retaining any incomplete trailing
/// multibyte sequence until the next read so codepoints split across pipe reads are
/// never dropped. Each stream owns its own decoder; the readability queue is serial
/// per file handle, so no locking is required within a single stream.
package final class IncrementalUTF8Decoder: @unchecked Sendable {
    private var buffer = Data()
    private static let lossyFlushThreshold = 1 << 20

    package func push(_ data: Data) -> String? {
        buffer.append(data)
        guard !buffer.isEmpty else { return nil }

        // A well-formed UTF-8 stream only fails to decode when the final codepoint is
        // truncated (at most 3 trailing bytes). Trim up to 3 bytes to find the boundary.
        let maxBackoff = min(3, buffer.count)
        for back in 0...maxBackoff {
            let length = buffer.count - back
            guard length > 0 else { break }
            if let decoded = String(data: buffer.prefix(length), encoding: .utf8) {
                buffer.removeFirst(length)
                return decoded.isEmpty ? nil : decoded
            }
        }

        // Genuinely malformed mid-stream bytes (should not happen from the CLI): avoid
        // unbounded buffering by flushing lossily once the backlog grows too large.
        if buffer.count > Self.lossyFlushThreshold {
            return flush()
        }
        return nil
    }

    package func flush() -> String? {
        guard !buffer.isEmpty else { return nil }
        let decoded = String(decoding: buffer, as: UTF8.self)
        buffer.removeAll(keepingCapacity: false)
        return decoded.isEmpty ? nil : decoded
    }
}

package struct MereRunProcessConfiguration: Equatable {
    package let executableURL: URL
    package let arguments: [String]
    package let currentDirectoryURL: URL
    package let environment: [String: String]
    package let keepsStandardInputOpen: Bool
}

package enum MereRunProcessInputError: LocalizedError {
    case unavailable

    package var errorDescription: String? {
        "This command does not accept interactive input."
    }
}

package protocol MereRunRunningProcess: AnyObject {
    func terminate()
    func interrupt()
    func sendStandardInput(_ text: String) throws
}

extension MereRunRunningProcess {
    package func interrupt() {
        terminate()
    }

    package func sendStandardInput(_ text: String) throws {
        throw MereRunProcessInputError.unavailable
    }
}

package protocol MereRunProcessRunning: AnyObject {
    func start(
        configuration: MereRunProcessConfiguration,
        stdout: @escaping @Sendable (String) -> Void,
        stderr: @escaping @Sendable (String) -> Void,
        termination: @escaping @Sendable (Int32) -> Void
    ) throws -> MereRunRunningProcess
}

/// A seam over filesystem existence checks so run-output detection can be unit-tested without
/// touching the real disk. `FileManager` is the production implementation.
package protocol MereRunFileProbing {
    func fileExists(atPath path: String) -> Bool
}

extension FileManager: MereRunFileProbing {}

package final class FoundationRunningProcess: MereRunRunningProcess, @unchecked Sendable {
    private let process: Process
    private let stdinPipe: Pipe?
    private let stdout: ProcessOutputReader
    private let stderr: ProcessOutputReader
    private let lock = NSLock()
    private let inputLock = NSLock()
    private var cancellationRequested = false

    fileprivate init(process: Process, stdinPipe: Pipe?, stdout: ProcessOutputReader, stderr: ProcessOutputReader) {
        self.process = process
        self.stdinPipe = stdinPipe
        self.stdout = stdout
        self.stderr = stderr
    }

    package func terminate() {
        lock.withLock {
            cancellationRequested = true
            try? stdinPipe?.fileHandleForWriting.close()
        }
    }

    package func interrupt() {
        process.interrupt()
    }

    package func sendStandardInput(_ text: String) throws {
        guard let stdinPipe else {
            throw MereRunProcessInputError.unavailable
        }
        inputLock.lock()
        defer { inputLock.unlock() }
        let data = Data(text.utf8)
        var offset = 0
        while offset < data.count {
            let written = try lock.withLock {
                guard !cancellationRequested, process.isRunning else { throw MereRunProcessInputError.unavailable }
                let count = data.withUnsafeBytes { bytes in
                    Darwin.write(stdinPipe.fileHandleForWriting.fileDescriptor,
                                 bytes.baseAddress?.advanced(by: offset), data.count - offset)
                }
                if count < 0 {
                    guard errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR else {
                        throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
                    }
                    return 0
                }
                return count
            }
            offset += written
            if written == 0 { Thread.sleep(forTimeInterval: 0.01) }
        }
    }

    package func waitUntilExit() -> Int32 {
        defer {
            lock.withLock { try? stdinPipe?.fileHandleForWriting.close() }
            stdout.close()
            stderr.close()
        }
        var stopStarted: TimeInterval?
        var didEscalate = false
        var exitedAt: TimeInterval?
        var outputError = false
        while true {
            let readOutput = stdout.drain()
            let readError = stderr.drain()
            outputError = outputError || stdout.failure != nil || stderr.failure != nil
            let now = ProcessInfo.processInfo.systemUptime
            let running = process.isRunning
            let cancelled = lock.withLock { cancellationRequested }
            if stopStarted == nil, cancelled || outputError || (!running && ownsLiveGroup) {
                signalGroup(SIGTERM)
                stopStarted = now
            }
            if let stopStarted, !didEscalate, now - stopStarted >= 0.2 {
                signalGroup(SIGKILL)
                didEscalate = true
            }
            if !running {
                if exitedAt == nil { exitedAt = now }
                if stdout.atEnd && stderr.atEnd && !ownsLiveGroup { break }
                // A deliberately detached descendant can retain a pipe without
                // belonging to our group. Bound final drainage after parent exit.
                if let exitedAt, now - exitedAt >= 0.4 { break }
            }
            if !readOutput && !readError { Thread.sleep(forTimeInterval: 0.01) }
        }
        // The loop already observed process exit. A second Foundation wait can
        // strand this worker in its run loop after the child has finished.
        stdout.finish()
        stderr.finish()
        if let error = stdout.failure ?? stderr.failure {
            stderr.report("Could not read process output: \(error.localizedDescription)\n")
            return -1
        }
        return process.terminationStatus
    }

    private var ownsLiveGroup: Bool {
        let pid = process.processIdentifier
        return pid > 0 && pid != getpgrp() && kill(-pid, 0) == 0
    }

    private func signalGroup(_ signal: Int32) {
        // Foundation gives this child its own process group. Match the CLI's
        // bounded process ownership and never signal the Studio process group.
        let pid = process.processIdentifier
        guard pid > 0, pid != getpgrp() else { return }
        kill(-pid, signal)
        if process.isRunning { kill(pid, signal) }
    }
}

/// Both readers are drained on the process waiter queue. Decoding and callbacks
/// finish before completion is delivered, including after a short-lived child.
private final class ProcessOutputReader {
    let pipe = Pipe()
    private let decoder = IncrementalUTF8Decoder()
    private let output: @Sendable (String) -> Void
    private var buffer = [UInt8](repeating: 0, count: 64 * 1_024)
    private(set) var atEnd = false
    private(set) var failure: POSIXError?

    init(output: @escaping @Sendable (String) -> Void) throws {
        self.output = output
        let descriptor = pipe.fileHandleForReading.fileDescriptor
        let flags = fcntl(descriptor, F_GETFL)
        guard flags != -1, fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) != -1 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
    }

    func report(_ message: String) { output(message) }

    func closeWriter() { try? pipe.fileHandleForWriting.close() }

    @discardableResult
    func drain() -> Bool {
        guard !atEnd else { return false }
        let count = buffer.withUnsafeMutableBytes {
            Darwin.read(pipe.fileHandleForReading.fileDescriptor, $0.baseAddress, $0.count)
        }
        if count == 0 { atEnd = true; return false }
        if count < 0 {
            if errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR {
                failure = POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
                atEnd = true
            }
            return false
        }
        if let text = decoder.push(Data(buffer.prefix(count))) { output(text) }
        return true
    }

    func finish() {
        for _ in 0..<16 where drain() {}
        if let text = decoder.flush() { output(text) }
    }

    func close() {
        try? pipe.fileHandleForReading.close()
        closeWriter()
    }
}

package final class FoundationMereRunProcessRunner: MereRunProcessRunning {
    package func start(
        configuration: MereRunProcessConfiguration,
        stdout: @escaping @Sendable (String) -> Void,
        stderr: @escaping @Sendable (String) -> Void,
        termination: @escaping @Sendable (Int32) -> Void
    ) throws -> MereRunRunningProcess {
        let process = Process()
        process.executableURL = configuration.executableURL
        process.arguments = configuration.arguments
        process.currentDirectoryURL = configuration.currentDirectoryURL
        process.environment = configuration.environment

        let stdinPipe = configuration.keepsStandardInputOpen ? Pipe() : nil
        if let stdinPipe {
            let descriptor = stdinPipe.fileHandleForWriting.fileDescriptor
            let flags = fcntl(descriptor, F_GETFL)
            guard flags != -1,
                  fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) != -1,
                  fcntl(descriptor, F_SETNOSIGPIPE, 1) != -1 else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
        }
        let stdoutReader = try ProcessOutputReader(output: stdout)
        let stderrReader = try ProcessOutputReader(output: stderr)
        process.standardInput = stdinPipe
        process.standardOutput = stdoutReader.pipe
        process.standardError = stderrReader.pipe

        let runningProcess = FoundationRunningProcess(
            process: process,
            stdinPipe: stdinPipe,
            stdout: stdoutReader,
            stderr: stderrReader
        )
        try process.run()
        stdoutReader.closeWriter()
        stderrReader.closeWriter()
        try? stdinPipe?.fileHandleForReading.close()

        DispatchQueue.global(qos: .userInitiated).async {
            termination(runningProcess.waitUntilExit())
        }

        return runningProcess
    }
}
