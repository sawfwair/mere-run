import Foundation
#if os(Linux)
import Glibc
#else
import Darwin
#endif

/// Runs an already configured child while draining both output streams. Capture
/// limits bound retained bytes, never how much output the child can write.
enum BoundedProcessRunner {
    enum Completion: Equatable, Sendable {
        case exited
        case timedOut
        case cancelled
    }

    struct Capture: Sendable {
        let text: String
        let truncated: Bool
    }

    struct Result: Sendable {
        let completion: Completion
        let status: Int32
        let seconds: Double
        let stdout: Capture
        let stderr: Capture
    }

    enum RunError: LocalizedError {
        case invalidLimits
        case outputIO(Int32)

        var errorDescription: String? {
            switch self {
            case .invalidLimits:
                return "Process timeout must be positive and finite, and output limit must be nonnegative."
            case .outputIO(let code):
                return "Could not read child process output (errno \(code))."
            }
        }
    }

    static func run(
        _ process: Process,
        timeout: TimeInterval? = nil,
        outputLimitBytes: Int = 256 * 1024,
        combineOutput: Bool = false,
        isCancelled: () -> Bool = { Task.isCancelled },
        onStarted: ((Int32) throws -> Void)? = nil,
        stdoutHandler: ((Data) throws -> Void)? = nil,
        stderrHandler: ((Data) throws -> Void)? = nil
    ) throws -> Result {
        guard timeout.map({ $0.isFinite && $0 > 0 }) ?? true, outputLimitBytes >= 0 else {
            throw RunError.invalidLimits
        }
        let stdout = try Output(limitBytes: outputLimitBytes)
        defer { stdout.close() }
        let stderr = try combineOutput ? nil : Output(limitBytes: outputLimitBytes)
        defer { stderr?.close() }
        process.standardOutput = stdout.pipe
        process.standardError = stderr?.pipe ?? stdout.pipe

        if isCancelled() { throw CancellationError() }
        let start = ProcessInfo.processInfo.systemUptime
        try process.run()
        // Only the child owns writers after launch; parent copies would prevent EOF.
        stdout.closeWriter()
        stderr?.closeWriter()

        var completion = Completion.exited
        do {
            try onStarted?(process.processIdentifier)
            while true {
                // One chunk per stream keeps a noisy stdout from starving stderr,
                // cancellation, or the deadline. Reads themselves never block.
                let readOutput = try stdout.drain(handler: stdoutHandler)
                let readError = try stderr?.drain(handler: stderrHandler) ?? false
                if !process.isRunning, stdout.atEnd, stderr?.atEnd ?? true { break }
                if isCancelled() {
                    completion = .cancelled
                    break
                }
                if let timeout, ProcessInfo.processInfo.systemUptime - start >= timeout {
                    completion = .timedOut
                    break
                }
                if !readOutput, !readError { Thread.sleep(forTimeInterval: 0.01) }
            }
            if completion != .exited { stop(process) }
            process.waitUntilExit()
            // Retain available tail output after termination without waiting for
            // a deliberately detached descendant that still holds a pipe open.
            for _ in 0..<16 {
                let readOutput = try stdout.drain(handler: stdoutHandler)
                let readError = try stderr?.drain(handler: stderrHandler) ?? false
                if !readOutput, !readError { break }
            }
        } catch {
            stop(process)
            process.waitUntilExit()
            throw error
        }
        return Result(
            completion: completion,
            status: process.terminationStatus,
            seconds: ProcessInfo.processInfo.systemUptime - start,
            stdout: stdout.capture,
            stderr: stderr?.capture ?? Capture(text: "", truncated: false)
        )
    }

    private static func stop(_ process: Process) {
        // Foundation launches Process in its own process group on macOS and
        // Linux. Signal the group even if its leader exited but a child retains
        // stdout/stderr; never signal the caller's group.
        let pid = process.processIdentifier
        guard pid > 0, pid != getpgrp() else { return }
        kill(-pid, SIGTERM)
        if process.isRunning { process.terminate() }
        Thread.sleep(forTimeInterval: 0.2)
        kill(-pid, SIGKILL)
        if process.isRunning { kill(pid, SIGKILL) }
    }

    private final class Output {
        let pipe = Pipe()
        private(set) var atEnd = false
        private let limitBytes: Int
        private var bytes = Data()
        private var truncated = false
        private var buffer = [UInt8](repeating: 0, count: 64 * 1024)

        init(limitBytes: Int) throws {
            self.limitBytes = limitBytes
            let fd = pipe.fileHandleForReading.fileDescriptor
            let flags = fcntl(fd, F_GETFL)
            guard flags != -1, fcntl(fd, F_SETFL, flags | O_NONBLOCK) != -1 else {
                let code = errno
                close()
                throw RunError.outputIO(code)
            }
        }

        func drain(handler: ((Data) throws -> Void)?) throws -> Bool {
            guard !atEnd else { return false }
            let count = buffer.withUnsafeMutableBytes {
                read(pipe.fileHandleForReading.fileDescriptor, $0.baseAddress, $0.count)
            }
            if count == 0 {
                atEnd = true
                return false
            }
            if count < 0 {
                let code = errno
                if code == EAGAIN || code == EWOULDBLOCK || code == EINTR { return false }
                throw RunError.outputIO(code)
            }
            try handler?(Data(buffer.prefix(count)))
            let retained = min(count, limitBytes - bytes.count)
            bytes.append(contentsOf: buffer.prefix(retained))
            if retained < count { truncated = true }
            return true
        }

        var capture: Capture {
            let text = String(decoding: bytes, as: UTF8.self)
            return Capture(text: truncated ? text + "\n[output truncated]" : text, truncated: truncated)
        }

        func closeWriter() { try? pipe.fileHandleForWriting.close() }

        func close() {
            closeWriter()
            try? pipe.fileHandleForReading.close()
        }
    }
}
