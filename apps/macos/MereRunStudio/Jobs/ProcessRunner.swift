import Foundation

/// Decodes a byte stream into UTF-8 incrementally, retaining any incomplete trailing
/// multibyte sequence until the next read so codepoints split across pipe reads are
/// never dropped. Each stream owns its own decoder; the readability queue is serial
/// per file handle, so no locking is required within a single stream.
final class IncrementalUTF8Decoder: @unchecked Sendable {
    private var buffer = Data()
    private static let lossyFlushThreshold = 1 << 20

    func push(_ data: Data) -> String? {
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

    func flush() -> String? {
        guard !buffer.isEmpty else { return nil }
        let decoded = String(decoding: buffer, as: UTF8.self)
        buffer.removeAll(keepingCapacity: false)
        return decoded.isEmpty ? nil : decoded
    }
}

struct MereRunProcessConfiguration: Equatable {
    let executableURL: URL
    let arguments: [String]
    let currentDirectoryURL: URL
    let environment: [String: String]
    let keepsStandardInputOpen: Bool
}

enum MereRunProcessInputError: LocalizedError {
    case unavailable

    var errorDescription: String? {
        "This command does not accept interactive input."
    }
}

protocol MereRunRunningProcess: AnyObject {
    func terminate()
    func interrupt()
    func sendStandardInput(_ text: String) throws
}

extension MereRunRunningProcess {
    func interrupt() {
        terminate()
    }

    func sendStandardInput(_ text: String) throws {
        throw MereRunProcessInputError.unavailable
    }
}

protocol MereRunProcessRunning: AnyObject {
    func start(
        configuration: MereRunProcessConfiguration,
        stdout: @escaping @Sendable (String) -> Void,
        stderr: @escaping @Sendable (String) -> Void,
        termination: @escaping @Sendable (Int32) -> Void
    ) throws -> MereRunRunningProcess
}

/// A seam over filesystem existence checks so run-output detection can be unit-tested without
/// touching the real disk. `FileManager` is the production implementation.
protocol MereRunFileProbing {
    func fileExists(atPath path: String) -> Bool
}

extension FileManager: MereRunFileProbing {}

final class FoundationRunningProcess: MereRunRunningProcess, @unchecked Sendable {
    private let process: Process
    private let stdinPipe: Pipe?
    private let stdoutPipe: Pipe
    private let stderrPipe: Pipe

    init(process: Process, stdinPipe: Pipe?, stdoutPipe: Pipe, stderrPipe: Pipe) {
        self.process = process
        self.stdinPipe = stdinPipe
        self.stdoutPipe = stdoutPipe
        self.stderrPipe = stderrPipe
    }

    func terminate() {
        try? stdinPipe?.fileHandleForWriting.close()
        process.terminate()
    }

    func interrupt() {
        process.interrupt()
    }

    func sendStandardInput(_ text: String) throws {
        guard let stdinPipe else {
            throw MereRunProcessInputError.unavailable
        }
        try stdinPipe.fileHandleForWriting.write(contentsOf: Data(text.utf8))
    }

    func cleanup() {
        try? stdinPipe?.fileHandleForWriting.close()
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        stderrPipe.fileHandleForReading.readabilityHandler = nil
    }

    func waitUntilExit() -> Int32 {
        process.waitUntilExit()
        cleanup()
        return process.terminationStatus
    }
}

final class FoundationMereRunProcessRunner: MereRunProcessRunning {
    func start(
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
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        let runningProcess = FoundationRunningProcess(
            process: process,
            stdinPipe: stdinPipe,
            stdoutPipe: stdoutPipe,
            stderrPipe: stderrPipe
        )

        let stdoutDecoder = IncrementalUTF8Decoder()
        let stderrDecoder = IncrementalUTF8Decoder()

        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                if let tail = stdoutDecoder.flush() { stdout(tail) }
                return
            }
            if let text = stdoutDecoder.push(data) { stdout(text) }
        }

        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                if let tail = stderrDecoder.flush() { stderr(tail) }
                return
            }
            if let text = stderrDecoder.push(data) { stderr(text) }
        }

        try process.run()

        DispatchQueue.global(qos: .userInitiated).async {
            termination(runningProcess.waitUntilExit())
        }

        return runningProcess
    }
}
