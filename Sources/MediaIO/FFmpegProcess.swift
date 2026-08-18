import Foundation
#if os(Linux)
import Glibc
#endif

enum FFmpegProcess {
    struct Result {
        let stdout: Data
        let stderr: String
    }

    #if os(iOS)
    // ffmpeg tool spawning needs child processes, which iOS forbids; media
    // conversion happens on nodes, never on the phone.
    static func run(
        tool: String,
        arguments: [String],
        stdin: Data? = nil
    ) throws -> Result {
        throw FFmpegProcessUnavailableError()
    }

    static func runStreamingInput(
        tool: String,
        arguments: [String],
        writeInput: (FileHandle) throws -> Void
    ) throws -> Result {
        throw FFmpegProcessUnavailableError()
    }
    #else
    static func run(
        tool: String,
        arguments: [String],
        stdin: Data? = nil
    ) throws -> Result {
        let executable = try resolveTool(tool)
        let effectiveArguments = argumentsWithDisabledStdin(
            for: executable,
            rawTool: tool,
            arguments: arguments
        )
        let scratchID = UUID().uuidString
        let tempRoot = FileManager.default.temporaryDirectory
        let stdoutURL = tempRoot.appendingPathComponent("mererun-ffmpeg-\(scratchID).stdout")
        let stderrURL = tempRoot.appendingPathComponent("mererun-ffmpeg-\(scratchID).stderr")
        let stdinURL = tempRoot.appendingPathComponent("mererun-ffmpeg-\(scratchID).stdin")
        _ = FileManager.default.createFile(atPath: stdoutURL.path, contents: nil)
        _ = FileManager.default.createFile(atPath: stderrURL.path, contents: nil)
        defer {
            try? FileManager.default.removeItem(at: stdoutURL)
            try? FileManager.default.removeItem(at: stderrURL)
            try? FileManager.default.removeItem(at: stdinURL)
        }

        if let stdin {
            try stdin.write(to: stdinURL)
        }

        var command = ([executable.path] + effectiveArguments).map(shellQuote).joined(separator: " ")
        if stdin != nil {
            command += " < \(shellQuote(stdinURL.path))"
        }
        command += " > \(shellQuote(stdoutURL.path)) 2> \(shellQuote(stderrURL.path))"

        #if os(Linux)
        let status = system(command)
        #elseif os(macOS)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        process.arguments = ["-c", command]
        do {
            try process.run()
        } catch {
            throw MediaIOError.missingTool(tool)
        }
        process.waitUntilExit()
        let status = process.terminationStatus
        #else
        throw MediaIOError.unsupportedPlatform("The FFmpeg MediaIO backend is only available on Linux and macOS.")
        #endif

        let stdout = try Data(contentsOf: stdoutURL)
        let stderrData = try Data(contentsOf: stderrURL)
        let stderr = String(decoding: stderrData, as: UTF8.self)
        guard status == 0 else {
            throw MediaIOError.processFailed(
                tool: tool,
                status: status,
                stderr: stderr
            )
        }

        return Result(stdout: stdout, stderr: stderr)
    }

    /// Runs a tool while the caller incrementally produces standard input.
    /// stdout/stderr are file-backed so neither side of the pipe can deadlock
    /// on an unread diagnostic buffer while large media inputs are encoded.
    static func runStreamingInput(
        tool: String,
        arguments: [String],
        writeInput: (FileHandle) throws -> Void
    ) throws -> Result {
        let executable = try resolveTool(tool)
        let effectiveArguments = argumentsWithDisabledStdin(
            for: executable,
            rawTool: tool,
            arguments: arguments
        )
        let scratchID = UUID().uuidString
        let tempRoot = FileManager.default.temporaryDirectory
        let stdoutURL = tempRoot.appendingPathComponent("mererun-ffmpeg-\(scratchID).stdout")
        let stderrURL = tempRoot.appendingPathComponent("mererun-ffmpeg-\(scratchID).stderr")
        _ = FileManager.default.createFile(atPath: stdoutURL.path, contents: nil)
        _ = FileManager.default.createFile(atPath: stderrURL.path, contents: nil)
        defer {
            try? FileManager.default.removeItem(at: stdoutURL)
            try? FileManager.default.removeItem(at: stderrURL)
        }

        let stdoutHandle = try FileHandle(forWritingTo: stdoutURL)
        let stderrHandle = try FileHandle(forWritingTo: stderrURL)
        let stdinPipe = Pipe()
        let process = Process()
        process.executableURL = executable
        process.arguments = effectiveArguments
        process.standardInput = stdinPipe
        process.standardOutput = stdoutHandle
        process.standardError = stderrHandle

        do {
            try process.run()
        } catch {
            try? stdoutHandle.close()
            try? stderrHandle.close()
            throw MediaIOError.missingTool(tool)
        }

        do {
            try writeInput(stdinPipe.fileHandleForWriting)
            try stdinPipe.fileHandleForWriting.close()
        } catch {
            try? stdinPipe.fileHandleForWriting.close()
            process.terminate()
            process.waitUntilExit()
            try? stdoutHandle.close()
            try? stderrHandle.close()
            throw error
        }

        process.waitUntilExit()
        try stdoutHandle.close()
        try stderrHandle.close()
        let stdout = try Data(contentsOf: stdoutURL)
        let stderr = String(decoding: try Data(contentsOf: stderrURL), as: UTF8.self)
        guard process.terminationStatus == 0 else {
            throw MediaIOError.processFailed(
                tool: tool,
                status: process.terminationStatus,
                stderr: stderr
            )
        }
        return Result(stdout: stdout, stderr: stderr)
    }

    private static func resolveTool(_ tool: String) throws -> URL {
        let raw = tool
        if raw.contains("/") {
            let url = URL(fileURLWithPath: raw)
            guard FileManager.default.isExecutableFile(atPath: url.path) else {
                throw MediaIOError.missingTool(tool)
            }
            return url
        }

        let pathEntries = ProcessInfo.processInfo.environment["PATH"]?.split(separator: ":") ?? []
        for entry in pathEntries {
            let candidate = URL(fileURLWithPath: String(entry), isDirectory: true)
                .appendingPathComponent(raw)
            if FileManager.default.isExecutableFile(atPath: candidate.path) {
                return candidate
            }
        }
        throw MediaIOError.missingTool(tool)
    }

    private static func argumentsWithDisabledStdin(
        for executable: URL,
        rawTool: String,
        arguments: [String]
    ) -> [String] {
        let executableName = executable.lastPathComponent.lowercased()
        let requestedName = URL(fileURLWithPath: rawTool).lastPathComponent.lowercased()
        guard executableName == "ffmpeg" || requestedName == "ffmpeg" else {
            return arguments
        }
        guard !arguments.contains("-nostdin") else {
            return arguments
        }
        return ["-nostdin"] + arguments
    }

    private static func shellQuote(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
    }
    #endif
}

struct FFmpegProcessUnavailableError: Error {}
