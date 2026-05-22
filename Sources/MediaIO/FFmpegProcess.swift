import Foundation
#if os(Linux)
import Glibc
#endif

enum FFmpegProcess {
    struct Result {
        let stdout: Data
        let stderr: String
    }

    static func run(
        tool: String,
        arguments: [String],
        stdin: Data? = nil
    ) throws -> Result {
        let executable = try resolveTool(tool)
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

        var command = ([executable.path] + arguments).map(shellQuote).joined(separator: " ")
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

    private static func shellQuote(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
    }
}
