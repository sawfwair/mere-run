import ArgumentParser
import Foundation

#if os(macOS)
import Darwin
#elseif os(Linux)
import Glibc
#endif

enum CodeExecutionSandboxMode: String, CaseIterable, ExpressibleByArgument, Codable {
    case auto
    case macOSSandboxExec = "macos-sandbox-exec"
    case bubblewrap
    case none
}

enum CodeExecutionSandboxBackend: String, Codable {
    case macOSSandboxExec = "macos-sandbox-exec"
    case bubblewrap
    case none
}

struct CodeExecutionSandboxResult {
    let backend: CodeExecutionSandboxBackend
    let passed: Bool
    let seconds: Double
    let timedOut: Bool
    let status: Int32
    let stdout: String
    let stderr: String
    let stdoutTruncated: Bool
    let stderrTruncated: Bool

    var errorSummary: String? {
        guard !passed else {
            return nil
        }
        if timedOut {
            return "Timed out after \(String(format: "%.2f", seconds))s."
        }
        let detail = [stderr, stdout]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .first { !$0.isEmpty } ?? "Process exited with status \(status)."
        return Self.tail(detail)
    }

    private static func tail(_ text: String, maxCharacters: Int = 800) -> String {
        guard text.count > maxCharacters else {
            return text
        }
        return String(text.suffix(maxCharacters))
    }
}

enum CodeExecutionSandbox {
    static let defaultOutputLimitBytes = 256 * 1024

    static func preflight(mode: CodeExecutionSandboxMode) throws {
        _ = try resolveBackend(mode: mode)
    }

    static func runPython(
        program: String,
        python: String,
        mode: CodeExecutionSandboxMode,
        timeout: Double,
        outputLimitBytes: Int = defaultOutputLimitBytes
    ) throws -> CodeExecutionSandboxResult {
        let fileManager = FileManager.default
        let root = fileManager.temporaryDirectory
            .appendingPathComponent("mererun-code-benchmark-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        let workDir = root.resolvingSymlinksInPath()
        defer {
            try? fileManager.removeItem(at: root)
            if workDir != root {
                try? fileManager.removeItem(at: workDir)
            }
        }

        let scriptURL = workDir.appendingPathComponent("candidate.py")
        try program.write(to: scriptURL, atomically: true, encoding: .utf8)

        let backend = try resolveBackend(mode: mode)
        let process = Process()
        process.currentDirectoryURL = workDir
        process.environment = sandboxEnvironment(workDir: workDir)

        switch backend {
        case .macOSSandboxExec:
            process.executableURL = URL(fileURLWithPath: "/usr/bin/sandbox-exec")
            process.arguments = [
                "-p",
                macOSSandboxProfile(workDir: workDir),
            ] + pythonInvocation(python: python, scriptPath: "candidate.py")

        case .bubblewrap:
            let bubblewrap = try requireExecutable(named: "bwrap")
            process.executableURL = URL(fileURLWithPath: bubblewrap)
            process.arguments = bubblewrapArguments(
                workDir: workDir,
                pythonArguments: pythonInvocation(python: python, scriptPath: "candidate.py")
            )

        case .none:
            let invocation = pythonInvocation(python: python, scriptPath: "candidate.py")
            process.executableURL = URL(fileURLWithPath: invocation[0])
            process.arguments = Array(invocation.dropFirst())
        }

        let stdout = LimitedOutputPipe(limitBytes: outputLimitBytes)
        let stderr = LimitedOutputPipe(limitBytes: outputLimitBytes)
        process.standardOutput = stdout.pipe
        process.standardError = stderr.pipe

        let start = Date()
        try process.run()
        var timedOut = false
        while process.isRunning {
            if Date().timeIntervalSince(start) > timeout {
                timedOut = true
                terminate(process)
                break
            }
            Thread.sleep(forTimeInterval: 0.05)
        }
        process.waitUntilExit()
        let seconds = Date().timeIntervalSince(start)
        let stdoutCapture = stdout.finish()
        let stderrCapture = stderr.finish()

        return CodeExecutionSandboxResult(
            backend: backend,
            passed: !timedOut && process.terminationStatus == 0,
            seconds: seconds,
            timedOut: timedOut,
            status: process.terminationStatus,
            stdout: stdoutCapture.text,
            stderr: stderrCapture.text,
            stdoutTruncated: stdoutCapture.truncated,
            stderrTruncated: stderrCapture.truncated
        )
    }

    private static func resolveBackend(mode: CodeExecutionSandboxMode) throws -> CodeExecutionSandboxBackend {
        switch mode {
        case .auto:
            #if os(macOS)
            if executableExists(at: "/usr/bin/sandbox-exec") {
                return .macOSSandboxExec
            }
            #elseif os(Linux)
            if findExecutable(named: "bwrap") != nil {
                return .bubblewrap
            }
            #endif
            throw ValidationError(
                "No supported code sandbox backend is available. Install bubblewrap on Linux or pass --sandbox none."
            )

        case .macOSSandboxExec:
            guard executableExists(at: "/usr/bin/sandbox-exec") else {
                throw ValidationError("macOS sandbox-exec is not available on this host.")
            }
            return .macOSSandboxExec

        case .bubblewrap:
            guard findExecutable(named: "bwrap") != nil else {
                throw ValidationError("bubblewrap sandbox is not available on this host.")
            }
            return .bubblewrap

        case .none:
            return .none
        }
    }

    private static func sandboxEnvironment(workDir: URL) -> [String: String] {
        [
            "PATH": ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": workDir.path,
            "TMPDIR": workDir.path,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        ]
    }

    private static func pythonInvocation(python: String, scriptPath: String) -> [String] {
        if python.contains("/") {
            return [python, scriptPath]
        }
        return ["/usr/bin/env", python, scriptPath]
    }

    private static func macOSSandboxProfile(workDir: URL) -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser
            .resolvingSymlinksInPath()
            .path
        let escapedHome = sandboxEscaped(home)
        let escapedWorkDir = sandboxEscaped(workDir.path)
        return """
        (version 1)
        (deny default)
        (allow process*)
        (allow sysctl-read)
        (allow mach-lookup)
        (allow file-read*)
        (deny file-read* (subpath "\(escapedHome)"))
        (allow file-read* (subpath "\(escapedWorkDir)"))
        (allow file-write* (subpath "\(escapedWorkDir)"))
        (deny network*)
        """
    }

    private static func bubblewrapArguments(workDir: URL, pythonArguments: [String]) -> [String] {
        [
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--proc", "/proc",
            "--dev", "/dev",
            "--ro-bind", "/", "/",
            "--bind", workDir.path, workDir.path,
            "--chdir", workDir.path,
            "--setenv", "PATH", ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin",
            "--setenv", "HOME", workDir.path,
            "--setenv", "TMPDIR", workDir.path,
            "--setenv", "PYTHONIOENCODING", "utf-8",
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--setenv", "PYTHONNOUSERSITE", "1",
        ] + pythonArguments
    }

    private static func sandboxEscaped(_ string: String) -> String {
        string
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }

    private static func requireExecutable(named name: String) throws -> String {
        guard let path = findExecutable(named: name) else {
            throw ValidationError("Required executable not found: \(name).")
        }
        return path
    }

    private static func findExecutable(named name: String) -> String? {
        if name.contains("/") {
            return executableExists(at: name) ? name : nil
        }
        let path = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
        for directory in path.split(separator: ":").map(String.init) {
            let candidate = URL(fileURLWithPath: directory).appendingPathComponent(name).path
            if executableExists(at: candidate) {
                return candidate
            }
        }
        return nil
    }

    private static func executableExists(at path: String) -> Bool {
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory)
            && !isDirectory.boolValue
            && FileManager.default.isExecutableFile(atPath: path)
    }

    private static func terminate(_ process: Process) {
        process.terminate()
        Thread.sleep(forTimeInterval: 0.2)
        guard process.isRunning else {
            return
        }
        #if os(macOS) || os(Linux)
        kill(pid_t(process.processIdentifier), SIGKILL)
        #else
        process.terminate()
        #endif
    }
}

private final class LimitedOutputPipe: @unchecked Sendable {
    struct Capture {
        let text: String
        let truncated: Bool
    }

    let pipe = Pipe()

    private let limitBytes: Int
    private let lock = NSLock()
    private var data = Data()
    private var truncated = false

    init(limitBytes: Int) {
        self.limitBytes = limitBytes
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            guard !chunk.isEmpty else {
                return
            }
            self?.append(chunk)
        }
    }

    func finish() -> Capture {
        pipe.fileHandleForReading.readabilityHandler = nil
        append(pipe.fileHandleForReading.readDataToEndOfFile())
        let text = String(data: data, encoding: .utf8) ?? String(decoding: data, as: UTF8.self)
        return Capture(
            text: truncated ? text + "\n[output truncated]" : text,
            truncated: truncated
        )
    }

    private func append(_ chunk: Data) {
        guard !chunk.isEmpty else {
            return
        }
        lock.lock()
        defer {
            lock.unlock()
        }
        let remaining = max(0, limitBytes - data.count)
        if remaining > 0 {
            data.append(chunk.prefix(remaining))
        }
        if chunk.count > remaining {
            truncated = true
        }
    }
}
