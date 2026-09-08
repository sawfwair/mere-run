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

    static func resolvedBackend(mode: CodeExecutionSandboxMode) throws -> CodeExecutionSandboxBackend {
        try resolveBackend(mode: mode)
    }

    static func resolvedPythonExecutable(_ python: String) throws -> URL {
        if python.contains("/") {
            let url = URL(fileURLWithPath: python).standardizedFileURL
            guard executableExists(at: url.path) else {
                throw ValidationError("Python executable not found or not executable: \(python).")
            }
            return url.resolvingSymlinksInPath()
        }
        guard let path = findExecutable(named: python) else {
            throw ValidationError("Python executable not found on PATH: \(python).")
        }
        return URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
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

        let result = try BoundedProcessRunner.run(
            process, timeout: timeout, outputLimitBytes: outputLimitBytes
        )
        if result.completion == .cancelled { throw CancellationError() }

        return CodeExecutionSandboxResult(
            backend: backend,
            passed: result.completion == .exited && result.status == 0,
            seconds: result.seconds,
            timedOut: result.completion == .timedOut,
            status: result.status,
            stdout: result.stdout.text,
            stderr: result.stderr.text,
            stdoutTruncated: result.stdout.truncated,
            stderrTruncated: result.stderr.truncated
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

}
