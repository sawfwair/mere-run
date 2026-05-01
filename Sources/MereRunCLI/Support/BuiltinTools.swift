import Foundation
import MereRunCore

enum BuiltinTools {
    struct ToolExecutionPolicy {
        let sandboxDir: URL
        let allowShellExec: Bool
        let allowAbsolutePaths: Bool
    }

    static let writeFile = ToolDefinition(
        name: "write_file",
        description: "Write content to a file at the given path. Creates parent directories if needed.",
        parameters: [
            "path": ToolParameterProperty(type: "string", description: "The file path to write to (relative to sandbox directory)"),
            "content": ToolParameterProperty(type: "string", description: "The content to write to the file"),
        ],
        required: ["path", "content"]
    )

    static let shellExec = ToolDefinition(
        name: "shell_exec",
        description: "Execute a shell command and return its combined stdout and stderr output.",
        parameters: [
            "command": ToolParameterProperty(type: "string", description: "The shell command to execute"),
        ],
        required: ["command"]
    )

    static let all: [String: ToolDefinition] = [
        "write_file": writeFile,
        "shell_exec": shellExec,
    ]

    static func resolve(names: [String]) throws -> [ToolDefinition] {
        try names.map { name in
            guard let tool = all[name] else {
                throw ToolError.unknownTool(name, available: Array(all.keys).sorted())
            }
            return tool
        }
    }

    static func execute(_ call: ToolCall, policy: ToolExecutionPolicy) throws -> String {
        switch call.name {
        case "write_file":
            return try executeWriteFile(call, policy: policy)
        case "shell_exec":
            return try executeShellExec(call, policy: policy)
        default:
            return "Error: unknown tool '\(call.name)'"
        }
    }

    static func canAutoApprove(_ call: ToolCall, autoApproveTools: Bool) -> Bool {
        autoApproveTools && call.name != shellExec.name
    }

    // MARK: - Tool Implementations

    private static func executeWriteFile(_ call: ToolCall, policy: ToolExecutionPolicy) throws -> String {
        guard let path = call.arguments["path"], !path.isEmpty else {
            return "Error: 'path' argument is required"
        }
        guard let content = call.arguments["content"] else {
            return "Error: 'content' argument is required"
        }

        var fileURL = try resolveWriteTarget(path: path, policy: policy)

        // Ensure parent directory exists
        let parentDir = fileURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)
        fileURL = try resolveSymlinkSafeWriteTarget(fileURL, originalPath: path, policy: policy)

        try content.write(to: fileURL, atomically: true, encoding: .utf8)
        return "Wrote \(content.count) bytes to \(fileURL.path)"
    }

    private static func executeShellExec(_ call: ToolCall, policy: ToolExecutionPolicy) throws -> String {
        guard let command = call.arguments["command"], !command.isEmpty else {
            return "Error: 'command' argument is required"
        }
        guard policy.allowShellExec else {
            return "Denied: 'shell_exec' requires --allow-shell-exec."
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-c", command]
        process.currentDirectoryURL = policy.sandboxDir

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        try process.run()
        process.waitUntilExit()

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        let status = process.terminationStatus

        if status == 0 {
            return output.isEmpty ? "(no output)" : output
        } else {
            return "Exit code \(status)\n\(output)"
        }
    }

    private static func resolveWriteTarget(path: String, policy: ToolExecutionPolicy) throws -> URL {
        let sandboxRoot = canonicalSandboxRoot(policy.sandboxDir)

        if path.hasPrefix("/") {
            guard policy.allowAbsolutePaths else {
                throw ToolPathError.absolutePathsDisabled
            }
            return URL(fileURLWithPath: path).standardizedFileURL
        }

        let candidate = sandboxRoot.appendingPathComponent(path).standardizedFileURL
        guard isWithinSandbox(candidate, sandboxRoot: sandboxRoot) else {
            throw ToolPathError.escapesSandbox(path: path, sandbox: sandboxRoot.path)
        }
        return candidate
    }

    private static func resolveSymlinkSafeWriteTarget(
        _ fileURL: URL,
        originalPath: String,
        policy: ToolExecutionPolicy
    ) throws -> URL {
        guard !originalPath.hasPrefix("/") else {
            return fileURL
        }

        let sandboxRoot = canonicalSandboxRoot(policy.sandboxDir)
        let resolvedParent = fileURL
            .deletingLastPathComponent()
            .resolvingSymlinksInPath()
            .standardizedFileURL
        guard isWithinSandbox(resolvedParent, sandboxRoot: sandboxRoot) else {
            throw ToolPathError.escapesSandbox(path: originalPath, sandbox: sandboxRoot.path)
        }

        if FileManager.default.fileExists(atPath: fileURL.path) {
            let resolvedFile = fileURL.resolvingSymlinksInPath().standardizedFileURL
            guard isWithinSandbox(resolvedFile, sandboxRoot: sandboxRoot) else {
                throw ToolPathError.escapesSandbox(path: originalPath, sandbox: sandboxRoot.path)
            }
        }

        return resolvedParent.appendingPathComponent(fileURL.lastPathComponent, isDirectory: false)
    }

    private static func canonicalSandboxRoot(_ sandboxDir: URL) -> URL {
        sandboxDir.standardizedFileURL.resolvingSymlinksInPath()
    }

    private static func isWithinSandbox(_ candidate: URL, sandboxRoot: URL) -> Bool {
        let sandboxPath = sandboxRoot.path
        let candidatePath = candidate.path
        return candidatePath == sandboxPath || candidatePath.hasPrefix(sandboxPath + "/")
    }

    enum ToolError: LocalizedError {
        case unknownTool(String, available: [String])

        var errorDescription: String? {
            switch self {
            case .unknownTool(let name, let available):
                return "Unknown tool '\(name)'. Available: \(available.joined(separator: ", "))"
            }
        }
    }

    enum ToolPathError: LocalizedError {
        case absolutePathsDisabled
        case escapesSandbox(path: String, sandbox: String)

        var errorDescription: String? {
            switch self {
            case .absolutePathsDisabled:
                return "Absolute paths are disabled for tool writes. Use a relative path inside the sandbox."
            case .escapesSandbox(let path, let sandbox):
                return "Path '\(path)' escapes sandbox '\(sandbox)'."
            }
        }
    }
}
