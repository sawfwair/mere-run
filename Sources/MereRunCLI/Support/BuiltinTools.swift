import Foundation
import MereRunCore

enum BuiltinTools {
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

    static func execute(_ call: ToolCall, sandboxDir: URL) throws -> String {
        switch call.name {
        case "write_file":
            return try executeWriteFile(call, sandboxDir: sandboxDir)
        case "shell_exec":
            return try executeShellExec(call, sandboxDir: sandboxDir)
        default:
            return "Error: unknown tool '\(call.name)'"
        }
    }

    // MARK: - Tool Implementations

    private static func executeWriteFile(_ call: ToolCall, sandboxDir: URL) throws -> String {
        guard let path = call.arguments["path"], !path.isEmpty else {
            return "Error: 'path' argument is required"
        }
        guard let content = call.arguments["content"] else {
            return "Error: 'content' argument is required"
        }

        let fileURL: URL
        if path.hasPrefix("/") {
            fileURL = URL(fileURLWithPath: path).standardizedFileURL
        } else {
            fileURL = sandboxDir.appendingPathComponent(path).standardizedFileURL
        }

        // Ensure parent directory exists
        let parentDir = fileURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

        try content.write(to: fileURL, atomically: true, encoding: .utf8)
        return "Wrote \(content.count) bytes to \(fileURL.path)"
    }

    private static func executeShellExec(_ call: ToolCall, sandboxDir: URL) throws -> String {
        guard let command = call.arguments["command"], !command.isEmpty else {
            return "Error: 'command' argument is required"
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-c", command]
        process.currentDirectoryURL = sandboxDir

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

    enum ToolError: LocalizedError {
        case unknownTool(String, available: [String])

        var errorDescription: String? {
            switch self {
            case .unknownTool(let name, let available):
                return "Unknown tool '\(name)'. Available: \(available.joined(separator: ", "))"
            }
        }
    }
}
