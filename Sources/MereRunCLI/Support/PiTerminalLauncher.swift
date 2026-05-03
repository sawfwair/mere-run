import Foundation

enum PiTerminalLauncher {
    static func launch(
        piURL: URL,
        modelID: String,
        prompt: String,
        homeDirectory: URL,
        workingDirectory: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    ) throws {
        try FileManager.default.createDirectory(at: homeDirectory, withIntermediateDirectories: true)

        let command = [
            "mkdir -p \(shellQuote(homeDirectory.path))",
            "export HOME=\(shellQuote(homeDirectory.path))",
            "export XDG_CONFIG_HOME=\(shellQuote(homeDirectory.appendingPathComponent(".config", isDirectory: true).path))",
            "export XDG_DATA_HOME=\(shellQuote(homeDirectory.appendingPathComponent(".local/share", isDirectory: true).path))",
            "cd \(shellQuote(workingDirectory.path))",
            [
                shellQuote(piURL.path),
                "--provider",
                "mere-run",
                "--model",
                shellQuote(modelID),
                shellQuote(prompt),
            ].joined(separator: " "),
            "printf '\\nPi exited. You can close this window.\\n'",
        ].joined(separator: "; ")

        let script = """
        tell application "Terminal"
          activate
          do script "\(appleScriptString(command))"
        end tell
        """

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-e", script]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw PiAgentIntegration.IntegrationError.terminalLaunchFailed
        }
    }

    private static func shellQuote(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
    }

    private static func appleScriptString(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }
}
