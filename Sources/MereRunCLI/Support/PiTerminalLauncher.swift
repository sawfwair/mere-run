import Foundation

enum PiTerminalLauncher {
    static var launchesDetached: Bool {
        #if os(macOS)
        return true
        #else
        return false
        #endif
    }

    static var launchProgressMessage: String {
        #if os(macOS)
        return "Opening Pi in Terminal.app."
        #else
        return "Starting Pi in this terminal."
        #endif
    }

    static var runningProgressMessage: String {
        #if os(macOS)
        return "Pi is running in Terminal.app. Keep this command running; press Ctrl+C here to stop the API server."
        #else
        return "Pi exited. Stopping the API server."
        #endif
    }

    static func launch(
        piURL: URL,
        modelID: String,
        prompt: String,
        homeDirectory: URL,
        workingDirectory: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    ) throws {
        try FileManager.default.createDirectory(at: homeDirectory, withIntermediateDirectories: true)

        #if os(macOS)
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
        #else
        let process = Process()
        process.executableURL = piURL
        process.arguments = [
            "--provider",
            "mere-run",
            "--model",
            modelID,
            prompt,
        ]
        process.currentDirectoryURL = workingDirectory
        var environment = ProcessInfo.processInfo.environment
        environment["HOME"] = homeDirectory.path
        environment["XDG_CONFIG_HOME"] = homeDirectory
            .appendingPathComponent(".config", isDirectory: true)
            .path
        environment["XDG_DATA_HOME"] = homeDirectory
            .appendingPathComponent(".local/share", isDirectory: true)
            .path
        process.environment = environment
        process.standardInput = FileHandle.standardInput
        process.standardOutput = FileHandle.standardOutput
        process.standardError = FileHandle.standardError
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw PiAgentIntegration.IntegrationError.terminalLaunchFailed
        }
        #endif
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
