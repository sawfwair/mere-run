import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

enum PiTerminalLauncher {
    static var launchesDetached: Bool {
        #if os(macOS)
        return true
        #else
        return false
        #endif
    }

    static func launchesDetached(inline: Bool) -> Bool {
        inline ? false : launchesDetached
    }

    static var launchProgressMessage: String {
        #if os(macOS)
        return "Opening Pi in Terminal.app."
        #else
        return "Starting Pi in this terminal."
        #endif
    }

    static func launchProgressMessage(inline: Bool) -> String {
        inline ? "Starting Pi in this terminal." : launchProgressMessage
    }

    static var runningProgressMessage: String {
        #if os(macOS)
        return "Pi is running in Terminal.app. Keep this command running; press Ctrl+C here to stop the API server."
        #else
        return "Pi exited. Stopping the API server."
        #endif
    }

    static func runningProgressMessage(inline: Bool) -> String {
        inline ? "Pi exited. Stopping the API server." : runningProgressMessage
    }

    static func launch(
        piURL: URL,
        modelID: String,
        prompt: String,
        homeDirectory: URL,
        workingDirectory: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
        additionalArguments: [String] = [],
        inline: Bool = false
    ) throws {
        try FileManager.default.createDirectory(at: homeDirectory, withIntermediateDirectories: true)

        if inline {
            try launchInline(
                piURL: piURL,
                modelID: modelID,
                prompt: prompt,
                homeDirectory: homeDirectory,
                workingDirectory: workingDirectory,
                additionalArguments: additionalArguments
            )
            return
        }

        #if os(macOS)
        let arguments = arguments(
            modelID: modelID,
            prompt: prompt,
            homeDirectory: homeDirectory,
            additionalArguments: additionalArguments
        )
        let command = [
            "mkdir -p \(shellQuote(homeDirectory.path))",
            "export HOME=\(shellQuote(homeDirectory.path))",
            "export XDG_CONFIG_HOME=\(shellQuote(homeDirectory.appendingPathComponent(".config", isDirectory: true).path))",
            "export XDG_DATA_HOME=\(shellQuote(homeDirectory.appendingPathComponent(".local/share", isDirectory: true).path))",
            "cd \(shellQuote(workingDirectory.path))",
            ([piURL.path] + arguments).map(shellQuote).joined(separator: " "),
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
        try launchInline(
            piURL: piURL,
            modelID: modelID,
            prompt: prompt,
            homeDirectory: homeDirectory,
            workingDirectory: workingDirectory,
            additionalArguments: additionalArguments
        )
        #endif
    }

    private static func launchInline(
        piURL: URL,
        modelID: String,
        prompt: String,
        homeDirectory: URL,
        workingDirectory: URL,
        additionalArguments: [String]
    ) throws {
        let process = Process()
        process.executableURL = piURL
        process.arguments = arguments(
            modelID: modelID,
            prompt: prompt,
            homeDirectory: homeDirectory,
            additionalArguments: additionalArguments
        )
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
        try waitForInteractiveProcess(process)
        guard process.terminationStatus == 0 else {
            throw PiAgentIntegration.IntegrationError.terminalLaunchFailed
        }
    }

    private static func waitForInteractiveProcess(_ process: Process) throws {
        #if os(macOS) || os(Linux)
        let terminal = FileHandle.standardInput.fileDescriptor
        guard isatty(terminal) == 1 else {
            process.waitUntilExit()
            return
        }
        let originalGroup = tcgetpgrp(terminal)
        let childGroup = getpgid(process.processIdentifier)
        guard originalGroup > 0, childGroup > 0 else {
            process.terminate()
            throw PiAgentIntegration.IntegrationError.terminalLaunchFailed
        }

        let previousSIGTTOUHandler = signal(SIGTTOU, SIG_IGN)
        guard tcsetpgrp(terminal, childGroup) == 0 else {
            _ = signal(SIGTTOU, previousSIGTTOUHandler)
            process.terminate()
            throw PiAgentIntegration.IntegrationError.terminalLaunchFailed
        }
        _ = kill(-childGroup, SIGCONT)
        process.waitUntilExit()
        _ = tcsetpgrp(terminal, originalGroup)
        _ = signal(SIGTTOU, previousSIGTTOUHandler)
        #else
        process.waitUntilExit()
        #endif
    }

    static func arguments(
        modelID: String,
        prompt: String,
        homeDirectory: URL,
        additionalArguments: [String]
    ) -> [String] {
        let providerExtension = homeDirectory
            .appendingPathComponent(".pi/agent/extensions", isDirectory: true)
            .appendingPathComponent("mere-run-local-provider.ts", isDirectory: false)
        return [
            "--provider", "mere-run",
            "--model", modelID,
            "--extension", providerExtension.path,
        ] + additionalArguments + [prompt]
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
