import AppKit
import Foundation

/// Builds the support bundle text for **Export Diagnostics**.
///
/// The report is meant to be pasted into an issue, so it carries versions, the resolved
/// executable, machine shape, and recent run outcomes — and deliberately carries no
/// console text or command arguments. Those surfaces can contain prompts, generated
/// content, local paths, configuration values, API keys, or tokens.
enum StudioDiagnostics {
    struct LogSummary: Equatable {
        var systemCount = 0
        var stdoutCount = 0
        var stderrCount = 0

        var totalCount: Int { systemCount + stdoutCount + stderrCount }
    }

    /// How many recent runs to include. Enough to show a failure in context without
    /// turning the report into a log dump.
    static let recentRunLimit = 20

    static func report(
        appVersion: String,
        cliVersion: String?,
        resolvedCLI: String,
        serverStatus: StudioServerStatus?,
        libraryItems: [StudioLibraryItem],
        recentLog: LogSummary,
        generatedAt: Date = Date()
    ) -> String {
        var lines: [String] = []

        lines.append("mere.run Studio diagnostics")
        lines.append("Generated \(ISO8601DateFormatter().string(from: generatedAt))")
        lines.append("")

        lines.append("## Versions")
        lines.append("App: \(appVersion)")
        lines.append("CLI: \(cliVersion ?? "not resolved")")
        lines.append("Resolved executable: \(resolvedCLI.isEmpty ? "not resolved" : resolvedCLI)")
        if let cliVersion, !cliVersion.isEmpty, cliVersion != appVersion {
            lines.append(
                "Note: the app and its embedded CLI report different versions. "
                    + "Sparkle updates both as one bundle, so a mismatch usually means the CLI "
                    + "was resolved from PATH instead of the bundle."
            )
        }
        lines.append("")

        lines.append("## Machine")
        let info = ProcessInfo.processInfo
        lines.append("macOS: \(info.operatingSystemVersionString)")
        lines.append("Architecture: \(machineArchitecture())")
        lines.append("Cores: \(info.processorCount)")
        lines.append(
            "Unified memory: \(bytes(Int64(info.physicalMemory)))"
        )
        lines.append("")

        lines.append("## Local server")
        if let serverStatus {
            lines.append("Health: \(serverStatus.health)")
            lines.append("Reachable: \(serverStatus.isReachable ? "yes" : "no")")
            lines.append("Installed models: \(serverStatus.installedCount)")
            if serverStatus.loadedModels.isEmpty {
                lines.append("Loaded models: none")
            } else {
                lines.append("Loaded models: \(serverStatus.loadedModels.joined(separator: ", "))")
            }
        } else {
            lines.append("No status snapshot has been taken this session.")
        }
        lines.append("")

        lines.append("## Recent runs (most recent first)")
        let recent = libraryItems
            .sorted { $0.updatedAt > $1.updatedAt }
            .prefix(recentRunLimit)
        if recent.isEmpty {
            lines.append("No runs recorded.")
        } else {
            for item in recent {
                let exit = item.exitCode.map(String.init) ?? "—"
                lines.append(
                    "- \(ISO8601DateFormatter().string(from: item.updatedAt)) "
                        + "[\(item.status.rawValue)] exit \(exit) · \(item.mode.rawValue)"
                )
            }
        }
        lines.append("")

        lines.append("## Console summary")
        if recentLog.totalCount == 0 {
            lines.append("No console output captured this session.")
        } else {
            lines.append("Captured lines: \(recentLog.totalCount)")
            lines.append("System: \(recentLog.systemCount)")
            lines.append("Standard output: \(recentLog.stdoutCount)")
            lines.append("Standard error: \(recentLog.stderrCount)")
            lines.append("Console text omitted to protect prompts, outputs, paths, and credentials.")
        }
        lines.append("")

        lines.append(
            "This report contains no command arguments or console text."
        )

        return lines.joined(separator: "\n")
    }

    static func suggestedFilename(generatedAt: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return "mere-run-diagnostics-\(formatter.string(from: generatedAt)).txt"
    }

    private static func machineArchitecture() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let machine = withUnsafeBytes(of: &systemInfo.machine) { raw -> String in
            let bytes = raw.prefix(while: { $0 != 0 })
            return String(decoding: bytes, as: UTF8.self)
        }
        return machine.isEmpty ? "unknown" : machine
    }

    private static func bytes(_ count: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .memory
        return formatter.string(fromByteCount: count)
    }
}
