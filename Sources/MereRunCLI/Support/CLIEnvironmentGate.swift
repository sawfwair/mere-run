import Foundation
import MereRunContract

/// Audits `MERERUN_*` environment variables at CLI startup and prepares non-fatal stderr
/// warnings for unknown variable names (with typo suggestions) and syntax-invalid values.
enum CLIEnvironmentGate {
    /// Runs `MereRunEnvironmentRegistry.audit(environment:)`.
    static func audit(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> MereRunEnvironmentAudit {
        MereRunEnvironmentRegistry.audit(environment: environment)
    }

    /// Returns stderr warning lines for `arguments` (process argv, executable first).
    /// Skips built-in help/version/completion invocations and `status`, which reports the
    /// same warnings in its own output, and suppresses output under `--quiet`.
    static func stderrLines(
        arguments: [String],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> [String] {
        let commandLine = Array(arguments.dropFirst())
        guard commandLine.first != "status",
              !CLICapabilityGate.requestsBuiltIn(commandLine),
              !isQuiet(commandLine) else {
            return []
        }
        let report = audit(environment: environment)
        guard report.hasIssues else {
            return []
        }
        return report.warnings.map { "Warning: \($0)\n" }
    }

    private static func isQuiet(_ commandLine: [String]) -> Bool {
        commandLine.prefix { $0 != "--" }.contains { token in
            if token == "--quiet" || token == "-q" {
                return true
            }
            let group = token.dropFirst()
            return token.hasPrefix("-") && !token.hasPrefix("--") && group.count > 1
                && group.allSatisfy { $0.isLetter || $0.isNumber } && group.contains("q")
        }
    }
}
