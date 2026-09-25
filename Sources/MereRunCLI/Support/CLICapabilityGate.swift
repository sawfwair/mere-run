import Foundation
import MereRunContract
import MereRunCore

/// Checks a command line against the capability contract before machine admission, model
/// resolution, download, or load. A model a command can't run, or an option its runtime family
/// rejects, fails here with one message; an option the family ignores prints a warning once the
/// command has parsed and validated, and the run continues. Commands without routing pass
/// untouched.
enum CLICapabilityGate {
    struct Rejection: LocalizedError, Equatable {
        let messages: [String]

        var errorDescription: String? {
            messages.joined(separator: "\n")
        }
    }

    /// `arguments` is the process argv, executable first. Throws when the gate refuses the run;
    /// otherwise returns the warning lines to print once the command has parsed and validated:
    /// none under `--quiet`, which keeps "has no effect" notes off a quiet run as the commands'
    /// own notes do.
    @discardableResult
    static func check(arguments: [String]) throws -> [String] {
        let commandLine = Array(arguments.dropFirst())
        guard !requestsBuiltIn(commandLine),
              let (capability, invocation) = invocation(commandLine: commandLine),
              capability.routing != nil else {
            return []
        }
        let report = report(capability, invocation)
        guard report.violations.isEmpty else {
            throw Rejection(messages: report.violations)
        }
        return invocation.contains("--quiet") ? [] : report.warnings.map { "Warning: \($0)\n" }
    }

    /// The contract's decision for a command line without the executable; `catalog resolve`
    /// prints it. `nil` when the line names no cataloged command.
    static func evaluate(
        commandLine: [String]
    ) -> (capability: MereRunCommandCapability, report: MereRunFamilyResolutionReport)? {
        invocation(commandLine: commandLine).map { capability, invocation in
            (capability, report(capability, invocation))
        }
    }

    /// True when ArgumentParser answers the command line itself (help, the help dump, the
    /// version, a completion script) instead of running a command, so there is nothing to check.
    static func requestsBuiltIn(_ commandLine: [String]) -> Bool {
        if commandLine.first == "---completion" { return true }
        return commandLine.prefix { $0 != "--" }.contains { token in
            let name = token.split(separator: "=", maxSplits: 1).first.map(String.init) ?? token
            if builtInFlags.contains(name) { return true }
            // A single-dash group asks for help when one of its letters is `h` (`-qh`).
            let group = token.dropFirst()
            return token.hasPrefix("-") && !token.hasPrefix("--") && group.count > 1
                && group.allSatisfy { $0.isLetter || $0.isNumber } && group.contains("h")
        }
    }

    /// ArgumentParser's own flags: help at both visibilities, in each spelling it accepts.
    private static let builtInFlags: Set<String> = [
        "-h", "--help", "-help", "--help-hidden", "-help-hidden",
        "--experimental-dump-help", "--version", "--generate-completion-script"
    ]

    static var platform: String {
        #if os(Linux)
        "linux"
        #else
        "macos"
        #endif
    }

    /// Commands that run a cataloged capability under another name, with the same options; the
    /// gate reads them as the capability.
    static let aliasCommands: [[String]: [String]] = [
        ["vision", "image-to-3d"]: ["image", "reconstruct-3d"],
        ["vision", "image-to-3d-trellis2"]: ["image", "reconstruct-3d-trellis2"],
        ["vision", "image-to-3d-multiview"]: ["image", "reconstruct-3d-multiview"]
    ]

    private static func invocation(
        commandLine: [String]
    ) -> (capability: MereRunCommandCapability, invocation: MereRunCommandInvocation)? {
        var commandLine = withoutRootOptions(commandLine)
        if let (alias, command) = aliasCommands.first(where: { commandLine.starts(with: $0.key) }) {
            commandLine = command + commandLine.dropFirst(alias.count)
        }
        return MereRunCapabilityCatalog.capability(forCommandLine: commandLine).map { capability, arguments in
            (capability, MereRunCommandInvocation(capability: capability, arguments: arguments))
        }
    }

    private static func report(
        _ capability: MereRunCommandCapability,
        _ invocation: MereRunCommandInvocation
    ) -> MereRunFamilyResolutionReport {
        capability.resolutionReport(
            invocation,
            platform: platform,
            identify: { ModelFamilyIdentifier.identify(capabilityID: capability.id, model: $0, invocation: invocation) },
            chooseDefault: { ModelFamilyIdentifier.machineDefault(capabilityID: capability.id, candidates: $0) },
            routedFamily: { CLIFamilyRouters.family(capabilityID: capability.id, invocation: invocation) }
        )
    }

    /// Drops the root command's `--models-root`, which ArgumentParser accepts anywhere before
    /// `--`, so the command path and the leaf's arguments read as the leaf sees them.
    private static func withoutRootOptions(_ commandLine: [String]) -> [String] {
        var result: [String] = []
        var index = commandLine.startIndex
        while index < commandLine.endIndex {
            let token = commandLine[index]
            if token == "--" {
                result.append(contentsOf: commandLine[index...])
                break
            }
            if token == "--models-root" {
                index += 2
                continue
            }
            if !token.hasPrefix("--models-root=") {
                result.append(token)
            }
            index += 1
        }
        return result
    }
}
