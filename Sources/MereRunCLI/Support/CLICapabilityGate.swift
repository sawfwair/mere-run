import Foundation
import MereRunContract
import MereRunCore

/// Checks a command line against the capability contract before machine admission, model
/// resolution, download, or load. A model a command can't run, or an option its runtime family
/// rejects, fails here with one message; an option the family ignores prints a warning and the
/// run continues. Commands without routing pass untouched.
enum CLICapabilityGate {
    struct Rejection: LocalizedError, Equatable {
        let messages: [String]

        var errorDescription: String? {
            messages.joined(separator: "\n")
        }
    }

    /// `arguments` is the process argv, executable first.
    static func check(arguments: [String]) throws {
        let commandLine = Array(arguments.dropFirst())
        let beforeTerminator = commandLine.prefix { $0 != "--" }
        guard !beforeTerminator.contains("--help"), !beforeTerminator.contains("-h"),
              let (capability, report) = evaluate(commandLine: commandLine),
              capability.routing != nil else {
            return
        }
        CLIInvocationContext.record(report)
        for warning in report.warnings {
            CLIStderr.write("Warning: \(warning)\n")
        }
        guard report.violations.isEmpty else {
            throw Rejection(messages: report.violations)
        }
    }

    /// The contract's decision for a command line without the executable; `catalog resolve`
    /// prints it. `nil` when the line names no cataloged command.
    static func evaluate(
        commandLine: [String]
    ) -> (capability: MereRunCommandCapability, report: MereRunFamilyResolutionReport)? {
        guard let (capability, arguments) = MereRunCapabilityCatalog.capability(
            forCommandLine: withoutRootOptions(commandLine)
        ) else {
            return nil
        }
        let invocation = MereRunCommandInvocation(capability: capability, arguments: arguments)
        let report = capability.resolutionReport(
            invocation,
            platform: platform,
            identify: { ModelFamilyIdentifier.identify(capabilityID: capability.id, model: $0, invocation: invocation) },
            chooseDefault: { ModelFamilyIdentifier.machineDefault(capabilityID: capability.id, candidates: $0) }
        )
        return (capability, report)
    }

    static var platform: String {
        #if os(Linux)
        "linux"
        #else
        "macos"
        #endif
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

/// What the capability gate resolved for this process's command line. Commands read the
/// runtime family here instead of re-deriving it.
enum CLIInvocationContext {
    private final class Storage: @unchecked Sendable {
        let lock = NSLock()
        var report: MereRunFamilyResolutionReport?
    }

    private static let storage = Storage()

    static var report: MereRunFamilyResolutionReport? {
        storage.lock.lock()
        defer { storage.lock.unlock() }
        return storage.report
    }

    /// The resolved family id, or `nil` when the gate could not identify one.
    static var family: String? {
        report?.family
    }

    static func record(_ report: MereRunFamilyResolutionReport) {
        storage.lock.lock()
        defer { storage.lock.unlock() }
        storage.report = report
    }
}
