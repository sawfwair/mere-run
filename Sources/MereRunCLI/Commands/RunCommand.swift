import ArgumentParser
import Foundation

struct Run: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Inspect durable mere.run workflow reports and run directories.",
        subcommands: [
            RunList.self,
            RunInspect.self,
        ]
    )
}

struct RunList: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "Find durable run directories, structured reports, and run plans under a root."
    )

    @Option(name: [.customLong("root")], help: "Root directory or file to scan.")
    var root: String

    @Option(name: [.customLong("max-depth")], help: "Maximum child-directory depth to scan from --root.")
    var maxDepth: Int = 4

    @Flag(name: [.customLong("json")], help: "Emit a structured JSON run-list report.")
    var json: Bool = false

    func run() throws {
        let envelope = try makeListEnvelope()
        if json {
            print(try StructuredRunOutput.encode(envelope))
        } else {
            print(envelope.summary)
            for entry in envelope.result.entries {
                let state = entry.state.map { " state=\($0)" } ?? ""
                let detail = entry.format.map { " format=\($0)" } ?? ""
                print("[\(entry.status.rawValue)] \(entry.relativePath) kind=\(entry.kind)\(state)\(detail)")
            }
            for diagnostic in envelope.diagnostics {
                stderr("[\(diagnostic.severity.rawValue)] \(diagnostic.title): \(diagnostic.message)")
            }
        }
        if envelope.status == .blocked {
            throw ExitCode.failure
        }
    }

    func makeListEnvelope(
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) throws -> RunListEnvelope {
        guard maxDepth >= 0 else {
            throw ValidationError("--max-depth must be >= 0")
        }
        return RunListAnalyzer(
            root: root,
            maxDepth: maxDepth,
            fileManager: fileManager,
            now: now
        ).envelope()
    }

    private func stderr(_ message: String) {
        FileHandle.standardError.write(Data("\(message)\n".utf8))
    }
}

struct RunInspect: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "inspect",
        abstract: "Inspect a run directory, structured report, or run plan."
    )

    @Argument(help: "Run directory, structured report JSON, or run plan JSON path.")
    var path: String

    @Flag(name: [.customLong("json")], help: "Emit a structured JSON inspection report.")
    var json: Bool = false

    func run() throws {
        let envelope = makeInspectionEnvelope()
        if json {
            print(try StructuredRunOutput.encode(envelope))
        } else {
            print(envelope.summary)
            if let runDirectory = envelope.result.runDirectory {
                print("Status: \(runDirectory.status)")
                if let manifest = runDirectory.manifest {
                    print("Format: \(manifest.format)")
                    print("Model: \(manifest.model)")
                    print("Progress: \(manifest.step)/\(manifest.totalSteps)")
                }
                print("Events: \(runDirectory.events.count)")
                print("Actions: \(runDirectory.actions.count)")
                print("Artifacts: \(runDirectory.artifacts.count)")
            } else if let report = envelope.result.report {
                print("Report: \(report.command.joined(separator: " ")) \(report.mode.rawValue) \(report.status.rawValue)")
                print("Diagnostics: \(report.diagnosticCount)")
                print("Actions: \(report.actionCount)")
            } else if let plan = envelope.result.plan {
                print("Plan: \(plan.kind)")
                print("Command: \(plan.command.joined(separator: " "))")
            }
            for diagnostic in envelope.diagnostics {
                stderr("[\(diagnostic.severity.rawValue)] \(diagnostic.title): \(diagnostic.message)")
            }
        }
        if envelope.status == .blocked {
            throw ExitCode.failure
        }
    }

    func makeInspectionEnvelope(
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) -> RunInspectionEnvelope {
        RunInspectionAnalyzer(
            path: path,
            fileManager: fileManager,
            now: now
        ).envelope()
    }

    private func stderr(_ message: String) {
        FileHandle.standardError.write(Data("\(message)\n".utf8))
    }
}
