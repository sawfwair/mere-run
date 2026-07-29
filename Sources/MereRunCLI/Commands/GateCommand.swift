import ArgumentParser
import Crypto
import Foundation
import MereRunCore

#if canImport(CoreGraphics)
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers
#endif

/// End-to-end quality gate: runs the real user-facing commands against the
/// installed models, hashes their outputs, checks in-run determinism, and
/// compares against machine-local baselines. Exists because the platform
/// shipped a broken decode path (stale Metal library, garbage above 1024
/// tokens) for two months without anything noticing: every check here runs
/// the actual product surface, and the text checks deliberately include a
/// long-context lane in the historically blind regime.
struct Gate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "gate",
        abstract: "Run the end-to-end quality gate against installed models.",
        discussion: """
        Each check shells out to this executable's real subcommands (text
        chat, speech synthesize/transcribe, vision ocr, image generate, text
        embed, and video generate). Deterministic checks run twice and compare
        hashes. Video release smokes run one minimum valid native generation,
        then decode the MP4 and promised audio. Correctness failures exit
        nonzero; performance regressions warn unless --strict-perf is set.
        Checks whose models are not installed are skipped unless --require-all
        is set. `--all-installed` replaces the fixed regression set with one
        true inference check for every model reported as installed by
        `mere.run model list`. An installed catalog entry without an explicit
        direct or companion-consumption recipe fails closed.

        First run on a machine: `mere.run gate --update-baselines` records
        the baseline hashes and timings. Re-run after intentional
        output-changing merges to refresh them.
        """
    )

    @Option(
        name: [.customLong("suite")],
        help: """
        Comma-separated fixed suites: text, speech, vision, image, embed, video.
        With --all-installed, use managed model category names (default: all).
        """
    )
    var suite: String = "all"

    @Flag(
        name: [.customLong("require-all")],
        help: "Fail when a selected check's model is not installed instead of reporting a skip."
    )
    var requireAll: Bool = false

    @Flag(
        name: [.customLong("all-installed")],
        help: "Run a fail-closed true-inference smoke for every installed managed model."
    )
    var allInstalled: Bool = false

    @Option(
        name: [.customLong("skip-model")],
        help: """
        Comma-separated installed model IDs to record as explicit quarantined skips.
        Valid only with --all-installed; unknown or non-installed IDs fail closed.
        """
    )
    var skipModel: String = ""

    @Flag(name: [.customLong("update-baselines")], help: "Record current outputs and timings as the new baselines.")
    var updateBaselines: Bool = false

    @Flag(name: [.customLong("strict-perf")], help: "Treat performance regressions (vs baseline) as failures, not warnings.")
    var strictPerf: Bool = false

    @Option(name: [.customLong("json-output")], help: "Write the full gate report as JSON to this path.")
    var jsonOutput: String?

    @Flag(name: [.customLong("list")], help: "List available checks and exit.")
    var listOnly: Bool = false

    func run() async throws {
        let skippedModelIDs = Set(
            skipModel.split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        )
        if !skippedModelIDs.isEmpty && !allInstalled {
            throw ValidationError("--skip-model is valid only with --all-installed.")
        }

        let checks: [GateCheck]
        if allInstalled {
            let installedIDs = Set(
                ModelInventory.rows()
                    .filter(\.isInstalled)
                    .map(\.id)
            )
            guard !installedIDs.isEmpty else {
                throw ValidationError("--all-installed found no installed managed models.")
            }
            let invalidSkips = skippedModelIDs.subtracting(installedIDs)
            guard invalidSkips.isEmpty else {
                throw ValidationError(
                    "--skip-model IDs are not installed managed models: "
                        + invalidSkips.sorted().joined(separator: ", ")
                )
            }
            let installedSpecs = ManagedModelCatalog.allSpecs.filter { installedIDs.contains($0.id) }
            let unmapped = installedSpecs.filter {
                InstalledModelSmokePlans.plan(for: $0, installedIDs: installedIDs) == nil
            }
            guard unmapped.isEmpty else {
                throw ValidationError(
                    "Installed models have no release smoke recipe: "
                        + unmapped.map(\.id).sorted().joined(separator: ", ")
                )
            }
            checks = installedSpecs.compactMap {
                InstalledModelSmokePlans.plan(for: $0, installedIDs: installedIDs)?.check
            }
        } else {
            checks = GateChecks.all
        }
        if listOnly {
            for check in checks {
                print(
                    "\(check.id)  [\(check.suite)]  requires: "
                        + check.requiredModels.joined(separator: ", ")
                        + "  \(check.successDetail)"
                )
            }
            return
        }

        let selectedSuites = Set(
            suite.lowercased() == "all"
                ? checks.map(\.suite)
                : suite.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
        )

        let runner = GateRunner(
            executableURL: URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL,
            workDirectory: FileManager.default.temporaryDirectory
                .appendingPathComponent("mererun-gate-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
        )
        try FileManager.default.createDirectory(at: runner.workDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: runner.workDirectory) }

        var baselines = GateBaselineStore.load()
        var results: [GateResult] = []

        for check in checks where selectedSuites.contains(check.suite) {
            if allInstalled,
               check.id.hasPrefix("installed-"),
               skippedModelIDs.contains(String(check.id.dropFirst("installed-".count))) {
                results.append(GateResult(
                    id: check.id,
                    status: .skipped,
                    detail: "explicit release quarantine via --skip-model",
                    observation: nil
                ))
                continue
            }
            guard check.requiredModels.allSatisfy(GateRunner.modelInstalled) else {
                results.append(GateResult(
                    id: check.id,
                    status: requireAll ? .failed : .skipped,
                    detail: "\(requireAll ? "required model missing" : "missing model"): "
                        + check.requiredModels.joined(separator: ", "),
                    observation: nil
                ))
                continue
            }
            FileHandle.standardError.write(Data("[gate] running \(check.id)…\n".utf8))
            do {
                let observation = try await check.run(runner)
                let result = Self.judge(
                    check: check,
                    observation: observation,
                    baseline: baselines.entries[check.id],
                    strictPerf: strictPerf
                )
                results.append(result)
                if check.comparesBaseline,
                   updateBaselines || (baselines.entries[check.id] == nil && result.status != .failed) {
                    baselines.entries[check.id] = GateBaseline(
                        hash: observation.hash,
                        wallSeconds: observation.wallSeconds,
                        decodeTps: observation.decodeTps
                    )
                }
            } catch {
                results.append(GateResult(
                    id: check.id, status: .failed,
                    detail: "check threw: \(error.localizedDescription)",
                    observation: nil
                ))
            }
        }

        if updateBaselines || results.contains(where: { $0.status != .skipped }) {
            try GateBaselineStore.save(baselines)
        }

        let report = GateReport(
            createdAt: ISO8601DateFormatter().string(from: Date()),
            hostModel: GateRunner.hardwareModel(),
            results: results
        )
        Self.printTable(report)
        if let jsonOutput {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: URL(fileURLWithPath: jsonOutput))
        }

        if results.contains(where: { $0.status == .failed }) {
            throw ExitCode(1)
        }
    }

    private static func judge(
        check: GateCheck,
        observation: GateObservation,
        baseline: GateBaseline?,
        strictPerf: Bool
    ) -> GateResult {
        if let deterministicHash = observation.secondRunHash, deterministicHash != observation.hash {
            return GateResult(
                id: check.id, status: .failed,
                detail: "NONDETERMINISTIC: run1=\(observation.hash.prefix(12)) run2=\(deterministicHash.prefix(12))",
                observation: observation
            )
        }
        if let semantic = observation.semanticFailure {
            return GateResult(id: check.id, status: .failed, detail: semantic, observation: observation)
        }
        guard check.comparesBaseline else {
            return GateResult(
                id: check.id,
                status: .passed,
                detail: check.successDetail,
                observation: observation
            )
        }
        guard let baseline else {
            return GateResult(
                id: check.id, status: .passed,
                detail: "baseline recorded (\(observation.hash.prefix(12)))",
                observation: observation
            )
        }
        if observation.hash != baseline.hash {
            return GateResult(
                id: check.id, status: .failed,
                detail: "OUTPUT CHANGED vs baseline: \(baseline.hash.prefix(12)) -> \(observation.hash.prefix(12)) (re-run with --update-baselines if intentional)",
                observation: observation
            )
        }
        var perfNotes: [String] = []
        var perfFailed = false
        if let baseTps = baseline.decodeTps, let tps = observation.decodeTps, tps < baseTps * 0.75 {
            perfNotes.append(String(format: "decode_tps %.1f < 0.75x baseline %.1f", tps, baseTps))
            perfFailed = true
        }
        if observation.wallSeconds > baseline.wallSeconds * 1.5 {
            perfNotes.append(String(format: "wall %.1fs > 1.5x baseline %.1fs", observation.wallSeconds, baseline.wallSeconds))
            perfFailed = true
        }
        if perfFailed {
            let detail = "perf regression: " + perfNotes.joined(separator: "; ")
            return GateResult(
                id: check.id,
                status: strictPerf ? .failed : .warned,
                detail: detail,
                observation: observation
            )
        }
        return GateResult(id: check.id, status: .passed, detail: "matches baseline", observation: observation)
    }

    private static func printTable(_ report: GateReport) {
        print("\nGATE REPORT (\(report.hostModel))")
        for result in report.results {
            let mark: String
            switch result.status {
            case .passed: mark = "PASS"
            case .warned: mark = "WARN"
            case .failed: mark = "FAIL"
            case .skipped: mark = "SKIP"
            }
            var line = "  [\(mark)] \(result.id) — \(result.detail)"
            if let observation = result.observation {
                line += String(format: " (%.1fs", observation.wallSeconds)
                if let tps = observation.decodeTps {
                    line += String(format: ", %.1f tok/s", tps)
                }
                line += ")"
            }
            print(line)
        }
        let failed = report.results.filter { $0.status == .failed }.count
        let warned = report.results.filter { $0.status == .warned }.count
        let passed = report.results.filter { $0.status == .passed }.count
        let skipped = report.results.filter { $0.status == .skipped }.count
        print("  \(passed) passed, \(warned) warned, \(failed) failed, \(skipped) skipped\n")
    }
}
