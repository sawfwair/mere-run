import Foundation
import MereRunCore
import MereRunEvaluation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

struct EvaluationScore {
    let passed: Bool
    let score: Double
    let metrics: [EvaluationMetric]
    let failedChecks: [String]
    let hardFailures: [String]
}

enum EvaluationScoring {
    static func score(
        response: String,
        benchmarkCase: LoadedEvaluationCase,
        pack: LoadedEvaluationPack,
        arm: EvaluationArmPlan,
        profileID: String,
        trial: Int,
        modelID: String,
        adapterSHA256: String?,
        allowExternalScorer: Bool
    ) throws -> EvaluationScore {
        let assertionScore = try scoreAssertions(
            response: response,
            assertions: benchmarkCase.specification.assertions
        )
        switch pack.manifest.scorer.kind {
        case .assertions:
            guard let assertionScore else {
                throw EvaluationScoringError.invalidResponse(
                    "assertions scorer has no assertions for case \(benchmarkCase.specification.id)"
                )
            }
            return assertionScore
        case .externalProcess:
            guard allowExternalScorer else {
                throw EvaluationScoringError.authorizationRequired
            }
            let external = try ExternalEvaluationScorer.run(
                response: response,
                benchmarkCase: benchmarkCase,
                pack: pack,
                arm: arm,
                profileID: profileID,
                trial: trial,
                modelID: modelID,
                adapterSHA256: adapterSHA256
            )
            guard let assertionScore else { return external }
            return EvaluationScore(
                passed: assertionScore.passed && external.passed,
                score: min(assertionScore.score, external.score),
                metrics: assertionScore.metrics + external.metrics,
                failedChecks: assertionScore.failedChecks + external.failedChecks,
                hardFailures: external.hardFailures
            )
        }
    }

    static func scoreAssertions(
        response: String,
        assertions: [EvaluationAssertion]
    ) throws -> EvaluationScore? {
        guard !assertions.isEmpty else { return nil }
        var failed: [String] = []
        for assertion in assertions {
            let passed = try assertionPasses(assertion, response: response)
            if !passed {
                failed.append(assertion.id)
            }
        }
        let passedCount = assertions.count - failed.count
        let score = Double(passedCount) / Double(assertions.count)
        return EvaluationScore(
            passed: failed.isEmpty,
            score: score,
            metrics: [EvaluationMetric(id: "assertion-pass-rate", value: score)],
            failedChecks: failed,
            hardFailures: []
        )
    }

    private static func assertionPasses(
        _ assertion: EvaluationAssertion,
        response: String
    ) throws -> Bool {
        switch assertion.kind {
        case .contains:
            return response.range(
                of: assertion.value ?? "",
                options: assertion.caseInsensitive ? .caseInsensitive : []
            ) != nil
        case .excludes:
            return response.range(
                of: assertion.value ?? "",
                options: assertion.caseInsensitive ? .caseInsensitive : []
            ) == nil
        case .regex, .notRegex:
            let options: NSRegularExpression.Options = assertion.caseInsensitive ? .caseInsensitive : []
            let expression = try NSRegularExpression(
                pattern: assertion.value ?? "",
                options: options
            )
            let range = NSRange(response.startIndex..<response.endIndex, in: response)
            let matched = expression.firstMatch(in: response, range: range) != nil
            return assertion.kind == .regex ? matched : !matched
        case .validJSONObject:
            guard let data = response.data(using: .utf8) else { return false }
            return (try? JSONDecoder().decode(
                [String: OpenAIJSONValue].self,
                from: data
            )) != nil
        }
    }
}

enum ExternalEvaluationScorer {
    private static let maximumOutputBytes = 1_048_576

    static func run(
        response: String,
        benchmarkCase: LoadedEvaluationCase,
        pack: LoadedEvaluationPack,
        arm: EvaluationArmPlan,
        profileID: String,
        trial: Int,
        modelID: String,
        adapterSHA256: String?
    ) throws -> EvaluationScore {
        guard let executableURL = pack.scorerExecutableURL else {
            throw EvaluationScoringError.invalidResponse("external scorer executable is unavailable")
        }
        let responseSHA256 = FusedBenchmarkHash.sha256(response)
        let request = EvaluationScorerRequest(
            packID: pack.manifest.id,
            packVersion: pack.manifest.version,
            packSHA256: pack.packSHA256,
            caseID: benchmarkCase.specification.id,
            caseSHA256: benchmarkCase.contentSHA256,
            armID: arm.id,
            profileID: profileID,
            trial: trial,
            modelID: modelID,
            adapterSHA256: adapterSHA256,
            response: response,
            responseSHA256: responseSHA256
        )
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-eval-scorer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: false
        )
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: directory.path
        )
        defer { try? FileManager.default.removeItem(at: directory) }
        let stdoutURL = directory.appendingPathComponent("stdout.json")
        let stderrURL = directory.appendingPathComponent("stderr.txt")
        guard FileManager.default.createFile(atPath: stdoutURL.path, contents: nil),
              FileManager.default.createFile(atPath: stderrURL.path, contents: nil) else {
            throw EvaluationScoringError.invalidResponse("cannot create scorer output files")
        }
        for outputURL in [stdoutURL, stderrURL] {
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600],
                ofItemAtPath: outputURL.path
            )
        }
        let stdoutHandle = try FileHandle(forWritingTo: stdoutURL)
        let stderrHandle = try FileHandle(forWritingTo: stderrURL)
        let stdinPipe = Pipe()
        let process = Process()
        process.executableURL = executableURL
        process.arguments = pack.manifest.scorer.arguments
        process.currentDirectoryURL = pack.rootURL
        process.environment = Self.scorerEnvironment()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutHandle
        process.standardError = stderrHandle
        let completion = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in completion.signal() }
        do {
            try process.run()
            var requestData = try EvaluationJSON.canonicalEncoder.encode(request)
            requestData.append(0x0A)
            try stdinPipe.fileHandleForWriting.write(contentsOf: requestData)
            try stdinPipe.fileHandleForWriting.close()
            let timeout = DispatchTime.now() + pack.manifest.scorer.timeoutSeconds
            if completion.wait(timeout: timeout) == .timedOut {
                process.terminate()
                if completion.wait(timeout: .now() + 2) == .timedOut,
                   process.isRunning {
                    _ = kill(process.processIdentifier, SIGKILL)
                    _ = completion.wait(timeout: .now() + 2)
                }
                throw EvaluationScoringError.timeout(pack.manifest.scorer.timeoutSeconds)
            }
            try stdoutHandle.close()
            try stderrHandle.close()
        } catch {
            try? stdinPipe.fileHandleForWriting.close()
            try? stdoutHandle.close()
            try? stderrHandle.close()
            if process.isRunning {
                process.terminate()
            }
            throw error
        }

        let outputData = try Data(contentsOf: stdoutURL)
        let errorData = try Data(contentsOf: stderrURL)
        guard outputData.count <= maximumOutputBytes,
              errorData.count <= maximumOutputBytes else {
            throw EvaluationScoringError.invalidResponse("scorer output exceeded 1 MiB")
        }
        guard process.terminationStatus == 0 else {
            let diagnostics = String(decoding: errorData, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            throw EvaluationScoringError.failed(
                status: process.terminationStatus,
                diagnostics: diagnostics
            )
        }
        let decoded: EvaluationScorerResponse
        do {
            decoded = try JSONDecoder().decode(EvaluationScorerResponse.self, from: outputData)
        } catch {
            throw EvaluationScoringError.invalidResponse("cannot decode scorer JSON: \(error)")
        }
        try validate(decoded)
        return EvaluationScore(
            passed: decoded.passed,
            score: decoded.score,
            metrics: decoded.metrics,
            failedChecks: [],
            hardFailures: decoded.hardFailures
        )
    }

    private static func scorerEnvironment() -> [String: String] {
        let environment = ProcessInfo.processInfo.environment
        return [
            "HOME": environment["HOME"] ?? "",
            "LANG": environment["LANG"] ?? "C.UTF-8",
            "LC_ALL": environment["LC_ALL"] ?? "C.UTF-8",
            "PATH": environment["PATH"] ?? "/usr/bin:/bin",
            "TMPDIR": environment["TMPDIR"] ?? FileManager.default.temporaryDirectory.path,
        ]
    }

    private static func validate(_ response: EvaluationScorerResponse) throws {
        guard response.schemaVersion == 1 else {
            throw EvaluationScoringError.invalidResponse(
                "unsupported schema_version \(response.schemaVersion)"
            )
        }
        guard response.score.isFinite, (0...1).contains(response.score) else {
            throw EvaluationScoringError.invalidResponse("score must be finite and from 0 through 1")
        }
        let metricIDs = response.metrics.map(\.id)
        guard Set(metricIDs).count == metricIDs.count,
              response.metrics.allSatisfy({ !$0.id.isEmpty && $0.value.isFinite }) else {
            throw EvaluationScoringError.invalidResponse(
                "metric ids must be unique and values must be finite"
            )
        }
        guard Set(response.hardFailures).count == response.hardFailures.count else {
            throw EvaluationScoringError.invalidResponse("hard_failures must be unique")
        }
    }
}

enum EvaluationScoringError: LocalizedError {
    case authorizationRequired
    case timeout(Double)
    case failed(status: Int32, diagnostics: String)
    case invalidResponse(String)

    var errorDescription: String? {
        switch self {
        case .authorizationRequired:
            "External evaluation scorers require --allow-external-scorer."
        case .timeout(let seconds):
            "External evaluation scorer exceeded its \(seconds)-second timeout."
        case .failed(let status, let diagnostics):
            diagnostics.isEmpty
                ? "External evaluation scorer exited with status \(status)."
                : "External evaluation scorer exited with status \(status): \(diagnostics)"
        case .invalidResponse(let detail):
            "Invalid external evaluation scorer response: \(detail)."
        }
    }
}
