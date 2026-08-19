import Foundation
import MereRunCore
import MereRunEvaluation

struct EvaluationPackIdentity: Codable, Hashable {
    let id: String
    let version: String
    let manifestSHA256: String
    let packSHA256: String
    let files: [EvaluationPackFilePin]
    let scorerKind: String
    let scorerExecutableSHA256: String?
}

struct EvaluationModelBinding: Codable, Hashable {
    let slot: String
    let id: String
    let installed: Bool
    let catalogRepository: String?
    let catalogRevision: String?
    let runtimeManifestID: String?
    let runtimeManifestSchemaVersion: Int?
    let runtimeManifestSHA256: String?

    static func resolve(
        slot: String,
        id: String,
        fileManager: FileManager = .default
    ) throws -> EvaluationModelBinding {
        let spec = ManagedModelCatalog.spec(for: id)
        let runtimeRoot = ManagedModelResolver.resolveInstalledModel(
            id: id,
            fileManager: fileManager
        )
        let manifestURL = runtimeRoot.map(MereRunModelManifest.url(in:))
        let manifestData: Data?
        let manifest: MereRunModelManifest?
        if let runtimeRoot,
           let manifestURL,
           fileManager.fileExists(atPath: manifestURL.path) {
            manifestData = try Data(contentsOf: manifestURL)
            manifest = try MereRunModelManifest.loadRequired(
                from: runtimeRoot,
                fileManager: fileManager
            )
        } else {
            manifestData = nil
            manifest = nil
        }
        return EvaluationModelBinding(
            slot: slot,
            id: id,
            installed: runtimeRoot != nil,
            catalogRepository: spec?.upstreamRepoId,
            catalogRevision: spec?.upstreamRevision,
            runtimeManifestID: manifest?.id,
            runtimeManifestSchemaVersion: manifest?.schemaVersion,
            runtimeManifestSHA256: manifestData.map(FusedBenchmarkHash.sha256)
        )
    }
}

struct EvaluationAdapterBinding: Codable, Hashable {
    let slot: String
    let catalogID: String?
    let byteCount: Int64
    let sha256: String
    let training: EvaluationTrainingBinding?
}

struct EvaluationTrainingBinding: Codable, Hashable {
    let manifestSHA256: String
    let schemaVersion: Int
    let format: String
    let baseModel: String
    let datasetFingerprint: String
    let seed: UInt64
    let status: String
}

struct ResolvedEvaluationAdapter {
    let binding: EvaluationAdapterBinding
    let path: String
}

struct EvaluationPromptBinding: Codable, Hashable {
    let id: String
    let sha256: String
}

struct EvaluationArmPlan: Codable, Hashable {
    let id: String
    let modelSlot: String
    let adapterSlot: String?
    let adapterScale: Double
    let promptSet: String?
    let profileIDs: [String]
}

struct EvaluationCasePlan: Codable, Hashable {
    let id: String
    let split: String
    let capabilityTags: [String]
    let contentSHA256: String
    let sourceFile: String
    let sourceLine: Int
}

struct EvaluationRunSettings: Codable, Hashable {
    let trials: Int
    let maxTokens: Int
    let contextSize: Int
    let logprobs: String
    let topLogprobs: Int
    let logResponses: Bool
    let externalScorerAuthorized: Bool
    let runtimeSeedControl: Bool
}

struct EvaluationRunPlan: Codable, Hashable {
    static let currentSchemaVersion = 1

    let schemaVersion: Int
    let runner: FusedBenchmarkRunnerIdentity
    let host: FusedBenchmarkHost
    let pack: EvaluationPackIdentity
    let models: [EvaluationModelBinding]
    let adapters: [EvaluationAdapterBinding]
    let prompts: [EvaluationPromptBinding]
    let arms: [EvaluationArmPlan]
    let profiles: [EvaluationSamplingProfile]
    let cases: [EvaluationCasePlan]
    let gates: [EvaluationGate]
    let settings: EvaluationRunSettings

    init(
        schemaVersion: Int = Self.currentSchemaVersion,
        runner: FusedBenchmarkRunnerIdentity,
        host: FusedBenchmarkHost,
        pack: EvaluationPackIdentity,
        models: [EvaluationModelBinding],
        adapters: [EvaluationAdapterBinding],
        prompts: [EvaluationPromptBinding],
        arms: [EvaluationArmPlan],
        profiles: [EvaluationSamplingProfile],
        cases: [EvaluationCasePlan],
        gates: [EvaluationGate],
        settings: EvaluationRunSettings
    ) {
        self.schemaVersion = schemaVersion
        self.runner = runner
        self.host = host
        self.pack = pack
        self.models = models
        self.adapters = adapters
        self.prompts = prompts
        self.arms = arms
        self.profiles = profiles
        self.cases = cases
        self.gates = gates
        self.settings = settings
    }

    var expectedResultKeys: Set<EvaluationResultKey> {
        var keys: Set<EvaluationResultKey> = []
        for arm in arms {
            for profileID in arm.profileIDs {
                for trial in 1...settings.trials {
                    for benchmarkCase in cases {
                        keys.insert(EvaluationResultKey(
                            armID: arm.id,
                            profileID: profileID,
                            trial: trial,
                            caseID: benchmarkCase.id
                        ))
                    }
                }
            }
        }
        return keys
    }

    func contentSHA256() throws -> String {
        FusedBenchmarkHash.sha256(try EvaluationJSON.canonicalEncoder.encode(self))
    }
}

struct EvaluationResultKey: Codable, Hashable {
    let armID: String
    let profileID: String
    let trial: Int
    let caseID: String
}

struct EvaluationResultRow: Codable, Hashable {
    let armID: String
    let modelID: String
    let adapterSHA256: String?
    let profileID: String
    let trial: Int
    let caseID: String
    let caseSHA256: String
    let split: String
    let capabilityTags: [String]
    let passed: Bool
    let score: Double
    let metrics: [EvaluationMetric]
    let failedChecks: [String]
    let hardFailures: [String]
    let generationSeconds: Double
    let scoringSeconds: Double
    let tokensGenerated: Int
    let logprobs: FusedBenchmarkLogprobMetrics?
    let responseSHA256: String
    let response: String?

    var key: EvaluationResultKey {
        EvaluationResultKey(
            armID: armID,
            profileID: profileID,
            trial: trial,
            caseID: caseID
        )
    }
}

struct EvaluationGateResult: Codable, Hashable {
    let id: String
    let required: Bool
    let aggregation: String
    let comparator: String
    let threshold: Double
    let evaluatedRows: Int
    let value: Double?
    let passed: Bool
    let error: String?
}

struct EvaluationSelectiveAccuracy: Codable, Hashable {
    let threshold: Double
    let retained: Int
    let accuracy: Double?
}

struct EvaluationCalibration: Codable, Hashable {
    let armID: String
    let profileID: String
    let evaluatedRows: Int
    let expectedCalibrationError: Double?
    let selectiveAccuracy: [EvaluationSelectiveAccuracy]
    let fragilePasses: [String]
    let confidentFailures: [String]

    static func calculate(
        armID: String,
        profileID: String,
        rows: [EvaluationResultRow]
    ) -> EvaluationCalibration {
        let labeled = rows.compactMap { row -> (EvaluationResultRow, Double)? in
            guard let mean = row.logprobs?.summary.meanRawLogprob else { return nil }
            return (row, min(max(exp(mean), 0), 1))
        }
        let bucketCount = 10
        var weightedError = 0.0
        for bucket in 0..<bucketCount {
            let lower = Double(bucket) / Double(bucketCount)
            let upper = Double(bucket + 1) / Double(bucketCount)
            let members = labeled.filter { item in
                item.1 >= lower && (bucket == bucketCount - 1 ? item.1 <= upper : item.1 < upper)
            }
            guard !members.isEmpty else { continue }
            let confidence = members.map { $0.1 }.reduce(0, +) / Double(members.count)
            let accuracy = Double(members.filter { $0.0.passed }.count) / Double(members.count)
            weightedError += abs(confidence - accuracy) * Double(members.count)
        }
        let thresholds = [0.25, 0.5, 0.75, 0.9]
        return EvaluationCalibration(
            armID: armID,
            profileID: profileID,
            evaluatedRows: labeled.count,
            expectedCalibrationError: labeled.isEmpty ? nil : weightedError / Double(labeled.count),
            selectiveAccuracy: thresholds.map { threshold in
                let retained = labeled.filter { $0.1 >= threshold }
                return EvaluationSelectiveAccuracy(
                    threshold: threshold,
                    retained: retained.count,
                    accuracy: retained.isEmpty
                        ? nil
                        : Double(retained.filter { $0.0.passed }.count) / Double(retained.count)
                )
            },
            fragilePasses: labeled.filter { $0.0.passed && $0.1 < 0.25 }.map { $0.0.caseID },
            confidentFailures: labeled.filter { !$0.0.passed && $0.1 >= 0.75 }.map { $0.0.caseID }
        )
    }
}

struct EvaluationProgress: Codable, Hashable {
    let expectedRows: Int
    let completedRows: Int
    let remainingRows: Int
    let complete: Bool
}

struct EvaluationRunReport: Codable {
    static let currentSchemaVersion = 1

    let schemaVersion: Int
    let planSHA256: String
    let plan: EvaluationRunPlan
    let results: [EvaluationResultRow]
    let progress: EvaluationProgress
    let gates: [EvaluationGateResult]
    let calibration: [EvaluationCalibration]
    let promotionEligible: Bool

    init(plan: EvaluationRunPlan, results: [EvaluationResultRow]) throws {
        try Self.validate(results: results, against: plan)
        schemaVersion = Self.currentSchemaVersion
        planSHA256 = try plan.contentSHA256()
        self.plan = plan
        self.results = results.sorted(by: Self.resultOrder)
        let expected = plan.expectedResultKeys
        let completed = Set(results.map(\.key)).intersection(expected).count
        progress = EvaluationProgress(
            expectedRows: expected.count,
            completedRows: completed,
            remainingRows: max(0, expected.count - completed),
            complete: completed == expected.count
        )
        gates = EvaluationGateEvaluator.evaluate(plan: plan, results: results)
        calibration = Self.calibrationGroups(results)
        promotionEligible = progress.complete
            && gates.contains(where: \.required)
            && gates.filter(\.required).allSatisfy(\.passed)
    }

    func jsonData() throws -> Data {
        try EvaluationJSON.prettyEncoder.encode(self)
    }

    func jsonString() throws -> String {
        String(decoding: try jsonData(), as: UTF8.self)
    }

    func renderText(dryRun: Bool) -> String {
        let modelSummary = plan.models.map { "\($0.slot)=\($0.id)" }.joined(separator: ", ")
        let adapterSummary = plan.adapters.map { "\($0.slot)=\($0.sha256)" }.joined(separator: ", ")
        var lines = [
            "External evaluation pack \(plan.pack.id)@\(plan.pack.version)",
            "pack sha256: \(plan.pack.packSHA256)",
            "runner sha256: \(plan.runner.executableSHA256)",
            "models: \(modelSummary)",
            "adapters: \(adapterSummary)",
            "arms: \(plan.arms.map(\.id).joined(separator: ", "))",
            "profiles: \(plan.profiles.map(\.id).joined(separator: ", "))",
            "cases: \(plan.cases.count)",
            "trials: \(plan.settings.trials)",
            "rows: \(progress.completedRows)/\(progress.expectedRows)",
        ]
        if dryRun {
            lines.append("dry run: no models loaded, no inference run, and no scorer executed")
        } else {
            lines.append("promotion eligible: \(promotionEligible)")
            for gate in gates {
                let value = gate.value.map { String($0) } ?? "unavailable"
                let status = gate.passed ? "pass" : "fail"
                lines.append(
                    "gate \(gate.id): \(status) "
                        + "(value \(value) \(gate.comparator) \(gate.threshold))"
                )
            }
        }
        return lines.joined(separator: "\n")
    }

    private static func calibrationGroups(_ rows: [EvaluationResultRow]) -> [EvaluationCalibration] {
        let grouped = Dictionary(grouping: rows) { row in
            "\(row.armID)\u{0}\(row.profileID)"
        }
        return grouped.keys.sorted().compactMap { key in
            guard let rows = grouped[key], let first = rows.first else { return nil }
            return EvaluationCalibration.calculate(
                armID: first.armID,
                profileID: first.profileID,
                rows: rows
            )
        }
    }

    private static func resultOrder(_ lhs: EvaluationResultRow, _ rhs: EvaluationResultRow) -> Bool {
        let left = [lhs.armID, lhs.profileID, String(format: "%09d", lhs.trial), lhs.caseID]
        let right = [rhs.armID, rhs.profileID, String(format: "%09d", rhs.trial), rhs.caseID]
        return left.lexicographicallyPrecedes(right)
    }

    private static func validate(
        results: [EvaluationResultRow],
        against plan: EvaluationRunPlan
    ) throws {
        let keys = results.map(\.key)
        guard Set(keys).count == keys.count else {
            throw EvaluationReportError.invalid("duplicate result rows")
        }
        guard Set(keys).isSubset(of: plan.expectedResultKeys) else {
            throw EvaluationReportError.invalid("result row is outside the run plan")
        }
        let arms = Dictionary(uniqueKeysWithValues: plan.arms.map { ($0.id, $0) })
        let models = Dictionary(uniqueKeysWithValues: plan.models.map { ($0.slot, $0) })
        let adapters = Dictionary(uniqueKeysWithValues: plan.adapters.map { ($0.slot, $0) })
        let cases = Dictionary(uniqueKeysWithValues: plan.cases.map { ($0.id, $0) })
        for row in results {
            guard let arm = arms[row.armID],
                  arm.profileIDs.contains(row.profileID),
                  let model = models[arm.modelSlot],
                  model.id == row.modelID,
                  let benchmarkCase = cases[row.caseID],
                  benchmarkCase.contentSHA256 == row.caseSHA256,
                  benchmarkCase.split == row.split,
                  benchmarkCase.capabilityTags == row.capabilityTags else {
                throw EvaluationReportError.invalid("result identity does not match the run plan")
            }
            let expectedAdapterSHA256 = arm.adapterSlot.flatMap { adapters[$0]?.sha256 }
            guard row.adapterSHA256 == expectedAdapterSHA256 else {
                throw EvaluationReportError.invalid("result adapter does not match its arm")
            }
            guard row.score.isFinite,
                  (0...1).contains(row.score),
                  row.generationSeconds.isFinite,
                  row.generationSeconds >= 0,
                  row.scoringSeconds.isFinite,
                  row.scoringSeconds >= 0,
                  row.tokensGenerated >= 0 else {
                throw EvaluationReportError.invalid("result contains invalid numeric values")
            }
            let metricIDs = row.metrics.map(\.id)
            guard Set(metricIDs).count == metricIDs.count,
                  row.metrics.allSatisfy({ !$0.id.isEmpty && $0.value.isFinite }) else {
                throw EvaluationReportError.invalid("result metrics are invalid")
            }
            guard isSHA256(row.responseSHA256) else {
                throw EvaluationReportError.invalid("result response hash is invalid")
            }
            if let response = row.response,
               FusedBenchmarkHash.sha256(response) != row.responseSHA256 {
                throw EvaluationReportError.invalid("logged response does not match its hash")
            }
        }
    }

    private static func isSHA256(_ value: String) -> Bool {
        value.count == 64 && value.unicodeScalars.allSatisfy {
            CharacterSet(charactersIn: "0123456789abcdef").contains($0)
        }
    }
}

enum EvaluationReportError: LocalizedError {
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let detail):
            "Invalid evaluation report: \(detail)."
        }
    }
}

enum EvaluationGateEvaluator {
    static func evaluate(
        plan: EvaluationRunPlan,
        results: [EvaluationResultRow]
    ) -> [EvaluationGateResult] {
        plan.gates.map { gate in
            let filtered = results.filter { matches($0, filter: gate.filter) }
            guard !filtered.isEmpty else {
                return EvaluationGateResult(
                    id: gate.id,
                    required: gate.required,
                    aggregation: gate.aggregation.rawValue,
                    comparator: gate.comparator.rawValue,
                    threshold: gate.threshold,
                    evaluatedRows: 0,
                    value: nil,
                    passed: false,
                    error: "gate filter selected no completed rows"
                )
            }
            let value: Double?
            switch gate.aggregation {
            case .passRate:
                value = Double(filtered.filter(\.passed).count) / Double(filtered.count)
            case .meanScore:
                value = filtered.map(\.score).reduce(0, +) / Double(filtered.count)
            case .hardFailureCount:
                value = Double(filtered.reduce(0) { $0 + $1.hardFailures.count })
            case .metricMean:
                let metricValues = filtered.compactMap { row in
                    row.metrics.first { $0.id == gate.metricID }?.value
                }
                value = metricValues.count != filtered.count
                    ? nil
                    : metricValues.reduce(0, +) / Double(metricValues.count)
            }
            guard let value else {
                return EvaluationGateResult(
                    id: gate.id,
                    required: gate.required,
                    aggregation: gate.aggregation.rawValue,
                    comparator: gate.comparator.rawValue,
                    threshold: gate.threshold,
                    evaluatedRows: filtered.count,
                    value: nil,
                    passed: false,
                    error: "gate metric is unavailable"
                )
            }
            return EvaluationGateResult(
                id: gate.id,
                required: gate.required,
                aggregation: gate.aggregation.rawValue,
                comparator: gate.comparator.rawValue,
                threshold: gate.threshold,
                evaluatedRows: filtered.count,
                value: value,
                passed: compare(value, gate.comparator, gate.threshold),
                error: nil
            )
        }
    }

    private static func matches(
        _ row: EvaluationResultRow,
        filter: EvaluationGateFilter?
    ) -> Bool {
        guard let filter else { return true }
        if let splits = filter.splits, !splits.map(\.rawValue).contains(row.split) {
            return false
        }
        if let arms = filter.arms, !arms.contains(row.armID) {
            return false
        }
        if let profiles = filter.profiles, !profiles.contains(row.profileID) {
            return false
        }
        if let capabilities = filter.capabilities,
           !Set(capabilities).isSubset(of: Set(row.capabilityTags)) {
            return false
        }
        return true
    }

    private static func compare(
        _ value: Double,
        _ comparator: EvaluationGateComparator,
        _ threshold: Double
    ) -> Bool {
        switch comparator {
        case .greaterThanOrEqual:
            value >= threshold
        case .lessThanOrEqual:
            value <= threshold
        case .equal:
            abs(value - threshold) <= 1e-12
        }
    }
}

struct EvaluationCheckpoint: Codable {
    static let currentSchemaVersion = 1

    let schemaVersion: Int
    let createdAt: Date
    let updatedAt: Date
    let planSHA256: String
    let plan: EvaluationRunPlan
    let results: [EvaluationResultRow]

    init(
        plan: EvaluationRunPlan,
        results: [EvaluationResultRow],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) throws {
        schemaVersion = Self.currentSchemaVersion
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        planSHA256 = try plan.contentSHA256()
        self.plan = plan
        self.results = results
        _ = try validatedResults(for: plan)
    }

    func validatedResults(for expectedPlan: EvaluationRunPlan) throws -> [EvaluationResultRow] {
        guard schemaVersion == Self.currentSchemaVersion else {
            throw EvaluationCheckpointError.invalid("unsupported schema version \(schemaVersion)")
        }
        let storedPlanSHA256 = try plan.contentSHA256()
        guard planSHA256 == storedPlanSHA256 else {
            throw EvaluationCheckpointError.invalid("stored plan hash does not match its payload")
        }
        let expectedPlanSHA256 = try expectedPlan.contentSHA256()
        guard planSHA256 == expectedPlanSHA256 else {
            throw EvaluationCheckpointError.invalid("run plan changed")
        }
        let keys = results.map(\.key)
        guard Set(keys).count == keys.count else {
            throw EvaluationCheckpointError.invalid("duplicate completed rows")
        }
        let unexpected = Set(keys).subtracting(expectedPlan.expectedResultKeys)
        guard unexpected.isEmpty else {
            throw EvaluationCheckpointError.invalid("checkpoint contains rows outside the run plan")
        }
        do {
            _ = try EvaluationRunReport(plan: expectedPlan, results: results)
        } catch {
            throw EvaluationCheckpointError.invalid("checkpoint result validation failed: \(error)")
        }
        return results
    }
}

enum EvaluationCheckpointError: LocalizedError {
    case invalid(String)
    case exists(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let detail):
            "Invalid evaluation checkpoint: \(detail)."
        case .exists(let path):
            "Evaluation checkpoint already exists at \(path); use --resume or choose another path."
        }
    }
}

struct EvaluationCheckpointStore {
    let url: URL

    func initialize(plan: EvaluationRunPlan) throws {
        guard !FileManager.default.fileExists(atPath: url.path) else {
            throw EvaluationCheckpointError.exists(url.path)
        }
        try write(EvaluationCheckpoint(plan: plan, results: []))
    }

    func loadResults(validating plan: EvaluationRunPlan) throws -> [EvaluationResultRow] {
        let checkpoint = try EvaluationJSON.decoder.decode(
            EvaluationCheckpoint.self,
            from: Data(contentsOf: url)
        )
        return try checkpoint.validatedResults(for: plan)
    }

    func write(plan: EvaluationRunPlan, results: [EvaluationResultRow]) throws {
        let createdAt: Date
        if FileManager.default.fileExists(atPath: url.path),
           let existing = try? EvaluationJSON.decoder.decode(
               EvaluationCheckpoint.self,
               from: Data(contentsOf: url)
           ) {
            createdAt = existing.createdAt
        } else {
            createdAt = Date()
        }
        try write(EvaluationCheckpoint(
            plan: plan,
            results: results,
            createdAt: createdAt,
            updatedAt: Date()
        ))
    }

    private func write(_ checkpoint: EvaluationCheckpoint) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try EvaluationJSON.prettyEncoder.encode(checkpoint).write(to: url, options: .atomic)
    }
}

struct EvaluationPromotionPayload: Codable, Hashable {
    let schemaVersion: Int
    let createdAt: Date
    let reportSHA256: String
    let planSHA256: String
    let pack: EvaluationPackIdentity
    let models: [EvaluationModelBinding]
    let adapters: [EvaluationAdapterBinding]
    let gates: [EvaluationGateResult]
}

struct EvaluationPromotionReceipt: Codable, Hashable {
    static let currentSchemaVersion = 1

    let schemaVersion: Int
    let payloadSHA256: String
    let payload: EvaluationPromotionPayload

    init(reportData: Data, report: EvaluationRunReport, createdAt: Date = Date()) throws {
        guard report.progress.complete else {
            throw EvaluationPromotionError.incomplete
        }
        guard report.gates.contains(where: \.required) else {
            throw EvaluationPromotionError.noRequiredGates
        }
        let failed = report.gates.filter { $0.required && !$0.passed }.map(\.id)
        guard failed.isEmpty, report.promotionEligible else {
            throw EvaluationPromotionError.failedGates(failed)
        }
        schemaVersion = Self.currentSchemaVersion
        payload = EvaluationPromotionPayload(
            schemaVersion: 1,
            createdAt: createdAt,
            reportSHA256: FusedBenchmarkHash.sha256(reportData),
            planSHA256: report.planSHA256,
            pack: report.plan.pack,
            models: report.plan.models,
            adapters: report.plan.adapters,
            gates: report.gates
        )
        payloadSHA256 = FusedBenchmarkHash.sha256(
            try EvaluationJSON.canonicalEncoder.encode(payload)
        )
    }
}

enum EvaluationPromotionError: LocalizedError {
    case incomplete
    case noRequiredGates
    case failedGates([String])

    var errorDescription: String? {
        switch self {
        case .incomplete:
            "Cannot promote an incomplete evaluation report."
        case .noRequiredGates:
            "Cannot promote an evaluation report without a required gate."
        case .failedGates(let gates):
            "Cannot promote an evaluation report with failed required gates: \(gates.joined(separator: ", "))."
        }
    }
}

enum EvaluationJSON {
    static var canonicalEncoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    static var prettyEncoder: JSONEncoder {
        let encoder = canonicalEncoder
        encoder.outputFormatting.insert(.prettyPrinted)
        return encoder
    }

    static var decoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
