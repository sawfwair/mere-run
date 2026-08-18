import ArgumentParser
import Crypto
import Foundation
import MereRunCore

enum FusedBenchmarkSuiteSelection: String, CaseIterable, ExpressibleByArgument, Codable {
    case lite
    case comprehensive
}

enum FusedBenchmarkLogprobMode: String, CaseIterable, ExpressibleByArgument, Codable {
    case summary
    case tokens
    case top

    func capture(topLogprobs: Int) -> ChatLogprobCapture {
        switch self {
        case .summary:
            return .summary
        case .tokens:
            return .tokens
        case .top:
            return .top(topLogprobs)
        }
    }
}

enum FusedBenchmarkLane: String, CaseIterable, Codable {
    case chat
    case code
    case tools
    case vision
}

enum FusedBenchmarkAdapter: String, Codable {
    case mereChat = "mere-chat"
    case mereTool = "mere-tool"
    case mereVision = "mere-vision"
    case humanEval = "human-eval"
    case externalChat = "external-chat"
    case externalCode = "external-code"
    case externalTool = "external-tool"
    case externalVision = "external-vision"

    var requiresImportedFixture: Bool {
        switch self {
        case .mereChat, .mereTool, .mereVision, .humanEval:
            return false
        case .externalChat, .externalCode, .externalTool, .externalVision:
            return true
        }
    }

    var lane: FusedBenchmarkLane {
        switch self {
        case .mereChat, .externalChat:
            return .chat
        case .humanEval, .externalCode:
            return .code
        case .mereTool, .externalTool:
            return .tools
        case .mereVision, .externalVision:
            return .vision
        }
    }
}

struct FusedBenchmarkSource: Codable, Hashable {
    let id: String
    let name: String
    let version: String
    let license: String
    let sourceURL: String
    let redistribution: String
}

struct FusedBenchmarkCaseDescriptor: Codable, Hashable {
    let id: String
    let sourceID: String
    let sourceCaseID: String
    let adapter: FusedBenchmarkAdapter
    let capabilityTags: [String]
    let difficulty: String
    let selectionRationale: String
    let lite: Bool
}

struct FusedBenchmarkManifest: Codable {
    let schemaVersion: Int
    let id: String
    let version: String
    let description: String
    let defaultLiteTrials: Int
    let defaultComprehensiveTrials: Int
    let sources: [FusedBenchmarkSource]
    let cases: [FusedBenchmarkCaseDescriptor]

    static func bundled() throws -> FusedBenchmarkManifest {
        let nested = Bundle.module.url(
            forResource: "mere-fused-v1",
            withExtension: "json",
            subdirectory: "BenchmarkSuites"
        )
        let flat = Bundle.module.url(forResource: "mere-fused-v1", withExtension: "json")
        guard let url = nested ?? flat else {
            throw FusedBenchmarkError.invalidManifest("bundled mere-fused-v1.json was not found")
        }
        return try load(from: url)
    }

    static func load(from url: URL) throws -> FusedBenchmarkManifest {
        let manifest = try JSONDecoder().decode(
            FusedBenchmarkManifest.self,
            from: Data(contentsOf: url)
        )
        try manifest.validate()
        return manifest
    }

    func selectedCases(for suite: FusedBenchmarkSuiteSelection) -> [FusedBenchmarkCaseDescriptor] {
        switch suite {
        case .lite:
            return cases.filter(\.lite)
        case .comprehensive:
            return cases
        }
    }

    func defaultTrials(for suite: FusedBenchmarkSuiteSelection) -> Int {
        switch suite {
        case .lite:
            return defaultLiteTrials
        case .comprehensive:
            return defaultComprehensiveTrials
        }
    }

    func source(id: String) -> FusedBenchmarkSource? {
        sources.first { $0.id == id }
    }

    func contentSHA256() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return FusedBenchmarkHash.sha256(try encoder.encode(self))
    }

    private func validate() throws {
        guard schemaVersion == 1 else {
            throw FusedBenchmarkError.invalidManifest(
                "unsupported schemaVersion \(schemaVersion)"
            )
        }
        guard !id.isEmpty, !version.isEmpty, !cases.isEmpty else {
            throw FusedBenchmarkError.invalidManifest("id, version, and cases are required")
        }
        let sourceIDs = Set(sources.map(\.id))
        guard sourceIDs.count == sources.count else {
            throw FusedBenchmarkError.invalidManifest("source ids must be unique")
        }
        let caseIDs = Set(cases.map(\.id))
        guard caseIDs.count == cases.count else {
            throw FusedBenchmarkError.invalidManifest("case ids must be unique")
        }
        for benchmarkCase in cases {
            guard sourceIDs.contains(benchmarkCase.sourceID) else {
                throw FusedBenchmarkError.invalidManifest(
                    "case \(benchmarkCase.id) references unknown source \(benchmarkCase.sourceID)"
                )
            }
            guard !benchmarkCase.capabilityTags.isEmpty,
                  !benchmarkCase.selectionRationale.isEmpty else {
                throw FusedBenchmarkError.invalidManifest(
                    "case \(benchmarkCase.id) needs capability tags and a selection rationale"
                )
            }
        }
        let comprehensiveFamilies = Set(cases.map(\.sourceID))
        let liteFamilies = Set(cases.filter(\.lite).map(\.sourceID))
        guard liteFamilies == comprehensiveFamilies else {
            let missing = comprehensiveFamilies.subtracting(liteFamilies).sorted()
            throw FusedBenchmarkError.invalidManifest(
                "Lite must represent every source family; missing \(missing.joined(separator: ", "))"
            )
        }
        guard defaultLiteTrials > 0, defaultComprehensiveTrials > 0 else {
            throw FusedBenchmarkError.invalidManifest("default trial counts must be positive")
        }
    }
}

struct FusedBenchmarkProvenance: Codable, Hashable {
    let source: String
    let sourceVersion: String
    let originalID: String
    let license: String
    let sourceURL: String
    let redistribution: String
    let referenceSHA256: String
    let contentSHA256: String?
    let imageSHA256: String?
    let capabilityTags: [String]
    let difficulty: String
    let selectionRationale: String

    init(
        descriptor: FusedBenchmarkCaseDescriptor,
        source: FusedBenchmarkSource,
        contentSHA256: String?,
        imageSHA256: String? = nil,
        sourceVersion: String? = nil,
        originalID: String? = nil
    ) {
        let resolvedVersion = sourceVersion ?? source.version
        let resolvedOriginalID = originalID ?? descriptor.sourceCaseID
        self.source = source.name
        self.sourceVersion = resolvedVersion
        self.originalID = resolvedOriginalID
        self.license = source.license
        self.sourceURL = source.sourceURL
        self.redistribution = source.redistribution
        self.referenceSHA256 = FusedBenchmarkHash.sha256(
            "\(source.id)|\(resolvedVersion)|\(resolvedOriginalID)|\(descriptor.adapter.rawValue)"
        )
        self.contentSHA256 = contentSHA256
        self.imageSHA256 = imageSHA256
        self.capabilityTags = descriptor.capabilityTags
        self.difficulty = descriptor.difficulty
        self.selectionRationale = descriptor.selectionRationale
    }
}

struct FusedBenchmarkSamplingProfile: Codable, Hashable {
    let name: String
    let temperature: Double
    let topP: Double
    let topK: Int
    let minP: Double
    let reasoningEffort: Double?
    let reasoningTier: String?
    let showThinking: Bool

    static func nativeProfiles(
        modelID: String,
        suite: FusedBenchmarkSuiteSelection
    ) -> [FusedBenchmarkSamplingProfile] {
        if Q35Resources.isQ38ModelId(modelID) {
            let tiers: [(String, Double)] = suite == .lite
                ? [("medium", 0.5)]
                : [("low", 0.2), ("medium", 0.5), ("xhigh", 1.0)]
            return tiers.map { tier, strength in
                FusedBenchmarkSamplingProfile(
                    name: "qwen3.8-native-\(tier)",
                    temperature: 1.0,
                    topP: 0.95,
                    topK: 20,
                    minP: 0,
                    reasoningEffort: strength,
                    reasoningTier: tier,
                    showThinking: true
                )
            }
        }
        if NemotronHResources.handles(modelSpec: modelID) {
            return [FusedBenchmarkSamplingProfile(
                name: "nemotron-lightning-native",
                temperature: NemotronHResources.recommendedTemperature,
                topP: NemotronHResources.recommendedTopP,
                topK: 0,
                minP: 0,
                reasoningEffort: nil,
                reasoningTier: nil,
                showThinking: true
            )]
        }
        if LagunaResources.handles(modelSpec: modelID) {
            return [FusedBenchmarkSamplingProfile(
                name: "laguna-native",
                temperature: LagunaResources.recommendedTemperature,
                topP: LagunaResources.recommendedTopP,
                topK: LagunaResources.recommendedTopK,
                minP: LagunaResources.recommendedMinP,
                reasoningEffort: nil,
                reasoningTier: nil,
                showThinking: true
            )]
        }
        if let recommended = Q35Resources.recommendedSampling(forModelId: modelID) {
            return [FusedBenchmarkSamplingProfile(
                name: "catalog-native",
                temperature: recommended.temperature,
                topP: recommended.topP,
                topK: recommended.topK,
                minP: 0,
                reasoningEffort: nil,
                reasoningTier: nil,
                showThinking: Q35Resources.thinkingDefault(forModelId: modelID)
            )]
        }
        // The fused quality lane intentionally never uses greedy decoding.
        return [FusedBenchmarkSamplingProfile(
            name: "sampled-fallback",
            temperature: 0.7,
            topP: 0.9,
            topK: 0,
            minP: 0,
            reasoningEffort: nil,
            reasoningTier: nil,
            showThinking: false
        )]
    }
}

struct FusedExternalBenchmarkCase: Codable {
    enum Kind: String, Codable {
        case chat
        case code
        case tool
        case vision
    }

    let id: String
    let kind: Kind
    let sourceVersion: String
    let originalID: String
    let messages: [ChatMessage]
    let tools: [ToolDefinition]?
    let requiredPhrases: [String]?
    let forbiddenPhrases: [String]?
    let expectedToolName: String?
    let expectedArguments: [String: String]?
    let entryPoint: String?
    let tests: String?
    let imageSHA256: String?
    let contentSHA256: String

    func computedContentSHA256() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let payload = FusedExternalBenchmarkHashPayload(
            id: id,
            kind: kind,
            sourceVersion: sourceVersion,
            originalID: originalID,
            messages: messages,
            tools: tools,
            requiredPhrases: requiredPhrases,
            forbiddenPhrases: forbiddenPhrases,
            expectedToolName: expectedToolName,
            expectedArguments: expectedArguments,
            entryPoint: entryPoint,
            tests: tests,
            imageSHA256: imageSHA256
        )
        return FusedBenchmarkHash.sha256(try encoder.encode(payload))
    }

    func stamped() throws -> FusedExternalBenchmarkCase {
        let imageSHA256 = try computedImageSHA256()
        let unstamped = FusedExternalBenchmarkCase(
            id: id,
            kind: kind,
            sourceVersion: sourceVersion,
            originalID: originalID,
            messages: messages,
            tools: tools,
            requiredPhrases: requiredPhrases,
            forbiddenPhrases: forbiddenPhrases,
            expectedToolName: expectedToolName,
            expectedArguments: expectedArguments,
            entryPoint: entryPoint,
            tests: tests,
            imageSHA256: imageSHA256,
            contentSHA256: ""
        )
        return FusedExternalBenchmarkCase(
            id: unstamped.id,
            kind: unstamped.kind,
            sourceVersion: unstamped.sourceVersion,
            originalID: unstamped.originalID,
            messages: unstamped.messages,
            tools: unstamped.tools,
            requiredPhrases: unstamped.requiredPhrases,
            forbiddenPhrases: unstamped.forbiddenPhrases,
            expectedToolName: unstamped.expectedToolName,
            expectedArguments: unstamped.expectedArguments,
            entryPoint: unstamped.entryPoint,
            tests: unstamped.tests,
            imageSHA256: unstamped.imageSHA256,
            contentSHA256: try unstamped.computedContentSHA256()
        )
    }

    private func computedImageSHA256() throws -> String? {
        let imagePaths = messages.compactMap(\.imageUrl)
        guard kind == .vision else {
            guard imagePaths.isEmpty, imageSHA256 == nil else {
                throw FusedBenchmarkError.invalidExternalFixture(
                    "\(id) is not a vision case but declares image content"
                )
            }
            return nil
        }
        guard imagePaths.count == 1, let path = imagePaths.first else {
            throw FusedBenchmarkError.invalidExternalFixture(
                "\(id) vision fixture must declare exactly one image"
            )
        }
        let url = URL(fileURLWithPath: path).standardizedFileURL
        guard url.path.hasPrefix("/"), FileManager.default.fileExists(atPath: url.path) else {
            throw FusedBenchmarkError.invalidExternalFixture(
                "\(id) vision image must be an existing absolute path"
            )
        }
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true else {
            throw FusedBenchmarkError.invalidExternalFixture(
                "\(id) vision image must be a regular non-symlink file"
            )
        }
        return FusedBenchmarkHash.sha256(try Data(contentsOf: url))
    }
}

private struct FusedExternalBenchmarkHashPayload: Codable {
    let id: String
    let kind: FusedExternalBenchmarkCase.Kind
    let sourceVersion: String
    let originalID: String
    let messages: [ChatMessage]
    let tools: [ToolDefinition]?
    let requiredPhrases: [String]?
    let forbiddenPhrases: [String]?
    let expectedToolName: String?
    let expectedArguments: [String: String]?
    let entryPoint: String?
    let tests: String?
    let imageSHA256: String?
}

enum FusedExternalBenchmarkLoader {
    static func load(paths: [String]) throws -> [String: FusedExternalBenchmarkCase] {
        var loaded: [String: FusedExternalBenchmarkCase] = [:]
        for path in paths {
            let url = URL(fileURLWithPath: path).standardizedFileURL
            let content = try String(contentsOf: url, encoding: .utf8)
            for (index, line) in content.split(whereSeparator: { $0.isNewline }).enumerated() {
                let benchmarkCase: FusedExternalBenchmarkCase
                do {
                    benchmarkCase = try JSONDecoder().decode(
                        FusedExternalBenchmarkCase.self,
                        from: Data(line.utf8)
                    )
                } catch {
                    throw FusedBenchmarkError.invalidExternalFixture(
                        "\(url.path):\(index + 1): \(error.localizedDescription)"
                    )
                }
                let stamped = try benchmarkCase.stamped()
                guard benchmarkCase.imageSHA256 == stamped.imageSHA256 else {
                    throw FusedBenchmarkError.invalidExternalFixture(
                        "\(benchmarkCase.id) image SHA-256 does not match"
                    )
                }
                guard benchmarkCase.contentSHA256 == stamped.contentSHA256 else {
                    throw FusedBenchmarkError.invalidExternalFixture(
                        "\(benchmarkCase.id) content SHA-256 does not match"
                    )
                }
                guard loaded[benchmarkCase.id] == nil else {
                    throw FusedBenchmarkError.invalidExternalFixture(
                        "duplicate external case id \(benchmarkCase.id)"
                    )
                }
                loaded[benchmarkCase.id] = benchmarkCase
            }
        }
        return loaded
    }
}

enum FusedExternalBenchmarkContract {
    static func validate(
        _ benchmarkCase: FusedExternalBenchmarkCase,
        for descriptor: FusedBenchmarkCaseDescriptor,
        source: FusedBenchmarkSource
    ) throws {
        let expectedKind: FusedExternalBenchmarkCase.Kind
        switch descriptor.adapter {
        case .externalChat:
            expectedKind = .chat
        case .externalCode:
            expectedKind = .code
        case .externalTool:
            expectedKind = .tool
        case .externalVision:
            expectedKind = .vision
        case .mereChat, .mereTool, .mereVision, .humanEval:
            throw FusedBenchmarkError.invalidManifest(
                "embedded adapter reached external fixture validation"
            )
        }
        guard benchmarkCase.kind == expectedKind else {
            throw FusedBenchmarkError.invalidExternalFixture(
                "\(descriptor.id) has kind \(benchmarkCase.kind.rawValue); expected \(expectedKind.rawValue)"
            )
        }
        guard benchmarkCase.originalID == descriptor.sourceCaseID else {
            throw FusedBenchmarkError.invalidExternalFixture(
                "\(descriptor.id) has original id \(benchmarkCase.originalID); expected \(descriptor.sourceCaseID)"
            )
        }
        if source.version != "pinned-by-import",
           benchmarkCase.sourceVersion != source.version {
            throw FusedBenchmarkError.invalidExternalFixture(
                "\(descriptor.id) has source version \(benchmarkCase.sourceVersion); expected \(source.version)"
            )
        }
    }
}

struct FusedBenchmarkLowConfidenceSpan: Codable, Hashable {
    let startToken: Int
    let endToken: Int
    let region: String
    let minimumRawLogprob: Double
    let meanRawLogprob: Double
}

struct FusedBenchmarkLogprobMetrics: Codable, Hashable {
    let summary: ChatLogprobSummary
    let captureSeconds: Double
    let lowConfidenceSpans: [FusedBenchmarkLowConfidenceSpan]

    init(_ diagnostics: ChatLogprobDiagnostics) {
        summary = diagnostics.summary
        captureSeconds = diagnostics.captureSeconds
        lowConfidenceSpans = Self.spans(tokens: diagnostics.tokens ?? [])
    }

    private static func spans(tokens: [ChatTokenLogprob]) -> [FusedBenchmarkLowConfidenceSpan] {
        let threshold = log(0.1)
        var spans: [FusedBenchmarkLowConfidenceSpan] = []
        var start: Int?
        var values: [Double] = []
        var region = ChatLogprobRegion.unknown.rawValue

        func appendSpan(endingBefore index: Int) {
            guard let start, !values.isEmpty else { return }
            spans.append(FusedBenchmarkLowConfidenceSpan(
                startToken: start,
                endToken: index - 1,
                region: region,
                minimumRawLogprob: values.min() ?? 0,
                meanRawLogprob: values.reduce(0, +) / Double(values.count)
            ))
        }

        for (index, token) in tokens.enumerated() {
            if token.rawLogprob < threshold {
                if start != nil, region != token.region.rawValue {
                    appendSpan(endingBefore: index)
                    start = nil
                    values = []
                }
                if start == nil {
                    start = index
                    region = token.region.rawValue
                }
                values.append(token.rawLogprob)
            } else {
                appendSpan(endingBefore: index)
                start = nil
                values = []
            }
        }
        appendSpan(endingBefore: tokens.count)
        return spans
    }
}

struct FusedBenchmarkSelectiveAccuracy: Codable, Hashable {
    let threshold: Double
    let retained: Int
    let accuracy: Double?
}

struct FusedBenchmarkCalibration: Codable, Hashable {
    let evaluatedCases: Int
    let expectedCalibrationError: Double?
    let selectiveAccuracy: [FusedBenchmarkSelectiveAccuracy]
    let fragilePasses: [String]
    let confidentFailures: [String]

    static func calculate(from results: [FusedBenchmarkCaseResult]) -> FusedBenchmarkCalibration {
        let labeled = results.compactMap { result -> (FusedBenchmarkCaseResult, Double, Bool)? in
            guard let passed = result.passed,
                  let mean = result.logprobs?.summary.meanRawLogprob else {
                return nil
            }
            return (result, min(max(exp(mean), 0), 1), passed)
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
            let meanConfidence = members.map { $0.1 }.reduce(0, +) / Double(members.count)
            let accuracy = Double(members.filter { $0.2 }.count) / Double(members.count)
            weightedError += abs(meanConfidence - accuracy) * Double(members.count)
        }
        let thresholds = [0.25, 0.5, 0.75, 0.9]
        return FusedBenchmarkCalibration(
            evaluatedCases: labeled.count,
            expectedCalibrationError: labeled.isEmpty ? nil : weightedError / Double(labeled.count),
            selectiveAccuracy: thresholds.map { threshold in
                let retained = labeled.filter { $0.1 >= threshold }
                return FusedBenchmarkSelectiveAccuracy(
                    threshold: threshold,
                    retained: retained.count,
                    accuracy: retained.isEmpty
                        ? nil
                        : Double(retained.filter { $0.2 }.count) / Double(retained.count)
                )
            },
            fragilePasses: labeled.filter { $0.2 && $0.1 < 0.25 }.map { $0.0.caseID },
            confidentFailures: labeled.filter { !$0.2 && $0.1 >= 0.75 }.map { $0.0.caseID }
        )
    }
}

struct FusedBenchmarkCaseResult: Codable, Hashable {
    let lane: String
    let model: String
    let profile: String
    let trial: Int
    let caseID: String
    let provenance: FusedBenchmarkProvenance
    let passed: Bool?
    let score: Double?
    let generationSeconds: Double?
    let executionSeconds: Double?
    let tokensGenerated: Int?
    let logprobs: FusedBenchmarkLogprobMetrics?
    let acceleration: ChatAccelerationDiagnostics?
    let response: String?
    let error: String?
}

enum FusedBenchmarkHash {
    static func sha256(_ string: String) -> String {
        sha256(Data(string.utf8))
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

enum FusedBenchmarkError: LocalizedError {
    case invalidManifest(String)
    case invalidExternalFixture(String)

    var errorDescription: String? {
        switch self {
        case .invalidManifest(let detail):
            return "Invalid fused benchmark manifest: \(detail)."
        case .invalidExternalFixture(let detail):
            return "Invalid fused benchmark fixture: \(detail)."
        }
    }
}
