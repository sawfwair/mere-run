import ArgumentParser
import Foundation
import MereRunCore

enum FusedBenchmarkPerformanceLane: String, CaseIterable, ExpressibleByArgument, Codable {
    case none
    case native
}

struct ModelBenchmarkFused: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "fused",
        abstract: "Run the versioned Mere Lite or Mere Comprehensive fused quality suite."
    )

    @Option(name: [.long], help: "Suite size: lite or comprehensive.")
    var suite: FusedBenchmarkSuiteSelection = .lite

    @Option(name: [.long], help: "Comma-separated model ids.")
    var models: String?

    @Option(name: [.long], help: "Override the bundled versioned suite manifest.")
    var manifest: String?

    @Option(
        name: [.long],
        help: "Comma-separated hash-verified normalized JSONL fixture files for reference-only sources."
    )
    var externalCases: String?

    @Option(name: [.long], help: "Comma-separated fused case ids.")
    var cases: String?

    @Option(name: [.long], help: "Comma-separated capability tags; all selected tags must match.")
    var capabilities: String?

    @Option(name: [.long], help: "Override the suite's repeated sampled trial count.")
    var trials: Int?

    @Option(name: [.long], help: "Override the per-case generation budget.")
    var maxTokens: Int?

    @Option(name: [.long], help: "Maximum context tokens passed to the runtime.")
    var contextSize: Int = 32_768

    @Option(name: [.long], help: "Logprob capture level for the quality lane.")
    var logprobs: FusedBenchmarkLogprobMode = .summary

    @Option(name: [.customLong("top-logprobs")], help: "Top candidates per token when --logprobs top is selected.")
    var topLogprobs: Int = 5

    @Option(name: [.long], help: "Optional separate performance lane with logprob capture disabled.")
    var performanceLane: FusedBenchmarkPerformanceLane = .none

    @Option(name: [.long], help: "Generated-code execution timeout in seconds.")
    var executionTimeout: Double = 5

    @Option(name: [.long], help: "Python executable for code cases.")
    var python: String = "python3"

    @Option(name: [.long], help: "Sandbox backend for generated-code execution.")
    var sandbox: CodeExecutionSandboxMode = .auto

    @Flag(name: [.long], help: "Acknowledge that quality scoring executes generated Python for code cases.")
    var allowCodeExecution: Bool = false

    @Flag(name: [.long], help: "Include visible model responses in the report.")
    var logResponses: Bool = false

    @Flag(name: [.long], help: "Print the complete plan without loading or running models.")
    var dryRun: Bool = false

    @Flag(name: [.long], help: "Emit machine-readable JSON.")
    var json: Bool = false

    func validate() throws {
        guard trials == nil || trials! > 0 else {
            throw ValidationError("--trials must be greater than zero.")
        }
        guard maxTokens == nil || maxTokens! > 0 else {
            throw ValidationError("--max-tokens must be greater than zero.")
        }
        guard contextSize > 0 else {
            throw ValidationError("--context-size must be greater than zero.")
        }
        guard (1...20).contains(topLogprobs) else {
            throw ValidationError("--top-logprobs must be between 1 and 20.")
        }
        guard executionTimeout > 0, executionTimeout.isFinite else {
            throw ValidationError("--execution-timeout must be a positive finite number.")
        }
        let manifest = try loadedManifest()
        let selected = try selectedCases(in: manifest)
        guard !selected.isEmpty else {
            throw ValidationError("The fused case selection is empty.")
        }
        if !dryRun, selected.contains(where: { descriptor in
            descriptor.adapter == .humanEval || descriptor.adapter == .externalCode
        }) {
            guard allowCodeExecution else {
                throw ValidationError(
                    "fused code scoring executes generated Python; re-run with --allow-code-execution."
                )
            }
            try CodeExecutionSandbox.preflight(mode: sandbox)
        }
        _ = try selectedModelIDs()
    }

    func run() async throws {
        let suiteManifest = try loadedManifest()
        let selected = try selectedCases(in: suiteManifest)
        let modelIDs = try selectedModelIDs()
        let fixturePaths = externalCases?.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        }.filter { !$0.isEmpty } ?? []
        let imported = try FusedExternalBenchmarkLoader.load(paths: fixturePaths)
        let resolvedCases = try selected.map {
            try FusedResolvedBenchmarkCase.resolve(
                descriptor: $0,
                manifest: suiteManifest,
                imported: imported
            )
        }
        let repeatedTrials = trials ?? suiteManifest.defaultTrials(for: suite)
        let capture = logprobs.capture(topLogprobs: topLogprobs)
        let plan = FusedBenchmarkPlan(
            manifestID: suiteManifest.id,
            manifestVersion: suiteManifest.version,
            manifestSHA256: try suiteManifest.contentSHA256(),
            suite: suite.rawValue,
            models: modelIDs.map { modelID in
                FusedBenchmarkPlannedModel(
                    id: modelID,
                    profiles: FusedBenchmarkSamplingProfile.nativeProfiles(
                        modelID: modelID,
                        suite: suite
                    )
                )
            },
            trials: repeatedTrials,
            qualityLane: "sampled-final-target; exact-policy; logprobs=\(capture.mode.rawValue)",
            performanceLane: performanceLane.rawValue,
            cases: resolvedCases.map(\.plan),
            importedFixtureFiles: fixturePaths
        )

        if dryRun {
            try printReport(FusedBenchmarkReport(plan: plan, results: []), dryRun: true)
            return
        }

        try MLXBundleSupport.ensureAvailable(quiet: json)
        var results: [FusedBenchmarkCaseResult] = []
        for modelID in modelIDs {
            if !json {
                CLIStderr.write("Fused quality benchmark: \(modelID)\n")
            }
            let defaultEngine = ManagedModelCatalog.apiProfile(for: modelID)?.servingEngine
                ?? .textChatQ36
            let pool = RuntimeModelPool(
                defaultModelID: modelID,
                defaultEngine: defaultEngine,
                startupModelPath: nil,
                lagunaContinuousBatchingEnabled: performanceLane == .native,
                q35ContinuousBatchingEnabled: performanceLane == .native
            )
            let profiles = FusedBenchmarkSamplingProfile.nativeProfiles(
                modelID: modelID,
                suite: suite
            )
            for profile in profiles {
                for trial in 1...repeatedTrials {
                    for benchmarkCase in resolvedCases {
                        results.append(try await runCase(
                            benchmarkCase,
                            modelID: modelID,
                            profile: profile,
                            trial: trial,
                            capture: capture,
                            lane: "quality-final-target",
                            scoreQuality: true,
                            pool: pool
                        ))
                    }
                }
                if performanceLane == .native {
                    for benchmarkCase in resolvedCases {
                        results.append(try await runCase(
                            benchmarkCase,
                            modelID: modelID,
                            profile: profile,
                            trial: 1,
                            capture: .none,
                            lane: "performance-native-runtime",
                            scoreQuality: false,
                            pool: pool
                        ))
                    }
                }
            }
            _ = try? await pool.unloadModel(idOrAlias: modelID)
        }

        try printReport(FusedBenchmarkReport(plan: plan, results: results), dryRun: false)
    }

    private func runCase(
        _ benchmarkCase: FusedResolvedBenchmarkCase,
        modelID: String,
        profile: FusedBenchmarkSamplingProfile,
        trial: Int,
        capture: ChatLogprobCapture,
        lane: String,
        scoreQuality: Bool,
        pool: RuntimeModelPool
    ) async throws -> FusedBenchmarkCaseResult {
        if benchmarkCase.descriptor.adapter.lane == .vision,
           ManagedModelCatalog.apiProfile(for: modelID)?.inputModalities.contains(.image) != true {
            return benchmarkCase.skippedResult(
                model: modelID,
                profile: profile.name,
                trial: trial,
                lane: lane,
                reason: "model catalog profile does not advertise image input"
            )
        }
        guard let request = benchmarkCase.request(
            profile: profile,
            capture: capture,
            maxTokens: maxTokens,
            contextSize: contextSize
        ) else {
            return benchmarkCase.unresolvedResult(
                model: modelID,
                profile: profile.name,
                trial: trial,
                lane: lane
            )
        }

        let openAIRequest = OpenAIChatRequest(
            model: modelID,
            messages: request.messages.map {
                OpenAIChatMessage(
                    role: $0.role.rawValue,
                    content: $0.content,
                    imageURLs: $0.imageUrl.map { [$0] } ?? []
                )
            },
            max_tokens: request.maxTokens,
            stream: false
        )
        let plan: RuntimeChatPlan
        do {
            plan = try await pool.makeChatPlan(
                for: openAIRequest,
                fallbackLoraPath: nil,
                serverContextSize: contextSize
            )
        } catch {
            return benchmarkCase.failedResult(
                model: modelID,
                profile: profile.name,
                trial: trial,
                lane: lane,
                error: String(describing: error)
            )
        }

        let started = Date()
        let response: ChatResponse
        do {
            response = try await plan.lease.chat(request, progressHandler: nil)
            await plan.lease.release()
        } catch {
            await plan.lease.release()
            return benchmarkCase.failedResult(
                model: modelID,
                profile: profile.name,
                trial: trial,
                lane: lane,
                error: String(describing: error),
                generationSeconds: Date().timeIntervalSince(started)
            )
        }
        let generationSeconds = Date().timeIntervalSince(started)
        let visible = ChatReasoningMarkup.splitThinkBlocks(in: response.response).visibleContent
        let scoring: FusedBenchmarkScoring
        if scoreQuality {
            scoring = try benchmarkCase.score(
                response: response,
                visibleResponse: visible,
                allowCodeExecution: allowCodeExecution,
                python: python,
                sandbox: sandbox,
                executionTimeout: executionTimeout
            )
        } else {
            scoring = FusedBenchmarkScoring(passed: nil, score: nil, executionSeconds: nil, error: nil)
        }
        return FusedBenchmarkCaseResult(
            lane: lane,
            model: modelID,
            profile: profile.name,
            trial: trial,
            caseID: benchmarkCase.descriptor.id,
            provenance: benchmarkCase.provenance,
            passed: scoring.passed,
            score: scoring.score,
            generationSeconds: generationSeconds,
            executionSeconds: scoring.executionSeconds,
            tokensGenerated: response.tokensGenerated,
            logprobs: response.logprobs.map(FusedBenchmarkLogprobMetrics.init),
            acceleration: response.acceleration,
            response: logResponses && scoreQuality ? visible : nil,
            error: scoring.error
        )
    }

    private func loadedManifest() throws -> FusedBenchmarkManifest {
        guard let manifest else {
            return try .bundled()
        }
        return try .load(from: URL(fileURLWithPath: manifest).standardizedFileURL)
    }

    private func selectedCases(
        in manifest: FusedBenchmarkManifest
    ) throws -> [FusedBenchmarkCaseDescriptor] {
        var selected = manifest.selectedCases(for: suite)
        if let capabilities {
            let tags = Set(capabilities.split(separator: ",").map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            }.filter { !$0.isEmpty })
            selected = selected.filter { descriptor in
                tags.isSubset(of: Set(descriptor.capabilityTags.map { $0.lowercased() }))
            }
        }
        if let cases {
            let requested = cases.split(separator: ",").map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines)
            }.filter { !$0.isEmpty }
            var byID = Dictionary(uniqueKeysWithValues: selected.map { ($0.id, $0) })
            var ordered: [FusedBenchmarkCaseDescriptor] = []
            for id in requested {
                guard let descriptor = byID.removeValue(forKey: id) else {
                    throw ValidationError("Unknown or filtered fused case id: \(id).")
                }
                ordered.append(descriptor)
            }
            selected = ordered
        }
        return selected
    }

    private func selectedModelIDs() throws -> [String] {
        let defaults = [
            Q35Resources.q38TwentySevenBModelId,
            Q35Resources.q38TwentySevenB4BitModelId,
            NemotronHResources.modelID,
            LagunaResources.xsModelID,
        ]
        let requested = models?.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        } ?? defaults
        var seen: Set<String> = []
        let selected = requested.filter { !$0.isEmpty && seen.insert($0).inserted }
        guard !selected.isEmpty else {
            throw ValidationError("--models must include at least one model id.")
        }
        return selected
    }

    private func printReport(_ report: FusedBenchmarkReport, dryRun: Bool) throws {
        if json {
            print(try report.jsonString())
        } else {
            print(report.renderText(dryRun: dryRun))
        }
    }
}

private struct FusedBenchmarkScoring {
    let passed: Bool?
    let score: Double?
    let executionSeconds: Double?
    let error: String?
}

private enum FusedResolvedBenchmarkPayload {
    case chat(ChatBenchmarkCase)
    case tool(ToolBenchmarkCase)
    case code(CodeBenchmarkTask)
    case external(FusedExternalBenchmarkCase)
    case unresolved(String)
}

private struct FusedResolvedBenchmarkCase {
    let descriptor: FusedBenchmarkCaseDescriptor
    let provenance: FusedBenchmarkProvenance
    let payload: FusedResolvedBenchmarkPayload

    static func resolve(
        descriptor: FusedBenchmarkCaseDescriptor,
        manifest: FusedBenchmarkManifest,
        imported: [String: FusedExternalBenchmarkCase]
    ) throws -> FusedResolvedBenchmarkCase {
        guard let source = manifest.source(id: descriptor.sourceID) else {
            throw FusedBenchmarkError.invalidManifest("unknown source \(descriptor.sourceID)")
        }
        let payload: FusedResolvedBenchmarkPayload
        let contentHash: String?
        switch descriptor.adapter {
        case .mereChat:
            guard let benchmarkCase = ChatBenchmarkCase.mereChatSlice.first(where: {
                $0.caseID == descriptor.sourceCaseID
            }) else {
                throw FusedBenchmarkError.invalidManifest(
                    "missing embedded chat case \(descriptor.sourceCaseID)"
                )
            }
            payload = .chat(benchmarkCase)
            contentHash = FusedBenchmarkHash.sha256(
                try JSONEncoder.sorted.encode(benchmarkCase)
            )
        case .mereTool:
            guard let benchmarkCase = ToolBenchmarkCase.mereToolSlice.first(where: {
                $0.caseID == descriptor.sourceCaseID
            }) else {
                throw FusedBenchmarkError.invalidManifest(
                    "missing embedded tool case \(descriptor.sourceCaseID)"
                )
            }
            payload = .tool(benchmarkCase)
            contentHash = FusedBenchmarkHash.sha256(
                try JSONEncoder.sorted.encode(benchmarkCase)
            )
        case .mereVision:
            let benchmarkCase = try FusedBenchmarkVisionFixtures.resolve(
                descriptor: descriptor,
                source: source
            )
            payload = .external(benchmarkCase)
            contentHash = benchmarkCase.contentSHA256
        case .humanEval:
            guard let benchmarkCase = CodeBenchmarkTask.humanEvalSlice.first(where: {
                $0.taskID == descriptor.sourceCaseID
            }) else {
                throw FusedBenchmarkError.invalidManifest(
                    "missing embedded HumanEval case \(descriptor.sourceCaseID)"
                )
            }
            payload = .code(benchmarkCase)
            contentHash = FusedBenchmarkHash.sha256(
                try JSONEncoder.sorted.encode(benchmarkCase)
            )
        case .externalChat, .externalCode, .externalTool, .externalVision:
            if let importedCase = imported[descriptor.id] {
                try FusedExternalBenchmarkContract.validate(
                    importedCase,
                    for: descriptor,
                    source: source
                )
                payload = .external(importedCase)
                contentHash = importedCase.contentSHA256
            } else {
                payload = .unresolved(
                    "reference-only case; provide a normalized hash-verified fixture with --external-cases"
                )
                contentHash = nil
            }
        }
        let imageSHA256: String?
        if case .external(let externalCase) = payload {
            imageSHA256 = externalCase.imageSHA256
        } else {
            imageSHA256 = nil
        }
        return FusedResolvedBenchmarkCase(
            descriptor: descriptor,
            provenance: FusedBenchmarkProvenance(
                descriptor: descriptor,
                source: source,
                contentSHA256: contentHash,
                imageSHA256: imageSHA256,
                sourceVersion: imported[descriptor.id]?.sourceVersion,
                originalID: imported[descriptor.id]?.originalID
            ),
            payload: payload
        )
    }

    var plan: FusedBenchmarkPlannedCase {
        let resolved: Bool
        let reason: String?
        if case .unresolved(let detail) = payload {
            resolved = false
            reason = detail
        } else {
            resolved = true
            reason = nil
        }
        return FusedBenchmarkPlannedCase(
            id: descriptor.id,
            adapter: descriptor.adapter.rawValue,
            lane: descriptor.adapter.lane.rawValue,
            resolved: resolved,
            unresolvedReason: reason,
            provenance: provenance
        )
    }

    func request(
        profile: FusedBenchmarkSamplingProfile,
        capture: ChatLogprobCapture,
        maxTokens: Int?,
        contextSize: Int
    ) -> ChatRequest? {
        let messages: [ChatMessage]
        let tools: [ToolDefinition]?
        let defaultBudget: Int
        let logprobRegionHint: ChatLogprobRegion?
        switch payload {
        case .chat(let benchmarkCase):
            messages = [
                ChatMessage(role: .system, content: Self.chatSystemPrompt),
                ChatMessage(role: .user, content: benchmarkCase.promptText),
            ]
            tools = nil
            defaultBudget = 256
            logprobRegionHint = .visible
        case .tool(let benchmarkCase):
            messages = [
                ChatMessage(role: .system, content: Self.toolSystemPrompt),
                ChatMessage(role: .user, content: benchmarkCase.promptText),
            ]
            tools = benchmarkCase.tools
            defaultBudget = 256
            logprobRegionHint = nil
        case .code(let benchmarkCase):
            messages = [
                ChatMessage(role: .system, content: Self.codeSystemPrompt),
                ChatMessage(role: .user, content: benchmarkCase.promptText),
            ]
            tools = nil
            defaultBudget = 1_024
            logprobRegionHint = .code
        case .external(let benchmarkCase):
            messages = benchmarkCase.messages
            tools = benchmarkCase.tools
            defaultBudget = benchmarkCase.kind == .code ? 1_024 : 256
            logprobRegionHint = benchmarkCase.kind == .code ? .code : .visible
        case .unresolved:
            return nil
        }
        return ChatRequest(
            messages: messages,
            maxTokens: maxTokens ?? defaultBudget,
            temperature: profile.temperature,
            topP: profile.topP,
            topK: profile.topK,
            minP: profile.minP,
            reasoningEffort: profile.reasoningEffort,
            showThinking: profile.showThinking,
            tools: tools,
            stopOnEOS: true,
            stopSequences: TextGenerationStopSequences.defaultRenderedChatStops,
            maxContextTokens: contextSize,
            logprobCapture: capture,
            logprobRegionHint: logprobRegionHint
        )
    }

    func score(
        response: ChatResponse,
        visibleResponse: String,
        allowCodeExecution: Bool,
        python: String,
        sandbox: CodeExecutionSandboxMode,
        executionTimeout: Double
    ) throws -> FusedBenchmarkScoring {
        switch payload {
        case .chat(let benchmarkCase):
            let evaluation = benchmarkCase.evaluate(visibleResponse)
            return FusedBenchmarkScoring(
                passed: evaluation.passed,
                score: Double(evaluation.passedChecks) / Double(max(1, evaluation.totalChecks)),
                executionSeconds: nil,
                error: evaluation.failedChecks.isEmpty
                    ? nil
                    : evaluation.failedChecks.joined(separator: "; ")
            )
        case .tool(let benchmarkCase):
            let calls = (response.toolCalls ?? []).map(ToolBenchmarkObservedCall.init(call:))
            let evaluation = benchmarkCase.expectation.evaluate(calls)
            return FusedBenchmarkScoring(
                passed: evaluation.passed,
                score: Double(evaluation.passedChecks) / Double(max(1, evaluation.totalChecks)),
                executionSeconds: nil,
                error: evaluation.failedChecks.isEmpty
                    ? nil
                    : evaluation.failedChecks.joined(separator: "; ")
            )
        case .code(let benchmarkCase):
            guard allowCodeExecution else {
                return FusedBenchmarkScoring(
                    passed: nil,
                    score: nil,
                    executionSeconds: nil,
                    error: "code execution was not authorized"
                )
            }
            let candidate = benchmarkCase.candidateProgram(from: visibleResponse)
            let execution = try CodeExecutionSandbox.runPython(
                program: benchmarkCase.testProgram(candidateProgram: candidate),
                python: python,
                mode: sandbox,
                timeout: executionTimeout
            )
            return FusedBenchmarkScoring(
                passed: execution.passed,
                score: execution.passed ? 1 : 0,
                executionSeconds: execution.seconds,
                error: execution.errorSummary
            )
        case .external(let benchmarkCase):
            return try scoreExternal(
                benchmarkCase,
                response: response,
                visibleResponse: visibleResponse,
                allowCodeExecution: allowCodeExecution,
                python: python,
                sandbox: sandbox,
                executionTimeout: executionTimeout
            )
        case .unresolved(let detail):
            return FusedBenchmarkScoring(
                passed: nil,
                score: nil,
                executionSeconds: nil,
                error: detail
            )
        }
    }

    private func scoreExternal(
        _ benchmarkCase: FusedExternalBenchmarkCase,
        response: ChatResponse,
        visibleResponse: String,
        allowCodeExecution: Bool,
        python: String,
        sandbox: CodeExecutionSandboxMode,
        executionTimeout: Double
    ) throws -> FusedBenchmarkScoring {
        switch benchmarkCase.kind {
        case .chat, .vision:
            let normalized = visibleResponse.lowercased()
            let required = benchmarkCase.requiredPhrases ?? []
            let forbidden = benchmarkCase.forbiddenPhrases ?? []
            let passedChecks = required.filter { normalized.contains($0.lowercased()) }.count
                + forbidden.filter { !normalized.contains($0.lowercased()) }.count
            let total = required.count + forbidden.count
            return FusedBenchmarkScoring(
                passed: total > 0 ? passedChecks == total : nil,
                score: total > 0 ? Double(passedChecks) / Double(total) : nil,
                executionSeconds: nil,
                error: total > 0 && passedChecks != total ? "external text checks failed" : nil
            )
        case .tool:
            let calls = response.toolCalls ?? []
            let match = calls.first { $0.name == benchmarkCase.expectedToolName }
            let argumentChecks = benchmarkCase.expectedArguments ?? [:]
            let argumentsPassed = match.map { call in
                argumentChecks.allSatisfy { call.arguments[$0.key] == $0.value }
            } ?? false
            let passed = match != nil && argumentsPassed
            return FusedBenchmarkScoring(
                passed: passed,
                score: passed ? 1 : 0,
                executionSeconds: nil,
                error: passed ? nil : "external tool expectation failed"
            )
        case .code:
            guard allowCodeExecution,
                  let entryPoint = benchmarkCase.entryPoint,
                  let tests = benchmarkCase.tests else {
                return FusedBenchmarkScoring(
                    passed: nil,
                    score: nil,
                    executionSeconds: nil,
                    error: "external code fixture is incomplete or execution was not authorized"
                )
            }
            let code = Self.extractCode(visibleResponse, entryPoint: entryPoint)
            let program = "\(code)\n\n\(tests)\n\ncheck(\(entryPoint))\n"
            let execution = try CodeExecutionSandbox.runPython(
                program: program,
                python: python,
                mode: sandbox,
                timeout: executionTimeout
            )
            return FusedBenchmarkScoring(
                passed: execution.passed,
                score: execution.passed ? 1 : 0,
                executionSeconds: execution.seconds,
                error: execution.errorSummary
            )
        }
    }

    func unresolvedResult(
        model: String,
        profile: String,
        trial: Int,
        lane: String
    ) -> FusedBenchmarkCaseResult {
        let detail: String
        if case .unresolved(let reason) = payload {
            detail = reason
        } else {
            detail = "case could not be resolved"
        }
        return FusedBenchmarkCaseResult(
            lane: lane,
            model: model,
            profile: profile,
            trial: trial,
            caseID: descriptor.id,
            provenance: provenance,
            passed: nil,
            score: nil,
            generationSeconds: nil,
            executionSeconds: nil,
            tokensGenerated: nil,
            logprobs: nil,
            acceleration: nil,
            response: nil,
            error: detail
        )
    }

    func skippedResult(
        model: String,
        profile: String,
        trial: Int,
        lane: String,
        reason: String
    ) -> FusedBenchmarkCaseResult {
        FusedBenchmarkCaseResult(
            lane: lane,
            model: model,
            profile: profile,
            trial: trial,
            caseID: descriptor.id,
            provenance: provenance,
            passed: nil,
            score: nil,
            generationSeconds: nil,
            executionSeconds: nil,
            tokensGenerated: nil,
            logprobs: nil,
            acceleration: nil,
            response: nil,
            error: reason
        )
    }

    func failedResult(
        model: String,
        profile: String,
        trial: Int,
        lane: String,
        error: String,
        generationSeconds: Double? = nil
    ) -> FusedBenchmarkCaseResult {
        FusedBenchmarkCaseResult(
            lane: lane,
            model: model,
            profile: profile,
            trial: trial,
            caseID: descriptor.id,
            provenance: provenance,
            passed: false,
            score: 0,
            generationSeconds: generationSeconds,
            executionSeconds: nil,
            tokensGenerated: nil,
            logprobs: nil,
            acceleration: nil,
            response: nil,
            error: error
        )
    }

    private static func extractCode(_ response: String, entryPoint: String) -> String {
        var text = response
        if let opening = text.range(of: "```") {
            let afterOpening = text[opening.upperBound...]
            let start = afterOpening.firstIndex(of: "\n").map { text.index(after: $0) }
                ?? opening.upperBound
            let tail = text[start...]
            text = tail.range(of: "```").map { String(tail[..<$0.lowerBound]) }
                ?? String(tail)
        }
        if let definition = text.range(of: "def \(entryPoint)") {
            text = String(text[definition.lowerBound...])
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static let chatSystemPrompt = """
    Answer only from the supplied evidence. Preserve exact identifiers and dates.
    Report missing or conflicting evidence explicitly. Never claim a local action completed unless execution evidence is present.
    """

    private static let toolSystemPrompt = """
    Use the provided local tool schema exactly when a lookup or action is required.
    Do not call a tool when the prompt already contains the answer. Never invent tool names or arguments.
    """

    private static let codeSystemPrompt = """
    Complete the Python task. Return only valid Python implementation code without Markdown or prose.
    """
}

private extension JSONEncoder {
    static var sorted: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }
}

private struct FusedBenchmarkPlannedModel: Codable {
    let id: String
    let profiles: [FusedBenchmarkSamplingProfile]
}

private struct FusedBenchmarkPlannedCase: Codable {
    let id: String
    let adapter: String
    let lane: String
    let resolved: Bool
    let unresolvedReason: String?
    let provenance: FusedBenchmarkProvenance
}

private struct FusedBenchmarkPlan: Codable {
    let manifestID: String
    let manifestVersion: String
    let manifestSHA256: String
    let suite: String
    let models: [FusedBenchmarkPlannedModel]
    let trials: Int
    let qualityLane: String
    let performanceLane: String
    let cases: [FusedBenchmarkPlannedCase]
    let importedFixtureFiles: [String]
}

private struct FusedBenchmarkBreakdown: Codable {
    let model: String
    let profile: String
    let key: String
    let evaluated: Int
    let passes: Int
    let passRate: Double?
    let meanScore: Double?
}

private struct FusedBenchmarkCalibrationGroup: Codable {
    let model: String
    let profile: String
    let calibration: FusedBenchmarkCalibration
}

private struct FusedBenchmarkReport: Codable {
    let plan: FusedBenchmarkPlan
    let results: [FusedBenchmarkCaseResult]
    let calibrationByModelProfile: [FusedBenchmarkCalibrationGroup]
    let laneBreakdown: [FusedBenchmarkBreakdown]
    let capabilityBreakdown: [FusedBenchmarkBreakdown]
    let sourceBreakdown: [FusedBenchmarkBreakdown]

    init(plan: FusedBenchmarkPlan, results: [FusedBenchmarkCaseResult]) {
        self.plan = plan
        self.results = results
        let quality = results.filter { $0.lane == "quality-final-target" }
        self.calibrationByModelProfile = Self.groups(in: quality).map { group in
            FusedBenchmarkCalibrationGroup(
                model: group.model,
                profile: group.profile,
                calibration: FusedBenchmarkCalibration.calculate(from: group.results)
            )
        }
        self.laneBreakdown = Self.breakdown(
            results: quality,
            keys: Dictionary(uniqueKeysWithValues: plan.cases.map {
                ($0.id, [$0.lane])
            })
        )
        self.capabilityBreakdown = Self.breakdown(
            results: quality,
            keys: Dictionary(uniqueKeysWithValues: plan.cases.map {
                ($0.id, $0.provenance.capabilityTags)
            })
        )
        self.sourceBreakdown = Self.breakdown(
            results: quality,
            keys: Dictionary(uniqueKeysWithValues: plan.cases.map {
                ($0.id, [$0.provenance.source])
            })
        )
    }

    func jsonString() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }

    func renderText(dryRun: Bool) -> String {
        let unresolved = plan.cases.filter { !$0.resolved }
        let laneCounts = Dictionary(grouping: plan.cases, by: \.lane)
            .mapValues(\.count)
        let laneSummary = FusedBenchmarkLane.allCases.map { lane in
            "\(lane.rawValue) \(laneCounts[lane.rawValue, default: 0])"
        }.joined(separator: ", ")
        var lines = [
            "Mere fused benchmark \(plan.manifestID)@\(plan.manifestVersion)",
            "manifest sha256: \(plan.manifestSHA256)",
            "suite: \(plan.suite)",
            "models: \(plan.models.map(\.id).joined(separator: ", "))",
            "cases: \(plan.cases.count) (resolved \(plan.cases.count - unresolved.count), reference-only \(unresolved.count))",
            "lanes: \(laneSummary)",
            "trials: \(plan.trials)",
            "quality lane: \(plan.qualityLane)",
            "performance lane: \(plan.performanceLane)",
        ]
        if dryRun {
            lines.append("dry run: no models loaded, no model inference, and no generated code executed")
            lines.append("")
            for model in plan.models {
                lines.append("\(model.id): \(model.profiles.map(\.name).joined(separator: ", "))")
            }
            if !unresolved.isEmpty {
                lines.append("")
                lines.append("reference-only cases awaiting imported fixtures:")
                lines.append(contentsOf: unresolved.map { "- \($0.id)" })
            }
            return lines.joined(separator: "\n")
        }

        let quality = results.filter { $0.lane == "quality-final-target" && $0.passed != nil }
        let passes = quality.filter { $0.passed == true }.count
        lines.append("quality: \(passes)/\(quality.count) case-trials passed")
        lines.append("")
        lines.append("calibration by model/profile:")
        lines.append(contentsOf: calibrationByModelProfile.map { group in
            let ece = group.calibration.expectedCalibrationError.map {
                String(format: "%.4f", $0)
            } ?? "n/a"
            return "- \(group.model) [\(group.profile)]: ECE \(ece)"
        })
        lines.append("")
        lines.append("lane breakdown:")
        lines.append(contentsOf: laneBreakdown.map { item in
            let rate = item.passRate.map { String(format: "%.1f%%", $0 * 100) } ?? "n/a"
            return "- \(item.model) [\(item.profile)] \(item.key): \(item.passes)/\(item.evaluated) (\(rate))"
        })
        lines.append("")
        lines.append("capability breakdown:")
        lines.append(contentsOf: capabilityBreakdown.map { item in
            let rate = item.passRate.map { String(format: "%.1f%%", $0 * 100) } ?? "n/a"
            return "- \(item.model) [\(item.profile)] \(item.key): \(item.passes)/\(item.evaluated) (\(rate))"
        })
        lines.append("")
        lines.append("source breakdown:")
        lines.append(contentsOf: sourceBreakdown.map { item in
            let rate = item.passRate.map { String(format: "%.1f%%", $0 * 100) } ?? "n/a"
            return "- \(item.model) [\(item.profile)] \(item.key): \(item.passes)/\(item.evaluated) (\(rate))"
        })
        return lines.joined(separator: "\n")
    }

    private static func breakdown(
        results: [FusedBenchmarkCaseResult],
        keys: [String: [String]]
    ) -> [FusedBenchmarkBreakdown] {
        let allKeys = Set(keys.values.flatMap { $0 }).sorted()
        return groups(in: results).flatMap { group in
            allKeys.map { key in
                let members = group.results.filter {
                    keys[$0.caseID]?.contains(key) == true && $0.passed != nil
                }
                let passes = members.filter { $0.passed == true }.count
                let scores = members.compactMap(\.score)
                return FusedBenchmarkBreakdown(
                    model: group.model,
                    profile: group.profile,
                    key: key,
                    evaluated: members.count,
                    passes: passes,
                    passRate: members.isEmpty ? nil : Double(passes) / Double(members.count),
                    meanScore: scores.isEmpty ? nil : scores.reduce(0, +) / Double(scores.count)
                )
            }
        }
    }

    private static func groups(
        in results: [FusedBenchmarkCaseResult]
    ) -> [(model: String, profile: String, results: [FusedBenchmarkCaseResult])] {
        let keys = Set(results.map { "\($0.model)\u{0}\($0.profile)" }).sorted()
        return keys.compactMap { key in
            let pieces = key.split(separator: "\u{0}", omittingEmptySubsequences: false)
            guard pieces.count == 2 else { return nil }
            let model = String(pieces[0])
            let profile = String(pieces[1])
            return (
                model: model,
                profile: profile,
                results: results.filter { $0.model == model && $0.profile == profile }
            )
        }
    }
}
