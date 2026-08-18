import ArgumentParser
import Foundation
import MereRunCore
import MereRunRelayKit

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

    @Option(
        name: [.long],
        help: "Atomically persist the run plan and every completed case-trial as JSON."
    )
    var checkpoint: String?

    @Flag(name: [.long], help: "Resume a checkpoint whose complete run plan exactly matches this invocation.")
    var resume: Bool = false

    @Option(
        name: [.customLong("case-trial-limit")],
        help: "Stop cleanly after this many new case-trials; requires --checkpoint."
    )
    var caseTrialLimit: Int?

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
        guard !resume || checkpoint != nil else {
            throw ValidationError("--resume requires --checkpoint.")
        }
        guard caseTrialLimit == nil || caseTrialLimit! > 0 else {
            throw ValidationError("--case-trial-limit must be greater than zero.")
        }
        guard caseTrialLimit == nil || checkpoint != nil else {
            throw ValidationError("--case-trial-limit requires --checkpoint.")
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
        let runner = try FusedBenchmarkRunnerIdentity.current()
        let host = FusedBenchmarkHost.current()
        let includesExecutableCode = selected.contains { descriptor in
            descriptor.adapter == .humanEval || descriptor.adapter == .externalCode
        }
        let codeRuntime = try includesExecutableCode && allowCodeExecution
            ? FusedBenchmarkCodeRuntimeIdentity.resolve(python: python, sandbox: sandbox)
            : nil
        let plan = FusedBenchmarkPlan(
            runnerVersion: runner.version,
            runnerExecutableByteCount: runner.executableByteCount,
            runnerExecutableSHA256: runner.executableSHA256,
            host: host,
            manifestID: suiteManifest.id,
            manifestVersion: suiteManifest.version,
            manifestSHA256: try suiteManifest.contentSHA256(),
            suite: suite.rawValue,
            models: try modelIDs.map { modelID in
                try FusedBenchmarkPlannedModel.resolve(
                    id: modelID,
                    suite: suite
                )
            },
            trials: repeatedTrials,
            qualityLane: "sampled-final-target; exact-policy; logprobs=\(capture.mode.rawValue)",
            performanceLane: performanceLane.rawValue,
            settings: FusedBenchmarkRunSettings(
                maxTokensOverride: maxTokens,
                contextSize: contextSize,
                logprobs: logprobs.rawValue,
                topLogprobs: topLogprobs,
                executionTimeout: executionTimeout,
                pythonExecutable: python,
                pythonResolvedPath: codeRuntime?.pythonResolvedPath,
                pythonExecutableSHA256: codeRuntime?.pythonExecutableSHA256,
                sandbox: sandbox.rawValue,
                sandboxBackend: codeRuntime?.sandboxBackend,
                allowCodeExecution: allowCodeExecution,
                logResponses: logResponses
            ),
            cases: resolvedCases.map(\.plan),
            importedFixtureFiles: fixturePaths
        )

        let checkpointStore = checkpoint.map {
            FusedBenchmarkCheckpointStore(
                url: URL(fileURLWithPath: $0).standardizedFileURL
            )
        }
        var results: [FusedBenchmarkCaseResult]
        if let checkpointStore {
            if resume {
                results = try checkpointStore.loadResults(validating: plan)
            } else {
                try checkpointStore.initialize(plan: plan)
                results = []
            }
        } else {
            results = []
        }

        if dryRun {
            try printReport(FusedBenchmarkReport(plan: plan, results: results), dryRun: true)
            return
        }

        var completedKeys = Set(results.map(\.key))
        let expectedKeys = plan.expectedResultKeys
        let pendingKeys = expectedKeys.subtracting(completedKeys)
        if pendingKeys.isEmpty {
            try printReport(FusedBenchmarkReport(plan: plan, results: results), dryRun: false)
            return
        }
        let pendingModelIDs = Set(pendingKeys.map(\.model))
        let unavailable = plan.models.filter {
            pendingModelIDs.contains($0.id) && !$0.installed
        }.map(\.id)
        guard unavailable.isEmpty else {
            throw ValidationError(
                "Fused benchmarks never download models. Mount or install these models under the selected "
                    + "--models-root before running: \(unavailable.joined(separator: ", "))."
            )
        }

        try MLXBundleSupport.ensureAvailable(quiet: json)
        var completedThisInvocation = 0
        var reachedCaseTrialLimit = false
        for modelID in modelIDs {
            if reachedCaseTrialLimit {
                break
            }
            let modelHasPendingResults = pendingKeys.contains { $0.model == modelID }
            if !modelHasPendingResults {
                continue
            }
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
            do {
                profileLoop: for profile in profiles {
                    for trial in 1...repeatedTrials {
                        for benchmarkCase in resolvedCases {
                            let key = FusedBenchmarkResultKey(
                                lane: "quality-final-target",
                                model: modelID,
                                profile: profile.name,
                                trial: trial,
                                caseID: benchmarkCase.descriptor.id
                            )
                            if completedKeys.contains(key) {
                                continue
                            }
                            let result = try await runCase(
                                benchmarkCase,
                                modelID: modelID,
                                profile: profile,
                                trial: trial,
                                capture: capture,
                                lane: "quality-final-target",
                                scoreQuality: true,
                                pool: pool
                            )
                            results.append(result)
                            completedKeys.insert(result.key)
                            completedThisInvocation += 1
                            try checkpointStore?.write(plan: plan, results: results)
                            if let checkpointStore {
                                CLIStderr.write(
                                    "Checkpointed \(results.count)/\(expectedKeys.count) case-trials at "
                                        + "\(checkpointStore.url.path).\n"
                                )
                            }
                            if let caseTrialLimit,
                               completedThisInvocation >= caseTrialLimit {
                                reachedCaseTrialLimit = true
                                break profileLoop
                            }
                        }
                    }
                    if performanceLane == .native {
                        for benchmarkCase in resolvedCases {
                            let key = FusedBenchmarkResultKey(
                                lane: "performance-native-runtime",
                                model: modelID,
                                profile: profile.name,
                                trial: 1,
                                caseID: benchmarkCase.descriptor.id
                            )
                            if completedKeys.contains(key) {
                                continue
                            }
                            let result = try await runCase(
                                benchmarkCase,
                                modelID: modelID,
                                profile: profile,
                                trial: 1,
                                capture: .none,
                                lane: "performance-native-runtime",
                                scoreQuality: false,
                                pool: pool
                            )
                            results.append(result)
                            completedKeys.insert(result.key)
                            completedThisInvocation += 1
                            try checkpointStore?.write(plan: plan, results: results)
                            if let checkpointStore {
                                CLIStderr.write(
                                    "Checkpointed \(results.count)/\(expectedKeys.count) case-trials at "
                                        + "\(checkpointStore.url.path).\n"
                                )
                            }
                            if let caseTrialLimit,
                               completedThisInvocation >= caseTrialLimit {
                                reachedCaseTrialLimit = true
                                break profileLoop
                            }
                        }
                    }
                }
            } catch {
                _ = try? await pool.unloadModel(idOrAlias: modelID)
                throw error
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
            throw FusedBenchmarkRuntimeError.modelPreparation(
                model: modelID,
                profile: profile.name,
                trial: trial,
                caseID: benchmarkCase.descriptor.id,
                detail: String(describing: error)
            )
        }

        let started = Date()
        let response: ChatResponse
        do {
            response = try await plan.lease.chat(request, progressHandler: nil)
            await plan.lease.release()
        } catch {
            await plan.lease.release()
            throw FusedBenchmarkRuntimeError.generation(
                model: modelID,
                profile: profile.name,
                trial: trial,
                caseID: benchmarkCase.descriptor.id,
                detail: String(describing: error)
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

struct FusedBenchmarkScoring {
    let passed: Bool?
    let score: Double?
    let executionSeconds: Double?
    let error: String?
}

enum FusedBenchmarkRuntimeError: LocalizedError {
    case modelPreparation(model: String, profile: String, trial: Int, caseID: String, detail: String)
    case generation(model: String, profile: String, trial: Int, caseID: String, detail: String)

    var errorDescription: String? {
        switch self {
        case .modelPreparation(let model, let profile, let trial, let caseID, let detail):
            return Self.message(
                stage: "model preparation",
                model: model,
                profile: profile,
                trial: trial,
                caseID: caseID,
                detail: detail
            )
        case .generation(let model, let profile, let trial, let caseID, let detail):
            return Self.message(
                stage: "generation",
                model: model,
                profile: profile,
                trial: trial,
                caseID: caseID,
                detail: detail
            )
        }
    }

    private static func message(
        stage: String,
        model: String,
        profile: String,
        trial: Int,
        caseID: String,
        detail: String
    ) -> String {
        "Fused benchmark \(stage) failed for \(model) [\(profile)] trial \(trial), case \(caseID); "
            + "the row remains pending: \(detail)"
    }
}

enum FusedResolvedBenchmarkPayload {
    case chat(ChatBenchmarkCase)
    case tool(ToolBenchmarkCase)
    case code(CodeBenchmarkTask)
    case external(FusedExternalBenchmarkCase)
    case unresolved(String)
}

struct FusedResolvedBenchmarkCase {
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
                sourceRevision: imported[descriptor.id]?.sourceRevision,
                originalID: imported[descriptor.id]?.originalID
            ),
            payload: payload
        )
    }

    fileprivate var plan: FusedBenchmarkPlannedCase {
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
            defaultBudget = benchmarkCase.generationBudget
                ?? (benchmarkCase.kind == .code ? 1_024 : 256)
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
            let metric = benchmarkCase.textMetric ?? .phraseChecks
            if metric != .phraseChecks {
                let references = benchmarkCase.referenceAnswers ?? []
                let score = references.map { reference in
                    Self.textScore(metric: metric, prediction: visibleResponse, reference: reference)
                }.max() ?? 0
                let threshold = benchmarkCase.passThreshold ?? 1
                let passed = score >= threshold
                return FusedBenchmarkScoring(
                    passed: passed,
                    score: score,
                    executionSeconds: nil,
                    error: passed ? nil : "external text score was below \(threshold)"
                )
            }
            let phraseResult = Self.phraseChecks(
                response: visibleResponse,
                required: benchmarkCase.requiredPhrases ?? [],
                forbidden: benchmarkCase.forbiddenPhrases ?? []
            )
            return FusedBenchmarkScoring(
                passed: phraseResult.total > 0 ? phraseResult.passed == phraseResult.total : nil,
                score: phraseResult.total > 0
                    ? Double(phraseResult.passed) / Double(phraseResult.total)
                    : nil,
                executionSeconds: nil,
                error: phraseResult.total > 0 && phraseResult.passed != phraseResult.total
                    ? "external text checks failed"
                    : nil
            )
        case .tool:
            let calls = response.toolCalls ?? []
            let phraseResult = Self.phraseChecks(
                response: visibleResponse,
                required: benchmarkCase.requiredPhrases ?? [],
                forbidden: benchmarkCase.forbiddenPhrases ?? []
            )
            let phrasesPassed = phraseResult.passed == phraseResult.total
            let expectedCalls: [FusedExternalToolExpectation]
            if let declared = benchmarkCase.expectedToolCalls {
                expectedCalls = declared
            } else if let name = benchmarkCase.expectedToolName {
                expectedCalls = [FusedExternalToolExpectation(
                    name: name,
                    arguments: (benchmarkCase.expectedArguments ?? [:]).mapValues { [$0] }
                )]
            } else {
                expectedCalls = []
            }
            let callsPassed = benchmarkCase.expectsNoToolCalls == true
                ? calls.isEmpty
                : Self.toolCallsMatch(expected: expectedCalls, observed: calls)
            let passed = callsPassed && phrasesPassed
            return FusedBenchmarkScoring(
                passed: passed,
                score: passed ? 1 : 0,
                executionSeconds: nil,
                error: passed ? nil : "external tool expectation failed"
            )
        case .code:
            guard allowCodeExecution else {
                return FusedBenchmarkScoring(
                    passed: nil,
                    score: nil,
                    executionSeconds: nil,
                    error: "external code execution was not authorized"
                )
            }
            let code = Self.extractCode(visibleResponse)
            let evaluation = benchmarkCase.codeEvaluation ?? .function
            let program: String
            switch evaluation {
            case .function:
                guard let entryPoint = benchmarkCase.entryPoint,
                      let tests = benchmarkCase.tests else {
                    return FusedBenchmarkScoring(
                        passed: nil,
                        score: nil,
                        executionSeconds: nil,
                        error: "external function fixture is incomplete"
                    )
                }
                program = "\(code)\n\n_mere_candidate = \(entryPoint)\n\n\(tests)\n\ncheck(_mere_candidate)\n"
            case .stdin, .functional:
                guard let codeTests = benchmarkCase.codeTests else {
                    return FusedBenchmarkScoring(
                        passed: nil,
                        score: nil,
                        executionSeconds: nil,
                        error: "external program fixture is incomplete"
                    )
                }
                program = try Self.programHarness(
                    candidate: code,
                    evaluation: evaluation,
                    entryPoint: benchmarkCase.entryPoint,
                    tests: codeTests
                )
            }
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

    private static func phraseChecks(
        response: String,
        required: [String],
        forbidden: [String]
    ) -> (passed: Int, total: Int) {
        let normalized = response.lowercased()
        let passed = required.filter { normalized.contains($0.lowercased()) }.count
            + forbidden.filter { !normalized.contains($0.lowercased()) }.count
        return (passed, required.count + forbidden.count)
    }

    private static func textScore(
        metric: FusedExternalTextMetric,
        prediction: String,
        reference: String
    ) -> Double {
        switch metric {
        case .phraseChecks:
            return prediction.localizedCaseInsensitiveContains(reference) ? 1 : 0
        case .qaF1:
            let predicted = normalizedAnswerTokens(prediction)
            let expected = normalizedAnswerTokens(reference)
            guard !predicted.isEmpty, !expected.isEmpty else {
                return predicted == expected ? 1 : 0
            }
            var remaining = Dictionary(grouping: expected, by: { $0 }).mapValues(\.count)
            var overlap = 0
            for token in predicted where remaining[token, default: 0] > 0 {
                overlap += 1
                remaining[token, default: 0] -= 1
            }
            guard overlap > 0 else { return 0 }
            let precision = Double(overlap) / Double(predicted.count)
            let recall = Double(overlap) / Double(expected.count)
            return 2 * precision * recall / (precision + recall)
        case .rougeL:
            let predicted = normalizedAnswerTokens(prediction)
            let expected = normalizedAnswerTokens(reference)
            guard !predicted.isEmpty, !expected.isEmpty else { return 0 }
            var previous = Array(repeating: 0, count: expected.count + 1)
            for predictedToken in predicted {
                var current = Array(repeating: 0, count: expected.count + 1)
                for (index, expectedToken) in expected.enumerated() {
                    current[index + 1] = predictedToken == expectedToken
                        ? previous[index] + 1
                        : max(previous[index + 1], current[index])
                }
                previous = current
            }
            let overlap = previous[expected.count]
            let precision = Double(overlap) / Double(predicted.count)
            let recall = Double(overlap) / Double(expected.count)
            return overlap == 0 ? 0 : 2 * precision * recall / (precision + recall)
        case .retrieval:
            let expectedNumbers = numericTokens(reference)
            guard let expected = expectedNumbers.first else { return 0 }
            let predicted = numericTokens(prediction)
            guard !predicted.isEmpty else { return 0 }
            return Double(predicted.filter { $0 == expected }.count) / Double(predicted.count)
        }
    }

    private static func normalizedAnswerTokens(_ text: String) -> [String] {
        let lowered = text.lowercased()
        let scalars = lowered.unicodeScalars.map { scalar -> Character in
            CharacterSet.punctuationCharacters.contains(scalar) ? " " : Character(String(scalar))
        }
        return String(scalars).split(whereSeparator: { $0.isWhitespace }).map(String.init)
            .filter { $0 != "a" && $0 != "an" && $0 != "the" }
    }

    private static func numericTokens(_ text: String) -> [String] {
        text.components(separatedBy: CharacterSet.decimalDigits.inverted).filter { !$0.isEmpty }
    }

    private static func toolCallsMatch(
        expected: [FusedExternalToolExpectation],
        observed: [ToolCall]
    ) -> Bool {
        guard expected.count == observed.count else { return false }

        func expectationMatches(
            _ expectation: FusedExternalToolExpectation,
            _ call: ToolCall
        ) -> Bool {
            guard expectation.name == call.name,
                  call.arguments.keys.allSatisfy({ expectation.arguments[$0] != nil }) else {
                return false
            }
            return expectation.arguments.allSatisfy { key, accepted in
                guard let value = call.arguments[key] else { return accepted.contains("") }
                let canonical = canonicalToolArgument(value)
                return accepted.contains { canonicalToolArgument($0) == canonical }
            }
        }

        func match(_ index: Int, remaining: [Int]) -> Bool {
            guard index < expected.count else { return remaining.isEmpty }
            for observedIndex in remaining where expectationMatches(expected[index], observed[observedIndex]) {
                if match(index + 1, remaining: remaining.filter { $0 != observedIndex }) {
                    return true
                }
            }
            return false
        }

        return match(0, remaining: Array(observed.indices))
    }

    private static func canonicalToolArgument(_ value: String) -> String {
        var trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.count >= 2,
           (trimmed.hasPrefix("'") && trimmed.hasSuffix("'")) {
            trimmed.removeFirst()
            trimmed.removeLast()
        }
        guard let data = trimmed.data(using: .utf8),
              let value = try? JSONDecoder().decode(OpenAIJSONValue.self, from: data) else {
            return trimmed
        }
        if case .string(let string) = value {
            return string
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let canonical = try? encoder.encode(value) else { return trimmed }
        return String(decoding: canonical, as: UTF8.self)
    }

    private static func programHarness(
        candidate: String,
        evaluation: FusedExternalCodeEvaluation,
        entryPoint: String?,
        tests: [FusedExternalCodeTest]
    ) throws -> String {
        let encodedCandidate = Data(candidate.utf8).base64EncodedString()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let encodedTests = (try encoder.encode(tests)).base64EncodedString()
        switch evaluation {
        case .stdin:
            return """
            import base64, contextlib, io, json, sys, typing
            candidate_source = base64.b64decode(\"\(encodedCandidate)\").decode(\"utf-8\")
            cases = json.loads(base64.b64decode(\"\(encodedTests)\").decode(\"utf-8\"))

            def normalize_output(value):
                return [line.rstrip().split() for line in value.strip().splitlines()]

            for case in cases:
                namespace = {\"__name__\": \"__main__\"}
                namespace.update(vars(typing))
                input_stream = io.StringIO(case[\"input\"])
                output_stream = io.StringIO()
                previous_stdin = sys.stdin
                sys.stdin = input_stream
                try:
                    with contextlib.redirect_stdout(output_stream):
                        exec(compile(candidate_source, \"candidate.py\", \"exec\"), namespace)
                finally:
                    sys.stdin = previous_stdin
                assert normalize_output(output_stream.getvalue()) == normalize_output(case[\"output\"])
            """
        case .functional:
            guard let entryPoint else {
                throw FusedBenchmarkError.invalidExternalFixture(
                    "functional code evaluation is missing its entry point"
                )
            }
            let encodedEntryPoint = Data(entryPoint.utf8).base64EncodedString()
            return """
            import base64, json, typing
            candidate_source = base64.b64decode(\"\(encodedCandidate)\").decode(\"utf-8\")
            entry_point = base64.b64decode(\"\(encodedEntryPoint)\").decode(\"utf-8\")
            cases = json.loads(base64.b64decode(\"\(encodedTests)\").decode(\"utf-8\"))
            namespace = {\"__name__\": \"candidate\"}
            namespace.update(vars(typing))
            exec(compile(candidate_source, \"candidate.py\", \"exec\"), namespace)
            for case in cases:
                args = [json.loads(line) for line in case[\"input\"].splitlines()]
                expected = json.loads(case[\"output\"])
                actual = getattr(namespace[\"Solution\"](), entry_point)(*args)
                assert actual == expected, (actual, expected)
            """
        case .function:
            throw FusedBenchmarkError.invalidExternalFixture(
                "function code evaluation does not use the program harness"
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

    private static func extractCode(_ response: String) -> String {
        var text = response
        if let opening = text.range(of: "```") {
            let afterOpening = text[opening.upperBound...]
            let start = afterOpening.firstIndex(of: "\n").map { text.index(after: $0) }
                ?? opening.upperBound
            let tail = text[start...]
            text = tail.range(of: "```").map { String(tail[..<$0.lowerBound]) }
                ?? String(tail)
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

struct FusedBenchmarkPlannedModel: Codable, Hashable {
    let id: String
    let profiles: [FusedBenchmarkSamplingProfile]
    let catalogRepository: String?
    let catalogRevision: String?
    let installed: Bool
    let runtimeManifestID: String?
    let runtimeManifestSchemaVersion: Int?
    let runtimeManifestSHA256: String?
    let runtimeManifestSources: [MereRunModelManifest.SourceProvenance]?

    static func resolve(
        id: String,
        suite: FusedBenchmarkSuiteSelection,
        fileManager: FileManager = .default
    ) throws -> FusedBenchmarkPlannedModel {
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
        return FusedBenchmarkPlannedModel(
            id: id,
            profiles: FusedBenchmarkSamplingProfile.nativeProfiles(
                modelID: id,
                suite: suite
            ),
            catalogRepository: spec?.upstreamRepoId,
            catalogRevision: spec?.upstreamRevision,
            installed: runtimeRoot != nil,
            runtimeManifestID: manifest?.id,
            runtimeManifestSchemaVersion: manifest?.schemaVersion,
            runtimeManifestSHA256: manifestData.map(FusedBenchmarkHash.sha256),
            runtimeManifestSources: manifest?.sources
        )
    }
}

struct FusedBenchmarkPlannedCase: Codable, Hashable {
    let id: String
    let adapter: String
    let lane: String
    let resolved: Bool
    let unresolvedReason: String?
    let provenance: FusedBenchmarkProvenance
}

struct FusedBenchmarkRunnerIdentity: Codable, Hashable {
    let version: String
    let executableByteCount: Int64
    let executableSHA256: String

    static func current() throws -> FusedBenchmarkRunnerIdentity {
        let executable = CurrentExecutable.url().standardizedFileURL.resolvingSymlinksInPath()
        return FusedBenchmarkRunnerIdentity(
            version: MereRunCLIVersion.current,
            executableByteCount: try ModelArtifactPin.fileByteCount(executable),
            executableSHA256: try ModelArtifactPin.fileSHA256(executable)
        )
    }
}

struct FusedBenchmarkHost: Codable, Hashable {
    let processorName: String
    let physicalMemoryBytes: UInt64
    let architecture: String
    let operatingSystemVersion: String
    let logicalProcessorCount: Int

    static func current(
        machine: MereRunMachineProfile = .current,
        processInfo: ProcessInfo = .processInfo
    ) -> FusedBenchmarkHost {
        #if arch(arm64)
        let architecture = "arm64"
        #elseif arch(x86_64)
        let architecture = "x86_64"
        #else
        let architecture = "unknown"
        #endif
        return FusedBenchmarkHost(
            processorName: machine.processorName,
            physicalMemoryBytes: machine.physicalMemoryBytes,
            architecture: architecture,
            operatingSystemVersion: processInfo.operatingSystemVersionString,
            logicalProcessorCount: processInfo.processorCount
        )
    }
}

struct FusedBenchmarkCodeRuntimeIdentity: Codable, Hashable {
    let pythonResolvedPath: String
    let pythonExecutableSHA256: String
    let sandboxBackend: String

    static func resolve(
        python: String,
        sandbox: CodeExecutionSandboxMode
    ) throws -> FusedBenchmarkCodeRuntimeIdentity {
        let executable = try CodeExecutionSandbox.resolvedPythonExecutable(python)
        return FusedBenchmarkCodeRuntimeIdentity(
            pythonResolvedPath: executable.path,
            pythonExecutableSHA256: try ModelArtifactPin.fileSHA256(executable),
            sandboxBackend: try CodeExecutionSandbox.resolvedBackend(mode: sandbox).rawValue
        )
    }
}

struct FusedBenchmarkPlan: Codable, Hashable {
    let runnerVersion: String
    let runnerExecutableByteCount: Int64
    let runnerExecutableSHA256: String
    let host: FusedBenchmarkHost
    let manifestID: String
    let manifestVersion: String
    let manifestSHA256: String
    let suite: String
    let models: [FusedBenchmarkPlannedModel]
    let trials: Int
    let qualityLane: String
    let performanceLane: String
    let settings: FusedBenchmarkRunSettings
    let cases: [FusedBenchmarkPlannedCase]
    let importedFixtureFiles: [String]

    var expectedResultKeys: Set<FusedBenchmarkResultKey> {
        var keys: Set<FusedBenchmarkResultKey> = []
        for model in models {
            for profile in model.profiles {
                for trial in 1...trials {
                    for benchmarkCase in cases {
                        keys.insert(FusedBenchmarkResultKey(
                            lane: "quality-final-target",
                            model: model.id,
                            profile: profile.name,
                            trial: trial,
                            caseID: benchmarkCase.id
                        ))
                    }
                }
                if performanceLane == FusedBenchmarkPerformanceLane.native.rawValue {
                    for benchmarkCase in cases {
                        keys.insert(FusedBenchmarkResultKey(
                            lane: "performance-native-runtime",
                            model: model.id,
                            profile: profile.name,
                            trial: 1,
                            caseID: benchmarkCase.id
                        ))
                    }
                }
            }
        }
        return keys
    }

    func contentSHA256() throws -> String {
        FusedBenchmarkHash.sha256(try JSONEncoder.sorted.encode(self))
    }
}

struct FusedBenchmarkRunSettings: Codable, Hashable {
    let maxTokensOverride: Int?
    let contextSize: Int
    let logprobs: String
    let topLogprobs: Int
    let executionTimeout: Double
    let pythonExecutable: String
    let pythonResolvedPath: String?
    let pythonExecutableSHA256: String?
    let sandbox: String
    let sandboxBackend: String?
    let allowCodeExecution: Bool
    let logResponses: Bool
}

struct FusedBenchmarkResultKey: Codable, Hashable {
    let lane: String
    let model: String
    let profile: String
    let trial: Int
    let caseID: String
}

extension FusedBenchmarkCaseResult {
    var key: FusedBenchmarkResultKey {
        FusedBenchmarkResultKey(
            lane: lane,
            model: model,
            profile: profile,
            trial: trial,
            caseID: caseID
        )
    }
}

struct FusedBenchmarkCheckpoint: Codable {
    static let currentSchemaVersion = 2

    let schemaVersion: Int
    let createdAt: Date
    let updatedAt: Date
    let planSHA256: String
    let plan: FusedBenchmarkPlan
    let results: [FusedBenchmarkCaseResult]

    init(
        plan: FusedBenchmarkPlan,
        results: [FusedBenchmarkCaseResult],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) throws {
        self.schemaVersion = Self.currentSchemaVersion
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.planSHA256 = try plan.contentSHA256()
        self.plan = plan
        self.results = results
    }

    func validatedResults(for expectedPlan: FusedBenchmarkPlan) throws -> [FusedBenchmarkCaseResult] {
        guard schemaVersion == Self.currentSchemaVersion else {
            throw FusedBenchmarkCheckpointError.invalid(
                "unsupported schema version \(schemaVersion)"
            )
        }
        let storedPlanSHA256 = try plan.contentSHA256()
        guard planSHA256 == storedPlanSHA256 else {
            throw FusedBenchmarkCheckpointError.invalid(
                "stored plan hash \(planSHA256) does not match its payload \(storedPlanSHA256)"
            )
        }
        let expectedPlanSHA256 = try expectedPlan.contentSHA256()
        guard planSHA256 == expectedPlanSHA256 else {
            throw FusedBenchmarkCheckpointError.invalid(
                "run plan changed (checkpoint \(planSHA256), invocation \(expectedPlanSHA256))"
            )
        }
        let keys = results.map(\.key)
        guard Set(keys).count == keys.count else {
            throw FusedBenchmarkCheckpointError.invalid("duplicate completed case-trials")
        }
        let unexpected = Set(keys).subtracting(expectedPlan.expectedResultKeys)
        guard unexpected.isEmpty else {
            throw FusedBenchmarkCheckpointError.invalid(
                "contains \(unexpected.count) result(s) outside the run plan"
            )
        }
        return results
    }
}

enum FusedBenchmarkCheckpointError: LocalizedError {
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let detail):
            return "Invalid fused benchmark checkpoint: \(detail)."
        }
    }
}

struct FusedBenchmarkCheckpointStore {
    let url: URL
    private let fileManager: FileManager

    init(url: URL, fileManager: FileManager = .default) {
        self.url = url
        self.fileManager = fileManager
    }

    func initialize(plan: FusedBenchmarkPlan) throws {
        guard !fileManager.fileExists(atPath: url.path) else {
            throw FusedBenchmarkCheckpointError.invalid(
                "file already exists at \(url.path); use --resume or choose a new path"
            )
        }
        try write(plan: plan, results: [])
    }

    func loadResults(validating plan: FusedBenchmarkPlan) throws -> [FusedBenchmarkCaseResult] {
        guard fileManager.fileExists(atPath: url.path) else {
            throw FusedBenchmarkCheckpointError.invalid("file does not exist at \(url.path)")
        }
        let checkpoint = try loadCheckpoint()
        return try checkpoint.validatedResults(for: plan)
    }

    func write(plan: FusedBenchmarkPlan, results: [FusedBenchmarkCaseResult]) throws {
        let now = Date()
        let createdAt = if fileManager.fileExists(atPath: url.path) {
            try loadCheckpoint().createdAt
        } else {
            now
        }
        let checkpoint = try FusedBenchmarkCheckpoint(
            plan: plan,
            results: results,
            createdAt: createdAt,
            updatedAt: now
        )
        let parent = url.deletingLastPathComponent()
        try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(checkpoint).write(to: url, options: .atomic)
    }

    private func loadCheckpoint() throws -> FusedBenchmarkCheckpoint {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(FusedBenchmarkCheckpoint.self, from: data)
    }
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

private struct FusedBenchmarkProgress: Codable {
    let expectedCaseTrials: Int
    let completedCaseTrials: Int
    let remainingCaseTrials: Int
    let complete: Bool
}

private struct FusedBenchmarkReport: Codable {
    let plan: FusedBenchmarkPlan
    let results: [FusedBenchmarkCaseResult]
    let progress: FusedBenchmarkProgress
    let calibrationByModelProfile: [FusedBenchmarkCalibrationGroup]
    let laneBreakdown: [FusedBenchmarkBreakdown]
    let capabilityBreakdown: [FusedBenchmarkBreakdown]
    let sourceBreakdown: [FusedBenchmarkBreakdown]

    init(plan: FusedBenchmarkPlan, results: [FusedBenchmarkCaseResult]) {
        self.plan = plan
        self.results = results
        let expected = plan.expectedResultKeys.count
        let completed = Set(results.map(\.key)).intersection(plan.expectedResultKeys).count
        self.progress = FusedBenchmarkProgress(
            expectedCaseTrials: expected,
            completedCaseTrials: completed,
            remainingCaseTrials: max(0, expected - completed),
            complete: completed == expected
        )
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
            "mere.run version: \(plan.runnerVersion)",
            "runner executable sha256: \(plan.runnerExecutableSHA256)",
            "host: \(plan.host.processorName), \(plan.host.physicalMemoryBytes) bytes, "
                + "\(plan.host.architecture), \(plan.host.operatingSystemVersion)",
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

        lines.append(
            "progress: \(progress.completedCaseTrials)/\(progress.expectedCaseTrials) case-trials "
                + "(\(progress.complete ? "complete" : "partial"))"
        )
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
