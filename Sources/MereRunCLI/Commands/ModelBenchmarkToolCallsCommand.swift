import ArgumentParser
import Foundation
import MereRunCore

struct ModelBenchmarkToolCalls: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tool-calls",
        abstract: "Run a small tool-call selection evaluation against local chat models."
    )

    @Option(name: [.long], help: "Comma-separated model ids. Defaults to Q36 and Gemma 4 12B 4-bit.")
    var models: String?

    @Option(
        name: [.long],
        help: "Override the installed Laguna target with a local poolside/Laguna-S-2.1-NVFP4-mlx checkpoint directory."
    )
    var lagunaPath: String?

    @Option(
        name: [.long],
        help: "Override the installed Laguna DFlash companion with a local poolside/Laguna-S-2.1-DFlash checkpoint directory."
    )
    var lagunaDflashPath: String?

    @Option(name: [.long], help: "Laguna DFlash speculative tokens per round (1...15).")
    var lagunaDflashTokens: Int = LagunaDFlashRouting.defaultSpeculativeTokens

    @Option(
        name: [.long],
        help: "Use Laguna DFlash when the effective output budget is at least this many tokens."
    )
    var lagunaDflashMinTokens: Int = LagunaDFlashRouting.defaultMinimumOutputTokens

    @Option(name: [.long], help: "Comma-separated tool-call case ids.")
    var cases: String?

    @Option(name: [.long], help: "Maximum generated tokens per case.")
    var maxTokens: Int = 192

    @Option(name: [.long], help: "Temperature for generation.")
    var temperature: Double = 0

    @Option(name: [.long], help: "Top-p for generation.")
    var topP: Double = 1

    @Option(name: [.customLong("top-k")], help: "Top-k for generation; zero disables it.")
    var topK: Int = 0

    @Option(name: [.customLong("min-p")], help: "Min-p cutoff relative to the most likely token.")
    var minP: Double?

    @Option(name: [.long], help: "Maximum context tokens passed to the runtime.")
    var contextSize: Int?

    @Flag(name: [.long], help: "Print the benchmark plan without loading models.")
    var dryRun: Bool = false

    @Flag(name: [.long], help: "Include model responses and observed calls in the report.")
    var logResponses: Bool = false

    @Flag(name: [.long], help: "Emit machine-readable JSON.")
    var json: Bool = false

    private static let defaultModelIDs = [
        Q35Resources.q36NanoModelId,
        Gemma4Resources.twelveB4BitModelId,
    ]

    func validate() throws {
        guard maxTokens > 0 else {
            throw ValidationError("--max-tokens must be greater than zero.")
        }
        guard (0...2).contains(temperature), temperature.isFinite else {
            throw ValidationError("--temperature must be finite and between 0 and 2.")
        }
        guard (0...1).contains(topP), topP.isFinite else {
            throw ValidationError("--top-p must be finite and between 0 and 1.")
        }
        guard topK >= 0 else {
            throw ValidationError("--top-k must be zero or greater.")
        }
        if let minP, !(0...1).contains(minP) || !minP.isFinite {
            throw ValidationError("--min-p must be finite and between 0 and 1.")
        }
        if let contextSize {
            guard contextSize > 0 else {
                throw ValidationError("--context-size must be greater than zero.")
            }
        }
        guard (1...15).contains(lagunaDflashTokens) else {
            throw ValidationError("--laguna-dflash-tokens must be between 1 and 15.")
        }
        guard lagunaDflashMinTokens > 0 else {
            throw ValidationError("--laguna-dflash-min-tokens must be greater than zero.")
        }
        _ = try selectedModelIDs()
        _ = try selectedCases()
    }

    func run() async throws {
        let modelIDs = try selectedModelIDs()
        let benchmarkCases = try selectedCases()
        let reportPlan = ToolBenchmarkPlan(
            models: modelIDs,
            cases: benchmarkCases.map(\.caseID),
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: resolvedMinP,
            contextSize: contextSize
        )

        if dryRun {
            let report = ToolBenchmarkReport(plan: reportPlan, models: [])
            try printReport(report)
            return
        }

        try MLXBundleSupport.ensureAvailable(quiet: json)

        var modelResults: [ToolBenchmarkModelResult] = []
        modelResults.reserveCapacity(modelIDs.count)
        for modelID in modelIDs {
            if !json {
                CLIStderr.write("Benchmarking tool calls for \(modelID)...\n")
            }
            modelResults.append(try await runModel(modelID, cases: benchmarkCases))
        }

        let report = ToolBenchmarkReport(plan: reportPlan, models: modelResults)
        try printReport(report)
    }

    func selectedModelIDs() throws -> [String] {
        let rawModels = models?.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        } ?? (lagunaPath == nil && lagunaDflashPath == nil
            ? Self.defaultModelIDs
            : [LagunaResources.modelID])
        let modelIDs = Self.deduplicated(rawModels.filter { !$0.isEmpty })
        guard !modelIDs.isEmpty else {
            throw ValidationError("--models must include at least one model id.")
        }
        return modelIDs
    }

    func selectedCases() throws -> [ToolBenchmarkCase] {
        guard let cases else {
            return ToolBenchmarkCase.mereToolSlice
        }
        let requestedIDs = cases
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !requestedIDs.isEmpty else {
            throw ValidationError("--cases must include at least one case id.")
        }
        var selected: [ToolBenchmarkCase] = []
        selected.reserveCapacity(requestedIDs.count)
        for caseID in requestedIDs {
            guard let task = ToolBenchmarkCase.mereToolSlice.first(where: { $0.caseID == caseID }) else {
                throw ValidationError("Unknown tool-call case id: \(caseID).")
            }
            selected.append(task)
        }
        return selected
    }

    private func runModel(_ modelID: String, cases: [ToolBenchmarkCase]) async throws -> ToolBenchmarkModelResult {
        if LagunaResources.handles(modelSpec: modelID) {
            let managedID = LagunaResources.managedModelID(for: modelID)
                ?? LagunaResources.modelID
            guard let resolvedLagunaPath = lagunaPath
                ?? ManagedModelResolver.resolveInstalledModel(id: managedID)?.path else {
                return ToolBenchmarkModelResult.missing(
                    model: modelID,
                    reason: "Model is not installed. Run "
                        + "`mere.run model pull \(managedID)` first."
                )
            }
            let resolvedDFlashPath = lagunaDflashPath
                ?? LagunaResources.installedDFlashPath(for: managedID)
            let generator = LagunaGenerator(
                dflashModelPath: resolvedDFlashPath,
                dflashSpeculativeTokens: lagunaDflashTokens,
                dflashMinimumOutputTokens: lagunaDflashMinTokens
            )
            do {
                let result = try await runCases(
                    modelID,
                    engine: resolvedDFlashPath == nil
                        ? "laguna-mlx"
                        : "laguna-mlx+dflash-auto-k\(lagunaDflashTokens)"
                            + "-min\(lagunaDflashMinTokens)",
                    modelPath: resolvedLagunaPath,
                    cases: cases,
                    generate: { request in
                        try await generator.chat(
                            request,
                            modelPath: resolvedLagunaPath,
                            progressHandler: nil
                        )
                    }
                )
                await generator.unload()
                return result
            } catch {
                await generator.unload()
                throw error
            }
        }

        guard let spec = ManagedModelCatalog.spec(for: modelID) else {
            return ToolBenchmarkModelResult.missing(model: modelID, reason: "Unknown model id.")
        }
        guard let installedURL = ManagedModelResolver.resolveInstalledModel(id: modelID) else {
            return ToolBenchmarkModelResult.missing(
                model: modelID,
                reason: "Model is not installed. Run `\(CLICommandDisplay.modelPullCommand(for: modelID))` first."
            )
        }

        switch spec.validationKind {
        case .gemma4, .gemma4Unified:
            let generator = Gemma4Generator(
                modelId: modelID,
                kvCacheQuantization: Self.defaultGemma4KVCacheQuantization(for: modelID)
            )
            let result = try await runCases(
                modelID,
                engine: "gemma4",
                modelPath: installedURL.path,
                cases: cases,
                generate: { request in
                    try await generator.chat(request, modelPath: installedURL.path, progressHandler: nil)
                }
            )
            await generator.unload()
            return result

        case .q35:
            let generator = Q35Generator(modelId: modelID)
            let result = try await runCases(
                modelID,
                engine: "q35-mlx",
                modelPath: installedURL.path,
                cases: cases,
                generate: { request in
                    try await generator.chat(request, modelPath: installedURL.path, progressHandler: nil)
                }
            )
            await generator.unload()
            return result

        case .lfm2:
            let generator = LFM2Generator(modelId: modelID)
            let result = try await runCases(
                modelID,
                engine: "lfm2-mlx",
                modelPath: installedURL.path,
                cases: cases,
                generate: { request in
                    try await generator.chat(request, modelPath: installedURL.path, progressHandler: nil)
                }
            )
            await generator.unload()
            return result

        default:
            return ToolBenchmarkModelResult.missing(
                model: modelID,
                reason: "Unsupported tool-call benchmark runtime: \(spec.validationKind.rawValue)."
            )
        }
    }

    private func runCases(
        _ modelID: String,
        engine: String,
        modelPath: String,
        cases: [ToolBenchmarkCase],
        generate: (ChatRequest) async throws -> ChatResponse
    ) async throws -> ToolBenchmarkModelResult {
        var caseResults: [ToolBenchmarkCaseResult] = []
        caseResults.reserveCapacity(cases.count)
        for benchmarkCase in cases {
            let request = ChatRequest(
                messages: [
                    ChatMessage(role: .system, content: Self.systemPrompt),
                    ChatMessage(role: .user, content: benchmarkCase.promptText),
                ],
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                topK: topK,
                minP: resolvedMinP,
                showThinking: false,
                tools: benchmarkCase.tools,
                stopOnEOS: true,
                maxContextTokens: contextSize
            )
            let start = Date()
            do {
                let response = try await generate(request)
                let generationSeconds = Date().timeIntervalSince(start)
                let observedCalls = (response.toolCalls ?? []).map(ToolBenchmarkObservedCall.init(call:))
                let evaluation = benchmarkCase.expectation.evaluate(observedCalls)
                caseResults.append(
                    ToolBenchmarkCaseResult(
                        caseID: benchmarkCase.caseID,
                        category: benchmarkCase.category,
                        passed: evaluation.passed,
                        passedChecks: evaluation.passedChecks,
                        totalChecks: evaluation.totalChecks,
                        failedChecks: evaluation.failedChecks,
                        generationSeconds: generationSeconds,
                        tokensGenerated: response.tokensGenerated,
                        decodeTokensPerSecond: response.decodeTokensPerSecond(elapsed: generationSeconds),
                        observedCalls: logResponses ? observedCalls : nil,
                        response: logResponses ? Self.cleanResponse(response.response) : nil,
                        error: nil
                    )
                )
            } catch {
                caseResults.append(
                    ToolBenchmarkCaseResult(
                        caseID: benchmarkCase.caseID,
                        category: benchmarkCase.category,
                        passed: false,
                        passedChecks: 0,
                        totalChecks: benchmarkCase.expectation.checkCount,
                        failedChecks: ["generation failed: \(String(describing: error))"],
                        generationSeconds: Date().timeIntervalSince(start),
                        tokensGenerated: 0,
                        decodeTokensPerSecond: nil,
                        observedCalls: nil,
                        response: nil,
                        error: String(describing: error)
                    )
                )
            }
        }
        return ToolBenchmarkModelResult(
            model: modelID,
            engine: engine,
            modelPath: modelPath,
            status: "completed",
            error: nil,
            cases: caseResults
        )
    }

    var resolvedMinP: Double {
        let includesLaguna = (try? selectedModelIDs().contains {
            LagunaResources.handles(modelSpec: $0)
        }) == true
        return minP ?? (includesLaguna ? LagunaResources.recommendedMinP : 0)
    }

    private func printReport(_ report: ToolBenchmarkReport) throws {
        if json {
            print(try report.jsonString())
        } else {
            print(report.renderText())
        }
    }

    private static func defaultGemma4KVCacheQuantization(for modelID: String) -> Gemma4KVCacheQuantization {
        let usesTurboDefaults = Gemma4Resources.usesTurboDefaults(modelSpec: modelID)
            && Gemma4Resources.supportsDefaultTurboKVQuantization
        return Gemma4KVCacheQuantization(
            bits: usesTurboDefaults ? Gemma4Resources.defaultTurboKVBits : nil,
            scheme: usesTurboDefaults ? Gemma4Resources.defaultTurboKVQuantizationScheme : Gemma4Resources.defaultKVQuantizationScheme,
            groupSize: Gemma4Resources.defaultKVGroupSize,
            quantizedStart: usesTurboDefaults ? Gemma4Resources.defaultTurboQuantizedKVStart : Gemma4Resources.defaultQuantizedKVStart
        )
    }

    private static func cleanResponse(_ response: String) -> String {
        var cleaned = response.replacingOccurrences(
            of: "(?is)<think>.*?</think>",
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: "(?is)<think>.*\\z",
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: "(?i)</think>",
            with: "",
            options: .regularExpression
        )
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func deduplicated(_ values: [String]) -> [String] {
        var seen: Set<String> = []
        var result: [String] = []
        result.reserveCapacity(values.count)
        for value in values {
            guard !seen.contains(value) else { continue }
            seen.insert(value)
            result.append(value)
        }
        return result
    }

    private static let systemPrompt = """
    You are running a local tool-calling benchmark. Use the provided tool schema exactly.
    If the user request requires a lookup or local action, call exactly one appropriate tool.
    If the answer is already fully present in the prompt evidence, do not call a tool.
    Do not invent tool names, arguments, emails, workspace ids, commands, or paths.
    """
}

struct ToolBenchmarkCase: Encodable {
    let caseID: String
    let category: String
    let title: String
    let prompt: String
    let tools: [ToolDefinition]
    let expectation: ToolBenchmarkExpectation

    var promptText: String {
        """
        Tool-call benchmark case: \(caseID)
        Category: \(category)
        Title: \(title)

        \(prompt)
        """
    }

    static let mereToolSlice = [
        ToolBenchmarkCase(
            caseID: "MereTool/0",
            category: "email-search",
            title: "Search email by sender and workspace",
            prompt: """
            User request: Find recent email from abenewsoil@gmail.com in sawfwair after 2026-06-01.
            Requirement: call the email search tool. Do not answer from memory.
            """,
            tools: [ToolBenchmarkTools.emailSearch],
            expectation: ToolBenchmarkExpectation(
                expectedToolName: "mere_email_search",
                expectedArguments: [
                    "sender": "abenewsoil@gmail.com",
                    "workspace": "sawfwair",
                    "after": "2026-06-01",
                ]
            )
        ),
        ToolBenchmarkCase(
            caseID: "MereTool/1",
            category: "email-open",
            title: "Open a specific local email",
            prompt: """
            User request: Open local email mail_801 so I can inspect the quote total.
            Requirement: call the email open tool with the local id.
            """,
            tools: [ToolBenchmarkTools.emailOpen],
            expectation: ToolBenchmarkExpectation(
                expectedToolName: "mere_email_open",
                expectedArguments: ["id": "mail_801"]
            )
        ),
        ToolBenchmarkCase(
            caseID: "MereTool/2",
            category: "project-search",
            title: "Find blocked projects in workspace",
            prompt: """
            User request: Show blocked projects in the sawfwair workspace.
            Requirement: call the project search tool.
            """,
            tools: [ToolBenchmarkTools.projectSearch],
            expectation: ToolBenchmarkExpectation(
                expectedToolName: "mere_project_search",
                expectedArguments: [
                    "workspace": "sawfwair",
                    "status": "blocked",
                ]
            )
        ),
        ToolBenchmarkCase(
            caseID: "MereTool/3",
            category: "audit-search",
            title: "Find failed email audit entries",
            prompt: """
            User request: Look up failed email operations in the sawfwair audit log.
            Requirement: call the audit search tool.
            """,
            tools: [ToolBenchmarkTools.auditSearch],
            expectation: ToolBenchmarkExpectation(
                expectedToolName: "mere_audit_search",
                expectedArguments: [
                    "workspace": "sawfwair",
                    "status": "failed",
                    "app": "email",
                ]
            )
        ),
        ToolBenchmarkCase(
            caseID: "MereTool/4",
            category: "no-tool",
            title: "Do not call a tool for visible evidence",
            prompt: """
            Evidence:
            auth_status="signed_in"
            workspace="sawfwair"

            User request: What is my auth status?
            Requirement: answer from evidence; do not call a tool.
            """,
            tools: [ToolBenchmarkTools.emailSearch],
            expectation: ToolBenchmarkExpectation(expectNoToolCall: true)
        ),
        ToolBenchmarkCase(
            caseID: "MereTool/5",
            category: "workspace-select",
            title: "Select a workspace by id",
            prompt: """
            User request: Switch the console to workspace ws_saw_01.
            Requirement: call the workspace select tool.
            """,
            tools: [ToolBenchmarkTools.workspaceSelect],
            expectation: ToolBenchmarkExpectation(
                expectedToolName: "mere_workspace_select",
                expectedArguments: ["workspace_id": "ws_saw_01"]
            )
        ),
        ToolBenchmarkCase(
            caseID: "MereTool/6",
            category: "command-plan",
            title: "Plan safe project export",
            prompt: """
            User request: Prepare a dry-run export for projects in sawfwair.
            Requirement: call the command plan tool with a safe dry-run command.
            """,
            tools: [ToolBenchmarkTools.commandPlan],
            expectation: ToolBenchmarkExpectation(
                expectedToolName: "mere_command_plan",
                expectedArguments: ["command": "mere projects export"],
                requiredArgumentFragments: ["--dry-run", "sawfwair"],
                forbiddenArgumentFragments: ["--apply"]
            )
        ),
        ToolBenchmarkCase(
            caseID: "MereTool/7",
            category: "command-plan",
            title: "Plan safe email import",
            prompt: """
            User request: Check what an email import would do for workspace sawfwair.
            Requirement: call the command plan tool. It must be a dry run.
            """,
            tools: [ToolBenchmarkTools.commandPlan],
            expectation: ToolBenchmarkExpectation(
                expectedToolName: "mere_command_plan",
                expectedArguments: ["command": "mere email import"],
                requiredArgumentFragments: ["--dry-run", "sawfwair"],
                forbiddenArgumentFragments: ["--apply"]
            )
        ),
        ToolBenchmarkCase(
            caseID: "MereTool/8",
            category: "email-search",
            title: "Search email by attachment flag",
            prompt: """
            User request: Find sawfwair emails from Abe that have attachments.
            Requirement: call the email search tool with an attachment filter.
            """,
            tools: [ToolBenchmarkTools.emailSearch],
            expectation: ToolBenchmarkExpectation(
                expectedToolName: "mere_email_search",
                expectedArguments: [
                    "workspace": "sawfwair",
                    "sender": "Abe",
                    "has_attachment": "true",
                ]
            )
        ),
        ToolBenchmarkCase(
            caseID: "MereTool/9",
            category: "no-tool",
            title: "Do not use tools for explicit refusal boundary",
            prompt: """
            Evidence:
            requested_action="delete archived emails"
            confirmation_present=false

            User request: Delete archived emails.
            Requirement: do not call a tool. Ask for confirmation instead.
            """,
            tools: [ToolBenchmarkTools.commandPlan],
            expectation: ToolBenchmarkExpectation(expectNoToolCall: true)
        ),
    ]
}

enum ToolBenchmarkTools {
    static let emailSearch = ToolDefinition(
        name: "mere_email_search",
        description: "Search local synced email metadata.",
        parameters: [
            "workspace": ToolParameterProperty(type: "string", description: "Workspace slug or id."),
            "sender": ToolParameterProperty(type: "string", description: "Optional sender name or email."),
            "after": ToolParameterProperty(type: "string", description: "Optional lower-bound ISO date."),
            "has_attachment": ToolParameterProperty(type: "string", description: "Optional true/false attachment filter."),
        ],
        required: ["workspace"]
    )

    static let emailOpen = ToolDefinition(
        name: "mere_email_open",
        description: "Open a local email record by local id.",
        parameters: [
            "id": ToolParameterProperty(type: "string", description: "Local email id."),
        ],
        required: ["id"]
    )

    static let projectSearch = ToolDefinition(
        name: "mere_project_search",
        description: "Search local project records.",
        parameters: [
            "workspace": ToolParameterProperty(type: "string", description: "Workspace slug or id."),
            "status": ToolParameterProperty(type: "string", description: "Optional project status."),
            "query": ToolParameterProperty(type: "string", description: "Optional text query."),
        ],
        required: ["workspace"]
    )

    static let auditSearch = ToolDefinition(
        name: "mere_audit_search",
        description: "Search local Mere operation audit records.",
        parameters: [
            "workspace": ToolParameterProperty(type: "string", description: "Workspace slug or id."),
            "app": ToolParameterProperty(type: "string", description: "Optional app name."),
            "status": ToolParameterProperty(type: "string", description: "Optional status such as failed or ok."),
        ],
        required: ["workspace"]
    )

    static let workspaceSelect = ToolDefinition(
        name: "mere_workspace_select",
        description: "Select the active workspace for the console.",
        parameters: [
            "workspace_id": ToolParameterProperty(type: "string", description: "Workspace id to select."),
        ],
        required: ["workspace_id"]
    )

    static let commandPlan = ToolDefinition(
        name: "mere_command_plan",
        description: "Prepare a safe local Mere CLI command plan without executing it.",
        parameters: [
            "command": ToolParameterProperty(type: "string", description: "Command to present for approval."),
        ],
        required: ["command"]
    )
}

struct ToolBenchmarkExpectation: Encodable {
    var expectedToolName: String?
    var expectedArguments: [String: String] = [:]
    var requiredArgumentFragments: [String] = []
    var forbiddenArgumentFragments: [String] = []
    var expectNoToolCall: Bool = false

    var checkCount: Int {
        if expectNoToolCall {
            return 1
        }
        return 1 + expectedArguments.count + requiredArgumentFragments.count + forbiddenArgumentFragments.count
    }

    func evaluate(_ observedCalls: [ToolBenchmarkObservedCall]) -> ToolBenchmarkEvaluation {
        var failedChecks: [String] = []
        if expectNoToolCall {
            if !observedCalls.isEmpty {
                failedChecks.append("expected no tool call, observed \(observedCalls.map(\.name).joined(separator: ", "))")
            }
            return ToolBenchmarkEvaluation(
                passed: failedChecks.isEmpty,
                passedChecks: failedChecks.isEmpty ? 1 : 0,
                totalChecks: 1,
                failedChecks: failedChecks
            )
        }

        guard let expectedToolName else {
            failedChecks.append("missing expected tool name in benchmark expectation")
            return ToolBenchmarkEvaluation(passed: false, passedChecks: 0, totalChecks: checkCount, failedChecks: failedChecks)
        }
        guard observedCalls.count == 1 else {
            failedChecks.append("expected exactly one tool call, observed \(observedCalls.count)")
            return ToolBenchmarkEvaluation(
                passed: false,
                passedChecks: max(0, checkCount - failedChecks.count),
                totalChecks: checkCount,
                failedChecks: failedChecks
            )
        }
        let call = observedCalls[0]
        if call.name != expectedToolName {
            failedChecks.append("expected tool \(expectedToolName), observed \(call.name)")
        }

        let argumentBlob = call.arguments
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: " ")
            .lowercased()
        for (key, expectedValue) in expectedArguments {
            guard let actual = call.arguments[key] else {
                failedChecks.append("missing argument: \(key)")
                continue
            }
            if !actual.lowercased().contains(expectedValue.lowercased()) {
                failedChecks.append("argument \(key) did not contain \(expectedValue)")
            }
        }
        for fragment in requiredArgumentFragments {
            if !argumentBlob.contains(fragment.lowercased()) {
                failedChecks.append("missing argument fragment: \(fragment)")
            }
        }
        for fragment in forbiddenArgumentFragments {
            if argumentBlob.contains(fragment.lowercased()) {
                failedChecks.append("forbidden argument fragment present: \(fragment)")
            }
        }

        return ToolBenchmarkEvaluation(
            passed: failedChecks.isEmpty,
            passedChecks: max(0, checkCount - failedChecks.count),
            totalChecks: checkCount,
            failedChecks: failedChecks
        )
    }
}

struct ToolBenchmarkEvaluation: Encodable {
    let passed: Bool
    let passedChecks: Int
    let totalChecks: Int
    let failedChecks: [String]
}

struct ToolBenchmarkObservedCall: Encodable {
    let name: String
    let arguments: [String: String]

    init(call: ToolCall) {
        self.name = call.name
        self.arguments = call.arguments
    }
}

private struct ToolBenchmarkPlan: Encodable {
    let models: [String]
    let cases: [String]
    let maxTokens: Int
    let temperature: Double
    let topP: Double
    let topK: Int
    let minP: Double
    let contextSize: Int?
}

private struct ToolBenchmarkReport: Encodable {
    let plan: ToolBenchmarkPlan
    let models: [ToolBenchmarkModelResult]

    func jsonString() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        return String(decoding: data, as: UTF8.self)
    }

    func renderText() -> String {
        var lines = [
            "Tool-call benchmark",
            "cases: \(plan.cases.joined(separator: ", "))",
            "models: \(plan.models.joined(separator: ", "))",
            "sampling: temperature=\(plan.temperature) top_p=\(plan.topP) "
                + "top_k=\(plan.topK) min_p=\(plan.minP) max_tokens=\(plan.maxTokens)",
            plan.contextSize.map { "context_size: \($0)" },
            "",
        ].compactMap { $0 }
        guard !models.isEmpty else {
            lines.append("dry run: no models loaded and no generations run")
            return lines.joined(separator: "\n")
        }
        lines.append("model                         engine       pass   checks   avg_gen_s  avg_tps")
        for result in models {
            lines.append(result.summaryLine())
        }
        lines.append("")
        for result in models {
            lines.append(result.renderDetails())
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }
}

private struct ToolBenchmarkModelResult: Encodable {
    let model: String
    let engine: String?
    let modelPath: String?
    let status: String
    let error: String?
    let cases: [ToolBenchmarkCaseResult]

    static func missing(model: String, reason: String) -> ToolBenchmarkModelResult {
        ToolBenchmarkModelResult(
            model: model,
            engine: nil,
            modelPath: nil,
            status: "skipped",
            error: reason,
            cases: []
        )
    }

    var passCount: Int {
        cases.filter(\.passed).count
    }

    var passedChecks: Int {
        cases.map(\.passedChecks).reduce(0, +)
    }

    var totalChecks: Int {
        cases.map(\.totalChecks).reduce(0, +)
    }

    func summaryLine() -> String {
        let passText = "\(passCount)/\(cases.count)"
        let checkText = "\(passedChecks)/\(totalChecks)"
        let avgGeneration = average(cases.compactMap(\.generationSeconds))
        let avgTPS = average(cases.compactMap(\.decodeTokensPerSecond))
        return [
            Self.padded(model, width: 29),
            Self.padded(engine ?? status, width: 12),
            Self.padded(passText, width: 6),
            Self.padded(checkText, width: 8),
            Self.padded(Self.format(avgGeneration), width: 10),
            Self.padded(Self.format(avgTPS), width: 7),
        ].joined(separator: " ")
    }

    func renderDetails() -> String {
        var lines = ["\(model): \(status)"]
        if let error {
            lines.append("  \(error)")
        }
        for result in cases {
            let mark = result.passed ? "pass" : "fail"
            let timing = [
                "gen=\(Self.format(result.generationSeconds))s",
                "checks=\(result.passedChecks)/\(result.totalChecks)",
                result.decodeTokensPerSecond.map { "tps=\(Self.format($0))" },
            ]
                .compactMap { $0 }
                .joined(separator: " ")
            lines.append("  \(mark) \(result.caseID) \(result.category) \(timing)")
            for failedCheck in result.failedChecks {
                lines.append("    \(failedCheck)")
            }
            if let observedCalls = result.observedCalls, !observedCalls.isEmpty {
                let rendered = observedCalls.map { call in
                    "\(call.name)(\(call.arguments.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: ", ")))"
                }
                    .joined(separator: "; ")
                lines.append("    calls: \(rendered)")
            }
            if let response = result.response, !response.isEmpty {
                lines.append("    response: \(response.replacingOccurrences(of: "\n", with: "\n      "))")
            }
            if let error = result.error {
                lines.append("    \(error.replacingOccurrences(of: "\n", with: "\n    "))")
            }
        }
        return lines.joined(separator: "\n")
    }

    private static func average(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        return values.reduce(0, +) / Double(values.count)
    }

    private func average(_ values: [Double]) -> Double? {
        Self.average(values)
    }

    private static func format(_ value: Double?) -> String {
        guard let value else { return "-" }
        return String(format: "%.2f", value)
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.2f", value)
    }

    private static func padded(_ value: String, width: Int) -> String {
        let padding = max(0, width - value.count)
        return value + String(repeating: " ", count: padding)
    }
}

private struct ToolBenchmarkCaseResult: Encodable {
    let caseID: String
    let category: String
    let passed: Bool
    let passedChecks: Int
    let totalChecks: Int
    let failedChecks: [String]
    let generationSeconds: Double
    let tokensGenerated: Int
    let decodeTokensPerSecond: Double?
    let observedCalls: [ToolBenchmarkObservedCall]?
    let response: String?
    let error: String?
}

private extension ChatResponse {
    func decodeTokensPerSecond(elapsed: Double) -> Double? {
        if let timing, let decodeTokensPerSecond = timing.decodeTokensPerSecond {
            return decodeTokensPerSecond
        }
        guard elapsed > 0, tokensGenerated > 0 else {
            return nil
        }
        return Double(tokensGenerated) / elapsed
    }
}
