import ArgumentParser
import Foundation
import MereRunCore

struct ModelBenchmarkCode: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "code",
        abstract: "Run a small real coding-eval slice against local coding models."
    )

    @Option(name: [.long], help: "Comma-separated model ids. Defaults to the installed coding comparison lane.")
    var models: String?

    @Option(name: [.long], help: "Benchmark suite.")
    var suite: CodeBenchmarkSuite = .humanEvalSlice

    @Option(name: [.long], help: "Comma-separated task ids from the selected suite.")
    var tasks: String?

    @Option(name: [.long], help: "Maximum generated tokens per task.")
    var maxTokens: Int = 1024

    @Option(name: [.long], help: "Temperature for generation.")
    var temperature: Double = 0

    @Option(name: [.long], help: "Top-p for generation.")
    var topP: Double = 1

    @Option(name: [.long], help: "Generated-code execution timeout in seconds.")
    var executionTimeout: Double = 5

    @Option(name: [.long], help: "Python executable used to run HumanEval slice tests.")
    var python: String = "python3"

    @Option(name: [.long], help: "Sandbox backend for generated-code execution.")
    var sandbox: CodeExecutionSandboxMode = .auto

    @Flag(name: [.long], help: "Print the benchmark plan without loading models or executing generated code.")
    var dryRun: Bool = false

    @Flag(name: [.long], help: "Acknowledge that benchmark scoring executes generated Python code locally.")
    var allowCodeExecution: Bool = false

    @Flag(name: [.long], help: "Emit machine-readable JSON.")
    var json: Bool = false

    private static let defaultModelIDs = [
        Q35Resources.ornith9BModelId,
        NorthMiniCodeResources.modelId,
        CodeGenResources.defaultModelId,
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
        guard executionTimeout > 0, executionTimeout.isFinite else {
            throw ValidationError("--execution-timeout must be a positive finite number.")
        }
        if !dryRun {
            try CodeExecutionSandbox.preflight(mode: sandbox)
        }
        if !dryRun && !allowCodeExecution {
            throw ValidationError(
                "model benchmark code executes generated Python. Re-run with --allow-code-execution."
            )
        }
        _ = try selectedModelIDs()
        _ = try selectedTasks()
    }

    func run() async throws {
        let modelIDs = try selectedModelIDs()
        let benchmarkTasks = try selectedTasks()
        let reportPlan = CodeBenchmarkPlan(
            suite: suite.rawValue,
            models: modelIDs,
            tasks: benchmarkTasks.map(\.taskID),
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            executionTimeout: executionTimeout,
            sandbox: sandbox.rawValue
        )

        if dryRun {
            let report = CodeBenchmarkReport(plan: reportPlan, models: [])
            try printReport(report)
            return
        }
        if modelIDs.contains(where: Self.requiresMLXBundle) {
            try MLXBundleSupport.ensureAvailable(quiet: json)
        }

        var modelResults: [CodeBenchmarkModelResult] = []
        modelResults.reserveCapacity(modelIDs.count)
        for modelID in modelIDs {
            if !json {
                CLIStderr.write("Benchmarking \(modelID)...\n")
            }
            modelResults.append(try await runModel(modelID, tasks: benchmarkTasks))
        }

        let report = CodeBenchmarkReport(plan: reportPlan, models: modelResults)
        try printReport(report)
    }

    private func selectedModelIDs() throws -> [String] {
        let rawModels = models?.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        } ?? Self.defaultModelIDs
        let modelIDs = rawModels.filter { !$0.isEmpty }
        guard !modelIDs.isEmpty else {
            throw ValidationError("--models must include at least one model id.")
        }
        return modelIDs
    }

    private func selectedTasks() throws -> [CodeBenchmarkTask] {
        let suiteTasks = suite.tasks
        guard let tasks else {
            return suiteTasks
        }
        let requestedIDs = tasks
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !requestedIDs.isEmpty else {
            throw ValidationError("--tasks must include at least one task id.")
        }
        var selected: [CodeBenchmarkTask] = []
        selected.reserveCapacity(requestedIDs.count)
        for taskID in requestedIDs {
            guard let task = suiteTasks.first(where: { $0.taskID == taskID }) else {
                throw ValidationError("Unknown task id for \(suite.rawValue): \(taskID).")
            }
            selected.append(task)
        }
        return selected
    }

    private func runModel(_ modelID: String, tasks: [CodeBenchmarkTask]) async throws -> CodeBenchmarkModelResult {
        guard let spec = ManagedModelCatalog.spec(for: modelID) else {
            return CodeBenchmarkModelResult.missing(model: modelID, reason: "Unknown model id.")
        }
        guard spec.category == .textCode else {
            return CodeBenchmarkModelResult.missing(model: modelID, reason: "Model is not a text-code model.")
        }
        guard let installedURL = ManagedModelResolver.resolveInstalledModel(id: modelID) else {
            return CodeBenchmarkModelResult.missing(
                model: modelID,
                reason: "Model is not installed. Run `mere.run model pull \(modelID)` first."
            )
        }

        switch spec.validationKind {
        case .codegenGGUF:
            let generator = CodeGenGenerator(modelId: modelID)
            do {
                let result = try await runTasks(
                    modelID,
                    engine: "codegen-gguf",
                    modelPath: installedURL.path,
                    tasks: tasks,
                    generate: { request in
                        try await generator.chat(request, modelPath: installedURL.path, progressHandler: nil)
                    }
                )
                await generator.unload()
                return result
            } catch {
                await generator.unload()
                throw error
            }

        case .q35:
            let generator = Q35Generator(modelId: modelID)
            do {
                let result = try await runTasks(
                    modelID,
                    engine: "q35-mlx",
                    modelPath: installedURL.path,
                    tasks: tasks,
                    generate: { request in
                        try await generator.chat(request, modelPath: installedURL.path, progressHandler: nil)
                    }
                )
                await generator.unload()
                return result
            } catch {
                await generator.unload()
                throw error
            }

        default:
            return CodeBenchmarkModelResult.missing(
                model: modelID,
                reason: "Unsupported coding benchmark runtime: \(spec.validationKind.rawValue)."
            )
        }
    }

    private func runTasks(
        _ modelID: String,
        engine: String,
        modelPath: String,
        tasks: [CodeBenchmarkTask],
        generate: (ChatRequest) async throws -> ChatResponse
    ) async throws -> CodeBenchmarkModelResult {
        var caseResults: [CodeBenchmarkCaseResult] = []
        caseResults.reserveCapacity(tasks.count)
        for task in tasks {
            let request = ChatRequest(
                messages: [
                    ChatMessage(role: .system, content: Self.systemPrompt),
                    ChatMessage(role: .user, content: task.promptText),
                ],
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                showThinking: false,
                stopOnEOS: true,
                stopSequences: Self.codeBenchmarkStopSequences
            )
            let generationStart = Date()
            do {
                let response = try await generate(request)
                let generationSeconds = Date().timeIntervalSince(generationStart)
                let candidate = task.candidateProgram(from: response.response)
                let execution = try CodeExecutionSandbox.runPython(
                    program: task.testProgram(candidateProgram: candidate),
                    python: python,
                    mode: sandbox,
                    timeout: executionTimeout
                )
                caseResults.append(
                    CodeBenchmarkCaseResult(
                        taskID: task.taskID,
                        entryPoint: task.entryPoint,
                        passed: execution.passed,
                        generationSeconds: generationSeconds,
                        executionSeconds: execution.seconds,
                        sandboxBackend: execution.backend.rawValue,
                        tokensGenerated: response.tokensGenerated,
                        decodeTokensPerSecond: response.decodeTokensPerSecond(elapsed: generationSeconds),
                        finishReason: response.finishReason?.rawValue,
                        reachedMaxTokens: response.reachedMaxTokens(limit: maxTokens),
                        error: execution.errorSummary
                    )
                )
            } catch {
                caseResults.append(
                    CodeBenchmarkCaseResult(
                        taskID: task.taskID,
                        entryPoint: task.entryPoint,
                        passed: false,
                        generationSeconds: Date().timeIntervalSince(generationStart),
                        executionSeconds: nil,
                        sandboxBackend: nil,
                        tokensGenerated: 0,
                        decodeTokensPerSecond: nil,
                        finishReason: nil,
                        reachedMaxTokens: false,
                        error: String(describing: error)
                    )
                )
            }
        }
        return CodeBenchmarkModelResult(
            model: modelID,
            engine: engine,
            modelPath: modelPath,
            status: "completed",
            error: nil,
            cases: caseResults
        )
    }

    private func printReport(_ report: CodeBenchmarkReport) throws {
        if json {
            print(try report.jsonString())
        } else {
            print(report.renderText())
        }
    }

    private static func requiresMLXBundle(modelID: String) -> Bool {
        ManagedModelCatalog.spec(for: modelID)?.validationKind == .q35
    }

    private static let systemPrompt = """
    You are completing Python programming benchmark tasks. Return only valid Python code.
    Do not include Markdown fences, prose, comments about your approach, or test code.
    Stop immediately after the requested function implementation.
    """

    private static let codeBenchmarkStopSequences = TextGenerationStopSequences.defaultRenderedChatStops + [
        "\n# The following code is for testing",
        "\nif __name__",
        "\ndef check(",
    ]
}

enum CodeBenchmarkSuite: String, CaseIterable, ExpressibleByArgument {
    case humanEvalSlice = "humaneval-slice"

    fileprivate var tasks: [CodeBenchmarkTask] {
        switch self {
        case .humanEvalSlice:
            return CodeBenchmarkTask.humanEvalSlice
        }
    }
}

private struct CodeBenchmarkTask: Encodable {
    let taskID: String
    let entryPoint: String
    let prompt: String
    let tests: String

    var promptText: String {
        """
        Complete the following HumanEval task. Return only the Python implementation.

        \(prompt)
        """
    }

    func candidateProgram(from response: String) -> String {
        let code = Self.extractCode(from: response, entryPoint: entryPoint)
        if code.contains("def \(entryPoint)") {
            return Self.withPromptImports(prompt: prompt, candidate: code)
        }
        return prompt + "\n" + code
    }

    func testProgram(candidateProgram: String) -> String {
        """
        \(candidateProgram)

        \(tests)

        check(\(entryPoint))
        """
    }

    private static func extractCode(from response: String, entryPoint: String) -> String {
        var text = fencedCodeBlock(in: response) ?? response
        text = text.replacingOccurrences(of: "\r\n", with: "\n")
        if let range = text.range(of: "from typing") {
            text = String(text[range.lowerBound...])
        } else if let range = text.range(of: "def \(entryPoint)") {
            text = String(text[range.lowerBound...])
        }
        if let fence = text.range(of: "```") {
            text = String(text[..<fence.lowerBound])
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func fencedCodeBlock(in response: String) -> String? {
        guard let opening = response.range(of: "```") else {
            return nil
        }
        var bodyStart = opening.upperBound
        if let newline = response[bodyStart...].firstIndex(of: "\n") {
            bodyStart = response.index(after: newline)
        }
        guard let closing = response[bodyStart...].range(of: "```") else {
            return String(response[bodyStart...])
        }
        return String(response[bodyStart..<closing.lowerBound])
    }

    private static func withPromptImports(prompt: String, candidate: String) -> String {
        let imports = prompt
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
            .filter { line in
                line.hasPrefix("import ") || line.hasPrefix("from ")
            }
        let missingImports = imports.filter { !candidate.contains($0) }
        guard !missingImports.isEmpty else {
            return candidate
        }
        return (missingImports + ["", candidate]).joined(separator: "\n")
    }

    // Three public OpenAI HumanEval tasks, kept as a tiny slice rather than a
    // leaderboard substitute. HumanEval is MIT licensed.
    static let humanEvalSlice = [
        CodeBenchmarkTask(
            taskID: "HumanEval/0",
            entryPoint: "has_close_elements",
            prompt: """
            from typing import List


            def has_close_elements(numbers: List[float], threshold: float) -> bool:
                \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
                given threshold.
                >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
                False
                >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
                True
                \"\"\"
            """,
            tests: """
            def check(candidate):
                assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
                assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
                assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
                assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
                assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
                assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
                assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False
            """
        ),
        CodeBenchmarkTask(
            taskID: "HumanEval/3",
            entryPoint: "below_zero",
            prompt: """
            from typing import List


            def below_zero(operations: List[int]) -> bool:
                \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with
                zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
                at that point function should return True. Otherwise it should return False.
                >>> below_zero([1, 2, 3])
                False
                >>> below_zero([1, 2, -4, 5])
                True
                \"\"\"
            """,
            tests: """
            def check(candidate):
                assert candidate([]) == False
                assert candidate([1, 2, -3, 1, 2, -3]) == False
                assert candidate([1, 2, -4, 5, 6]) == True
                assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False
                assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True
                assert candidate([1, -2, 2, -2, 5, -5, 4, -4]) == True
            """
        ),
        CodeBenchmarkTask(
            taskID: "HumanEval/8",
            entryPoint: "sum_product",
            prompt: """
            from typing import List, Tuple


            def sum_product(numbers: List[int]) -> Tuple[int, int]:
                \"\"\" For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
                Empty sum should be equal to 0 and empty product should be equal to 1.
                >>> sum_product([])
                (0, 1)
                >>> sum_product([1, 2, 3, 4])
                (10, 24)
                \"\"\"
            """,
            tests: """
            def check(candidate):
                assert candidate([]) == (0, 1)
                assert candidate([1, 1, 1]) == (3, 1)
                assert candidate([100, 0]) == (100, 0)
                assert candidate([3, 5, 7]) == (3 + 5 + 7, 3 * 5 * 7)
                assert candidate([10]) == (10, 10)
            """
        ),
    ]
}

private struct CodeBenchmarkPlan: Encodable {
    let suite: String
    let models: [String]
    let tasks: [String]
    let maxTokens: Int
    let temperature: Double
    let topP: Double
    let executionTimeout: Double
    let sandbox: String
}

private struct CodeBenchmarkReport: Encodable {
    let plan: CodeBenchmarkPlan
    let models: [CodeBenchmarkModelResult]

    func jsonString() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        return String(decoding: data, as: UTF8.self)
    }

    func renderText() -> String {
        var lines = [
            "Code benchmark",
            "suite: \(plan.suite)",
            "tasks: \(plan.tasks.joined(separator: ", "))",
            "models: \(plan.models.joined(separator: ", "))",
            "sampling: temperature=\(plan.temperature) top_p=\(plan.topP) max_tokens=\(plan.maxTokens)",
            "sandbox: \(plan.sandbox)",
            "",
        ]
        guard !models.isEmpty else {
            lines.append("dry run: no models loaded and no generated code executed")
            return lines.joined(separator: "\n")
        }
        lines.append("model                         engine       pass  avg_gen_s  avg_exec_s  avg_tps")
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

private struct CodeBenchmarkModelResult: Encodable {
    let model: String
    let engine: String?
    let modelPath: String?
    let status: String
    let error: String?
    let cases: [CodeBenchmarkCaseResult]

    static func missing(model: String, reason: String) -> CodeBenchmarkModelResult {
        CodeBenchmarkModelResult(
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

    func summaryLine() -> String {
        let passText = "\(passCount)/\(cases.count)"
        let avgGeneration = average(cases.compactMap(\.generationSeconds))
        let avgExecution = average(cases.compactMap(\.executionSeconds))
        let avgTPS = average(cases.compactMap(\.decodeTokensPerSecond))
        return [
            Self.padded(model, width: 29),
            Self.padded(engine ?? status, width: 12),
            Self.padded(passText, width: 5),
            Self.padded(Self.format(avgGeneration), width: 10),
            Self.padded(Self.format(avgExecution), width: 11),
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
                result.executionSeconds.map { "exec=\(Self.format($0))s" },
                result.sandboxBackend.map { "sandbox=\($0)" },
                result.decodeTokensPerSecond.map { "tps=\(Self.format($0))" },
                "tokens=\(result.tokensGenerated)",
                result.finishReason.map { "finish=\($0)" },
                result.reachedMaxTokens ? "capped=true" : nil,
            ]
                .compactMap { $0 }
                .joined(separator: " ")
            lines.append("  \(mark) \(result.taskID) \(result.entryPoint) \(timing)")
            if let error = result.error {
                lines.append("    \(error.replacingOccurrences(of: "\n", with: "\n    "))")
            }
        }
        return lines.joined(separator: "\n")
    }

    private static func average(_ values: [Double]) -> Double? {
        guard !values.isEmpty else {
            return nil
        }
        return values.reduce(0, +) / Double(values.count)
    }

    private func average(_ values: [Double]) -> Double? {
        Self.average(values)
    }

    private static func format(_ value: Double?) -> String {
        guard let value else {
            return "-"
        }
        return String(format: "%.2f", value)
    }

    private static func padded(_ value: String, width: Int) -> String {
        let padding = max(0, width - value.count)
        return value + String(repeating: " ", count: padding)
    }
}

private struct CodeBenchmarkCaseResult: Encodable {
    let taskID: String
    let entryPoint: String
    let passed: Bool
    let generationSeconds: Double
    let executionSeconds: Double?
    let sandboxBackend: String?
    let tokensGenerated: Int
    let decodeTokensPerSecond: Double?
    let finishReason: String?
    let reachedMaxTokens: Bool
    let error: String?
}

private extension ChatResponse {
    func reachedMaxTokens(limit: Int) -> Bool {
        if finishReason == .length {
            return true
        }
        guard finishReason == nil else {
            return false
        }
        return tokensGenerated >= limit
    }

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
