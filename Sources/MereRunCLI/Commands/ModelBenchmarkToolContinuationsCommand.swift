import ArgumentParser
import Foundation
import MereRunCore

struct ModelBenchmarkToolContinuations: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tool-continuations",
        abstract: "Evaluate Gemma 4 continuation after completed tool calls."
    )

    @Option(name: [.long], help: "Managed Gemma 4 model id.")
    var model: String = Gemma4Resources.twelveB4BitModelId

    @Option(name: [.long], help: "Override the Gemma 4 model root.")
    var modelRoot: String?

    @Option(name: [.long], help: "Maximum generated tokens per case.")
    var maxTokens: Int = 96

    @Option(name: [.long], help: "Maximum context tokens passed to the runtime.")
    var contextSize: Int?

    @Flag(name: [.long], help: "Print the benchmark plan without loading the model.")
    var dryRun: Bool = false

    @Flag(name: [.long], help: "Include model responses in the report.")
    var logResponses: Bool = false

    @Flag(name: [.long], help: "Emit machine-readable JSON.")
    var json: Bool = false

    func validate() throws {
        guard maxTokens > 0 else {
            throw ValidationError("--max-tokens must be greater than zero.")
        }
        if let contextSize {
            guard contextSize > 0 else {
                throw ValidationError("--context-size must be greater than zero.")
            }
        }
        guard Gemma4Resources.handles(modelSpec: model) else {
            throw ValidationError("--model must identify a Gemma 4 model.")
        }
        if let modelRoot {
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: modelRoot, isDirectory: &isDirectory),
                  isDirectory.boolValue else {
                throw ValidationError("--model-root must be an existing directory.")
            }
        }
    }

    func run() async throws {
        let plan = ToolContinuationBenchmarkPlan(
            model: model,
            modelRoot: modelRoot,
            cases: ToolContinuationBenchmarkCase.cases.map(\.caseID),
            maxTokens: maxTokens,
            contextSize: contextSize
        )
        if dryRun {
            try printReport(ToolContinuationBenchmarkReport(plan: plan, result: nil))
            return
        }

        try MLXBundleSupport.ensureAvailable(quiet: json)
        let resolvedRoot = try resolveModelRoot()
        let generator = Gemma4Generator(modelId: model)
        var caseResults: [ToolContinuationBenchmarkCaseResult] = []
        for benchmarkCase in ToolContinuationBenchmarkCase.cases {
            caseResults.append(
                await runCase(
                    benchmarkCase,
                    modelPath: resolvedRoot.path,
                    generator: generator
                )
            )
        }
        await generator.unload()

        let result = ToolContinuationBenchmarkModelResult(
            model: model,
            modelPath: resolvedRoot.path,
            cases: caseResults
        )
        try printReport(ToolContinuationBenchmarkReport(plan: plan, result: result))
    }

    private func resolveModelRoot() throws -> URL {
        if let modelRoot {
            return URL(fileURLWithPath: modelRoot, isDirectory: true).standardizedFileURL
        }
        guard let installed = ManagedModelResolver.resolveInstalledModel(id: model) else {
            throw ValidationError(
                "Model is not installed. Run `\(CLICommandDisplay.modelPullCommand(for: model))` first."
            )
        }
        return installed
    }

    private func runCase(
        _ benchmarkCase: ToolContinuationBenchmarkCase,
        modelPath: String,
        generator: Gemma4Generator
    ) async -> ToolContinuationBenchmarkCaseResult {
        let request = ChatRequest(
            messages: benchmarkCase.messages,
            maxTokens: maxTokens,
            temperature: 0,
            topP: 1,
            showThinking: false,
            stopOnEOS: true,
            maxContextTokens: contextSize
        )
        let start = Date()
        do {
            let response = try await generator.chat(
                request,
                modelPath: modelPath,
                progressHandler: nil
            )
            let elapsed = Date().timeIntervalSince(start)
            let evaluation = benchmarkCase.evaluate(response)
            return ToolContinuationBenchmarkCaseResult(
                caseID: benchmarkCase.caseID,
                passed: evaluation.failedChecks.isEmpty,
                failedChecks: evaluation.failedChecks,
                generationSeconds: elapsed,
                promptTokens: response.promptTokens,
                completionTokens: response.tokensGenerated,
                response: logResponses ? response.response : nil,
                error: nil
            )
        } catch {
            return ToolContinuationBenchmarkCaseResult(
                caseID: benchmarkCase.caseID,
                passed: false,
                failedChecks: ["generation failed"],
                generationSeconds: Date().timeIntervalSince(start),
                promptTokens: nil,
                completionTokens: 0,
                response: nil,
                error: String(describing: error)
            )
        }
    }

    private func printReport(_ report: ToolContinuationBenchmarkReport) throws {
        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            print(String(decoding: try encoder.encode(report), as: UTF8.self))
        } else {
            print(report.renderText())
        }
    }
}

private struct ToolContinuationBenchmarkCase {
    let caseID: String
    let messages: [ChatMessage]
    let requiredFragments: [String]
    let forbiddenFragments: [String]

    func evaluate(_ response: ChatResponse) -> ToolContinuationBenchmarkEvaluation {
        let normalized = response.response.lowercased()
        var failures: [String] = []
        for fragment in requiredFragments where !normalized.contains(fragment.lowercased()) {
            failures.append("missing response fragment: \(fragment)")
        }
        for fragment in forbiddenFragments where normalized.contains(fragment.lowercased()) {
            failures.append("forbidden response fragment: \(fragment)")
        }
        if response.toolCalls?.isEmpty == false {
            failures.append("model called another tool after completed tool results")
        }
        return ToolContinuationBenchmarkEvaluation(failedChecks: failures)
    }

    static let cases = [
        ToolContinuationBenchmarkCase(
            caseID: "Gemma4ToolContinuation/0",
            messages: [
                ChatMessage(
                    role: .system,
                    content: "Use completed tool results as authoritative. Do not call another tool. Answer briefly."
                ),
                ChatMessage(role: .user, content: "Check order A-17 and tell me its status."),
                ChatMessage(
                    role: .assistant,
                    content: "",
                    reasoningContent: "I need the order record.",
                    toolCalls: [
                        ChatMessageToolCall(
                            id: "call_order",
                            name: "lookup_order",
                            arguments: [
                                "order_id": .string("A-17"),
                                "include_eta": .bool(false),
                                "metadata": .object(["source": .null]),
                            ]
                        ),
                    ]
                ),
                ChatMessage(
                    role: .tool,
                    content: #"{"order_id":"A-17","status":"ready for pickup"}"#,
                    name: "lookup_order",
                    toolCallID: "call_order"
                ),
            ],
            requiredFragments: ["A-17", "ready for pickup"],
            forbiddenFragments: ["shipped", "<|tool_call>"]
        ),
        ToolContinuationBenchmarkCase(
            caseID: "Gemma4ToolContinuation/1",
            messages: [
                ChatMessage(
                    role: .system,
                    content: "Use completed tool results as authoritative. Do not call another tool. Answer briefly."
                ),
                ChatMessage(role: .user, content: "What is the Halifax weather in Fahrenheit?"),
                ChatMessage(
                    role: .assistant,
                    content: "",
                    reasoningContent: "First I need current weather.",
                    toolCalls: [
                        ChatMessageToolCall(
                            id: "call_weather",
                            name: "get_weather",
                            arguments: [
                                "city": .string("Halifax"),
                                "units": .string("celsius"),
                            ]
                        ),
                    ]
                ),
                ChatMessage(
                    role: .tool,
                    content: #"{"temperature_c":10,"condition":"rainy"}"#,
                    name: "get_weather",
                    toolCallID: "call_weather"
                ),
                ChatMessage(
                    role: .assistant,
                    content: "",
                    reasoningContent: "Now convert 10 Celsius.",
                    toolCalls: [
                        ChatMessageToolCall(
                            id: "call_convert",
                            name: "convert_temperature",
                            arguments: [
                                "value": .number(10),
                                "from": .string("celsius"),
                                "to": .string("fahrenheit"),
                            ]
                        ),
                    ]
                ),
                ChatMessage(
                    role: .tool,
                    content: #"{"value":50}"#,
                    name: "convert_temperature",
                    toolCallID: "call_convert"
                ),
            ],
            requiredFragments: ["Halifax", "50", "rain"],
            forbiddenFragments: ["<|tool_call>"]
        ),
    ]
}

private struct ToolContinuationBenchmarkEvaluation {
    let failedChecks: [String]
}

private struct ToolContinuationBenchmarkPlan: Encodable {
    let model: String
    let modelRoot: String?
    let cases: [String]
    let maxTokens: Int
    let contextSize: Int?
}

private struct ToolContinuationBenchmarkReport: Encodable {
    let plan: ToolContinuationBenchmarkPlan
    let result: ToolContinuationBenchmarkModelResult?

    func renderText() -> String {
        var lines = [
            "Gemma 4 tool-continuation benchmark",
            "model: \(plan.model)",
            "cases: \(plan.cases.joined(separator: ", "))",
        ]
        guard let result else {
            lines.append("dry run: no model loaded and no generations run")
            return lines.joined(separator: "\n")
        }
        lines.append("result: \(result.passCount)/\(result.cases.count) cases passed")
        for benchmarkCase in result.cases {
            lines.append(benchmarkCase.renderText())
        }
        return lines.joined(separator: "\n")
    }
}

private struct ToolContinuationBenchmarkModelResult: Encodable {
    let model: String
    let modelPath: String
    let cases: [ToolContinuationBenchmarkCaseResult]

    var passCount: Int {
        cases.filter(\.passed).count
    }
}

private struct ToolContinuationBenchmarkCaseResult: Encodable {
    let caseID: String
    let passed: Bool
    let failedChecks: [String]
    let generationSeconds: Double
    let promptTokens: Int?
    let completionTokens: Int
    let response: String?
    let error: String?

    func renderText() -> String {
        let status = passed ? "pass" : "fail"
        var line = "  \(status) \(caseID) gen=\(String(format: "%.2f", generationSeconds))s"
        if !failedChecks.isEmpty {
            line += " [\(failedChecks.joined(separator: "; "))]"
        }
        if let error {
            line += " error=\(error)"
        }
        if let response {
            line += " response=\(response.replacingOccurrences(of: "\n", with: " "))"
        }
        return line
    }
}
