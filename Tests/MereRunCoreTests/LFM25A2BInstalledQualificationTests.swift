import Foundation
import MLX
import MLXNN
import MereRunAdmission
import XCTest
@testable import MereRunCore

/// Real-checkpoint qualification is opt-in and retains the shared admission floor.
final class LFM25A2BInstalledQualificationTests: MereRunCoreTestCase {
    private struct Suite: Decodable {
        let cases: [Case]
    }

    private struct Case: Decodable {
        struct Message: Decodable {
            let role: String
            let content: String
        }
        struct Tool: Decodable {
            struct Function: Decodable {
                let name: String
                let description: String
                let parameters: [String: OpenAIJSONValue]

                var definition: ToolDefinition {
                    ToolDefinition(name: name, description: description, parameterSchema: parameters)
                }
            }
            let function: Function
        }
        let id: String
        let split: String?
        let task: String?
        let tools: [Tool]?
        let messages: [Message]
        let completion: String
        let renderedPrompt: String
        let tokens: [Int]
        let scoreStart: Int

        enum CodingKeys: String, CodingKey {
            case id, split, task, tools, messages, completion, tokens
            case renderedPrompt = "rendered_prompt"
            case scoreStart = "score_start"
        }
    }

    private struct CaseReceipt: Encodable {
        let id: String
        let scoredTokens: Int
        let activeMLXBytes: Int
        let peakMLXBytes: Int
        let cacheMLXBytes: Int
        let physicalFootprintBytes: UInt64?
        let elapsedSeconds: TimeInterval
    }

    private struct CheckpointReceipt: Encodable {
        let modelPath: String
        let physicalMemoryBytes: UInt64
        let availableMemoryBeforeBytes: UInt64?
        let cases: [CaseReceipt]
    }

    private func verifyTokenizer(root: URL, config: LFM2Config, suite: Suite) async throws {
        let tokenizer = try await LFM2TokenizerAndTemplate.load(from: root, maxLengthOverride: config.maxPositionEmbeddings)
        // Tokenizer/template parity is checked before any trained tensor is evaluated.
        for row in suite.cases {
            let messages = try row.messages.map { item in
                ChatMessage(role: try XCTUnwrap(ChatMessage.Role(rawValue: item.role)), content: item.content)
            }
            let prompt = try LFM2TokenizerAndTemplate.renderPrompt(
                messages: messages, tools: row.tools?.map { $0.function.definition },
                includeThinking: true, generationPromptSuffix: tokenizer.generationPromptSuffix
            )
            XCTAssertEqual(prompt, row.renderedPrompt, row.id)
            XCTAssertEqual(tokenizer.encodeRaw(prompt + row.completion), row.tokens, row.id)
        }
        for fixture in try LFM25ToolTemplateFixture.load() {
            let tokens = try tokenizer.encodeForGeneration(
                messages: fixture.messages, tools: LFM25ToolTemplateFixture.tools,
                includeThinking: false, maxLength: config.maxPositionEmbeddings
            )
            XCTAssertEqual(tokens, fixture.tokens, fixture.id.rawValue)
        }
    }

    func testInstalledTokenizerMatchesFrozenSuiteWithoutLoadingWeights() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let rootPath = env["MERERUN_LFM25_A2B_QUALIFICATION_ROOT"],
              let suitePath = env["MERERUN_LFM25_A2B_QUALIFICATION_SUITE"] else {
            throw XCTSkip("Provide the local A2B tokenizer and frozen suite.")
        }
        let root = URL(fileURLWithPath: rootPath)
        let config = try JSONDecoder().decode(LFM2Config.self, from: Data(contentsOf: root.appending(path: "config.json")))
        let suite = try JSONDecoder().decode(Suite.self, from: Data(contentsOf: URL(fileURLWithPath: suitePath)))
        try await verifyTokenizer(root: root, config: config, suite: suite)
    }

    func testInstalledMixedCheckpointTokenizerAndFullVocabularyLogits() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let rootPath = env["MERERUN_LFM25_A2B_QUALIFICATION_ROOT"],
              let suitePath = env["MERERUN_LFM25_A2B_QUALIFICATION_SUITE"],
              let outputPath = env["MERERUN_LFM25_A2B_QUALIFICATION_OUTPUT"] else {
            throw XCTSkip("Provide an installed A2B mixed checkpoint, frozen suite, and output folder.")
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Set MERERUN_TEST_MLX_DEVICE=gpu for trained-checkpoint qualification.")
        }
        let indexURL = URL(fileURLWithPath: rootPath).appending(path: "model.safetensors.index.json")
        let coordinator = MachineInferenceCoordinator(
            stateDirectory: MereRunModelPaths.applicationSupportBase.appending(path: "admission"),
            hostSnapshot: {
                MachineInferenceCoordinator.currentHostSnapshot(
                    diskURL: MereRunModelPaths.applicationSupportBase.deletingLastPathComponent()
                )
            }
        )
        let lease = try await coordinator.acquire(.init(label: "LFM2.5 A2B checkpoint qualification", resourceClass: .standard))
        defer { lease.release() }
        let root = URL(fileURLWithPath: rootPath)
        let output = URL(fileURLWithPath: outputPath)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let config = try JSONDecoder().decode(LFM2Config.self, from: Data(contentsOf: root.appending(path: "config.json")))
        XCTAssertTrue(config.isA2B24B)
        XCTAssertEqual(config.moduleQuantization.count, 114)
        let suite = try JSONDecoder().decode(Suite.self, from: Data(contentsOf: URL(fileURLWithPath: suitePath)))
        try await verifyTokenizer(root: root, config: config, suite: suite)
        let before = RuntimeMemorySample.current()
        let started = ProcessInfo.processInfo.systemUptime
        MLX.Memory.peakMemory = 0
        var cases: [CaseReceipt] = []
        let model = LFM2Model(config: config)
        try HFSafetensorsWeightsLoader.applyQuantizedWeights(
            indexURL: root.appending(path: "model.safetensors.index.json"), to: model,
            groupSize: try XCTUnwrap(config.quantization).groupSize,
            bits: try XCTUnwrap(config.quantization).bits, mapper: LFM2Resources.mapWeight(key:value:)
        )
        for row in suite.cases {
            let pressure = RuntimeMemoryPressurePolicy.default.pressure(for: .current())
            guard pressure != .critical else {
                XCTFail("Checkpoint qualification reached critical memory pressure before " + row.id)
                return
            }
            let input = MLXArray(row.tokens.dropLast().map(Int32.init), [1, row.tokens.count - 1])
            let hidden = model.model(input, cache: nil)
            let logits = model.logits(from: hidden)[0, (row.scoreStart - 1)..., 0...].asType(.float32)
            MLX.eval(logits)
            XCTAssertEqual(logits.shape, [row.tokens.count - row.scoreStart, 128_000])
            XCTAssertTrue(MLX.all(isFinite(logits)).item(Bool.self), row.id)
            try MLX.save(arrays: ["logits": logits], url: output.appending(path: row.id + ".safetensors"))
            cases.append(CaseReceipt(
                id: row.id, scoredTokens: row.tokens.count - row.scoreStart,
                activeMLXBytes: MLX.Memory.activeMemory, peakMLXBytes: MLX.Memory.peakMemory,
                cacheMLXBytes: MLX.Memory.cacheMemory,
                physicalFootprintBytes: RuntimeMemorySample.current().physicalFootprintBytes,
                elapsedSeconds: ProcessInfo.processInfo.systemUptime - started
            ))
            let receipt = CheckpointReceipt(
                modelPath: root.path, physicalMemoryBytes: before.physicalBytes,
                availableMemoryBeforeBytes: before.availableBytes, cases: cases
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(receipt).write(to: output.appending(path: "qualification-receipt.json"), options: .atomic)
            MLX.Memory.clearCache()
            print("LFM2.5 native full-logit case completed: " + row.id)
        }
    }

    private struct GenerationCaseReceipt: Encodable {
        struct Call: Encodable {
            let name: String
            let arguments: [String: String]
        }
        let id: String
        let text: String
        let generatedTokens: Int
        let elapsedSeconds: TimeInterval
        let peakMLXBytes: Int
        let finishReason: String?
        let toolCalls: [Call]
    }

    private struct GenerationReceipt: Encodable {
        let modelPath: String
        let temperature: Double
        let maximumNewTokens: Int
        let kvBits: Int
        let cases: [GenerationCaseReceipt]
    }

    func testInstalledCodingAndToolGeneration() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let rootPath = env["MERERUN_LFM25_A2B_QUALIFICATION_ROOT"],
              let suitePath = env["MERERUN_LFM25_A2B_QUALIFICATION_SUITE"],
              let outputPath = env["MERERUN_LFM25_A2B_QUALIFICATION_GENERATIONS"] else {
            throw XCTSkip("Provide an installed A2B checkpoint, coding/tool suite, and generation receipt path.")
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Set MERERUN_TEST_MLX_DEVICE=gpu for trained-checkpoint generation.")
        }
        let maximumNewTokens = try env["MERERUN_LFM25_A2B_QUALIFICATION_MAX_TOKENS"].map {
            try XCTUnwrap(Int($0), "Provide an integer qualification output-token budget.")
        } ?? 2_048
        let coordinator = MachineInferenceCoordinator(
            stateDirectory: MereRunModelPaths.applicationSupportBase.appending(path: "admission"),
            hostSnapshot: {
                MachineInferenceCoordinator.currentHostSnapshot(
                    diskURL: MereRunModelPaths.applicationSupportBase.deletingLastPathComponent()
                )
            }
        )
        let lease = try await coordinator.acquire(.init(label: "LFM2.5 coding/tool qualification", resourceClass: .standard))
        defer { lease.release() }
        let root = URL(fileURLWithPath: rootPath)
        let config = try JSONDecoder().decode(LFM2Config.self, from: Data(contentsOf: root.appending(path: "config.json")))
        let suite = try JSONDecoder().decode(Suite.self, from: Data(contentsOf: URL(fileURLWithPath: suitePath)))
        try await verifyTokenizer(root: root, config: config, suite: suite)
        let generator = LFM2Generator(modelId: rootPath, prefixKVCacheEnabled: false, continuousBatchingEnabled: false)
        var cases: [GenerationCaseReceipt] = []
        for row in suite.cases where row.split == "heldout" && ["coding", "tool-use"].contains(row.task ?? "") {
            let messages = try row.messages.map { item in
                ChatMessage(role: try XCTUnwrap(ChatMessage.Role(rawValue: item.role)), content: item.content)
            }
            let request = ChatRequest(
                messages: messages, maxTokens: maximumNewTokens, temperature: 0, topP: 1,
                showThinking: true, tools: row.tools?.map { $0.function.definition },
                kvCacheMode: .affine8, maxContextTokens: 8_192
            )
            let started = ProcessInfo.processInfo.systemUptime
            MLX.Memory.peakMemory = 0
            let response = try await generator.chat(request, modelPath: rootPath, progressHandler: nil)
            cases.append(GenerationCaseReceipt(
                id: row.id, text: response.response, generatedTokens: response.tokensGenerated,
                elapsedSeconds: ProcessInfo.processInfo.systemUptime - started,
                peakMLXBytes: MLX.Memory.peakMemory, finishReason: response.finishReason?.rawValue,
                toolCalls: (response.toolCalls ?? []).map { .init(name: $0.name, arguments: $0.arguments) }
            ))
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.keyEncodingStrategy = .convertToSnakeCase
            try encoder.encode(GenerationReceipt(
                modelPath: rootPath, temperature: 0, maximumNewTokens: maximumNewTokens, kvBits: 8, cases: cases
            )).write(to: URL(fileURLWithPath: outputPath), options: .atomic)
            MLX.Memory.clearCache()
        }
        await generator.unload()
        XCTAssertEqual(cases.count, 24)
    }
}
