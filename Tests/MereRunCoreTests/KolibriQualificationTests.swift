import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunCore

final class KolibriQualificationTests: MLXTestCase {
    private func checkpoint() throws -> URL {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("KolibriRuntimeTests/Fixtures")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("kolibri-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: fixture.appendingPathComponent("config.json"),
                                         to: root.appendingPathComponent("config.json"))
        var arrays = try MLX.loadArrays(url: fixture.appendingPathComponent("weights.safetensors"))
        for key in arrays.keys where !key.hasSuffix("e_score_correction_bias") { arrays[key] = arrays[key]?.asType(.bfloat16) }
        try MLX.save(arrays: arrays, url: root.appendingPathComponent("weights.safetensors"))
        struct Index: Encodable { let weight_map: [String: String] }
        try JSONEncoder().encode(Index(weight_map: arrays.mapValues { _ in "weights.safetensors" }))
            .write(to: root.appendingPathComponent("model.safetensors.index.json"))
        return root
    }

    func testLoaderRefusesWrongDtypeAndIncompleteShardOwnership() throws {
        let root = try checkpoint()
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try KolibriLoader.load(root: root)
        let weightsURL = root.appendingPathComponent("weights.safetensors")
        var arrays = try MLX.loadArrays(url: weightsURL)
        arrays["model.norm.weight"] = arrays["model.norm.weight"]?.asType(.float32)
        eval(Array(arrays.values))
        try MLX.save(arrays: arrays, url: weightsURL)
        XCTAssertThrowsError(try KolibriLoader.load(root: root))
        arrays.removeValue(forKey: "model.norm.weight")
        eval(Array(arrays.values))
        try MLX.save(arrays: arrays, url: weightsURL)
        XCTAssertThrowsError(try KolibriLoader.load(root: root))
    }

    func testScoringUsesContinuationBoundaryAndOnlyCalibratesCalibrationCases() throws {
        let root = try checkpoint()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = root.appendingPathComponent("suite.json")
        try Data("""
        [{"id":"cal","language":"en","task":"test","split":"calibration","tokens":[3,11,7,23,41,19],"scoreStart":3},
         {"id":"held","language":"de","task":"test","split":"heldout","tokens":[61,37,2,17,53],"scoreStart":2}]
        """.utf8).write(to: suite)
        let output = root.appendingPathComponent("results")
        let calibration = root.appendingPathComponent("moments.safetensors")
        try KolibriLogprobBenchmark.run(modelRoot: root, suite: suite, output: output, chunkSize: 2,
                                       calibrationOutput: calibration)
        let logits = try MLX.loadArrays(url: output.appendingPathComponent("cal.safetensors"))
        XCTAssertEqual(logits["logits"]?.shape, [3, 97])
        let receipt = try JSONDecoder().decode(KolibriLogprobBenchmark.Receipt.self,
                                               from: Data(contentsOf: output.appendingPathComponent("receipt.json")))
        XCTAssertEqual(receipt.cases.map(\.tokenCount), [3, 3])
        XCTAssertTrue(receipt.cases.allSatisfy { $0.meanNegativeLogLikelihood.isFinite && $0.logitsSHA256.count == 64 })
        let moments = try MLX.loadArrays(url: calibration)
        XCTAssertEqual(moments.count, 6)
        XCTAssertTrue(moments.values.allSatisfy { $0.shape == [64] })
        XCTAssertThrowsError(try KolibriLogprobBenchmark.run(modelRoot: root, suite: suite, output: output))
    }

    func testChatTemplateToolHistoryAndGeneratedTokenLogprobsWhenFixtureProvided() async throws {
        guard let root = ProcessInfo.processInfo.environment["MERERUN_KOLIBRI_CHAT_FIXTURE"] else {
            throw XCTSkip("Provide the small fixture from prepare_kolibri_chat_fixture.py; it uses synthetic weights and the pinned tokenizer.")
        }
        let generator = KolibriGenerator()
        try await generator.prepare(modelPath: root)
        let histories: [[ChatMessage]] = [
            [.init(role: .user, content: "Say hello.")],
            [.init(role: .user, content: "Check Halifax weather."),
             .init(role: .assistant, content: "", toolCalls: [.init(id: "call-1", name: "get_weather", arguments: ["city": .string("Halifax")])]),
             .init(role: .tool, content: "12 degrees Celsius", name: "get_weather", toolCallID: "call-1"),
             .init(role: .user, content: "Summarize briefly.")]
        ]
        for messages in histories {
            let request = ChatRequest(messages: messages, maxTokens: 3, temperature: 0, showThinking: false,
                                      stopOnEOS: false, maxContextTokens: 512, logprobCapture: .top(3))
            let response = try await generator.chat(request, progressHandler: nil)
            XCTAssertEqual(response.tokensGenerated, 3)
            XCTAssertGreaterThan(response.promptTokens ?? 0, 0)
            let tokens = try XCTUnwrap(response.logprobs?.tokens)
            XCTAssertEqual(tokens.count, 3)
            XCTAssertTrue(tokens.allSatisfy { $0.rawLogprob.isFinite && $0.rawEntropy.isFinite && $0.topLogprobs.count == 3 })
        }
        await generator.unload()
    }
}
