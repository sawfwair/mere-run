import Foundation
import MLX
import MereRunInkling2Model
import XCTest
@testable import MereRunCore

final class Inkling2MiniLoRATests: MereRunCoreTestCase {
    func testTrainSaveReloadAndScaleOnLoopedDecoder() async throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMiniFixture(to: root)
        let data = root.appendingPathComponent("train.jsonl")
        try Data(#"{"sources":["fixture"],"messages":[{"role":"system","content":"ssss"},{"role":"user","content":"uuuu"},{"role":"assistant","content":"aaaa"}]}"#.utf8).write(to: data)
        let output = root.appendingPathComponent("mini.safetensors")
        let options = TextLoRATrainingOptions(data: data.path, output: output.path, model: root.path,
                                               eval: data.path, trainingSteps: 3, batchSize: 2, learningRate: 0.1, rank: 2)
        let result = try await TextLoRATrainingOperation.execute(.resolve(options))
        let report = try XCTUnwrap(result.report)
        XCTAssertEqual(result.manifest.lora.targetModules, ["unembed"])
        XCTAssertEqual(report.layerCount, 1)
        XCTAssertEqual(report.evaluationTargetTokenCount, 4)
        XCTAssertLessThan(try XCTUnwrap(report.finalEvaluationLoss), try XCTUnwrap(report.initialEvaluationLoss))

        let config = try Inkling2Configuration.decode(Data(contentsOf: root.appendingPathComponent("config.json")))
        let model = try Inkling2Model(config: config, weights: Inkling2Weights(root: root))
        let tokens: [Int32] = [3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2]
        let input = try model.embeddings(MLXArray(tokens).reshaped(1, -1))
        let base = try model.forward(embeddings: input, cache: model.makeCache(), lastPositionOnly: false)
        let generator = Inkling2Generator()
        for scale in [1.0, 0.0] {
            let adapter = try Inkling2MiniLoRALayer.load(from: output, for: config, scale: scale)
            let adapted = try model.forward(embeddings: input, cache: model.makeCache(), lastPositionOnly: false, outputDelta: adapter.callAsFunction)
            XCTAssertEqual(arrayEqual(adapted, base).item(Bool.self), scale == 0)
            let request = ChatRequest(messages: [.init(role: .system, content: "ssss"), .init(role: .user, content: "uuuu")],
                                      maxTokens: 2, temperature: 0, lora: scale == 0 ? nil : .local(path: output.path, scale: scale),
                                      stopOnEOS: false, logprobCapture: .tokens)
            let response = try await generator.chat(request, modelPath: root.path)
            let generated = try XCTUnwrap(response.logprobs?.tokens)
            XCTAssertEqual(generated.count, 2)
            var history = Array(tokens.prefix(8))
            for token in generated {
                let hidden = try model.embeddings(MLXArray(history).reshaped(1, -1))
                let logits = try model.forward(embeddings: hidden, cache: model.makeCache(), outputDelta: adapter.callAsFunction).reshaped(-1)
                let logprob = (logits[token.tokenID] - logSumExp(logits)).item(Float.self)
                XCTAssertEqual(token.rawLogprob, Double(logprob), accuracy: 0.002)
                history.append(Int32(token.tokenID))
            }
            if scale == 1 {
                let loss = TextSFTTrainingLoss.gatheredNextTokenCrossEntropy(logits: adapted[0, 7..., 0...], labels: MLXArray([Int32](repeating: 2, count: 4)))
                XCTAssertEqual(loss.item(Float.self), try XCTUnwrap(report.finalEvaluationLoss), accuracy: 0.002)
            }
        }
    }

    private func writeMiniFixture(to root: URL) throws {
        struct Tensor: Decodable { let shape: [Int]; let values: [Float] }
        struct Mini: Decodable { let weights: [String: Tensor] }
        let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Inkling2RuntimeTests/Fixtures")
        try FileManager.default.copyItem(at: fixtures.appendingPathComponent("tiny-mini-config.json"), to: root.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode([String: Tensor].self, from: Data(contentsOf: fixtures.appendingPathComponent("weights.json")))
        let mini = try JSONDecoder().decode(Mini.self, from: Data(contentsOf: fixtures.appendingPathComponent("mini-reference.json")))
        let tensors = base.filter { $0.key.hasPrefix("model.llm.") }.merging(mini.weights) { _, new in new }
        let shard = "model.safetensors"
        try MLX.save(arrays: tensors.mapValues { MLXArray($0.values, $0.shape) }, url: root.appendingPathComponent(shard))
        try JSONEncoder().encode(["weight_map": tensors.mapValues { _ in shard }]).write(to: root.appendingPathComponent("model.safetensors.index.json"))
        try Data(#"{"model":{"type":"BPE","unk_token":"[UNK]","vocab":{"[UNK]":0,"u":1,"a":2,"s":3},"merges":[]},"decoder":{"type":"Fuse"}}"#.utf8)
            .write(to: root.appendingPathComponent("tokenizer.json"))
        try Data(#"{"tokenizer_class":"Qwen2Tokenizer","pad_token":"[UNK]","model_max_length":32}"#.utf8)
            .write(to: root.appendingPathComponent("tokenizer_config.json"))
        try Data("{% for message in messages %}{{ message.content }}{% endfor %}".utf8)
            .write(to: root.appendingPathComponent("chat_template.jinja"))
    }
}
