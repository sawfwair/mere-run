import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class TextLoRATrainerTests: MereRunCoreTestCase {
    func testTrainingOrderIsSeededAndCoversEpochs() {
        let first = TextLoRATrainer.makeTrainingOrder(exampleCount: 5, drawCount: 7, seed: 42)
        let second = TextLoRATrainer.makeTrainingOrder(exampleCount: 5, drawCount: 7, seed: 42)
        let different = TextLoRATrainer.makeTrainingOrder(exampleCount: 5, drawCount: 7, seed: 43)

        XCTAssertEqual(first, second)
        XCTAssertNotEqual(first, different)
        XCTAssertEqual(Set(first.prefix(5)), Set(0..<5))
        XCTAssertEqual(first.count, 7)
    }

    func testNativeTrainerUpdatesLoRAAndWritesAdapter() throws {
        let model = TinyCausalLM()
        let layers = try Gemma4TextLoRAInjector.inject(into: model, rank: 2, targetSuffixes: ["proj"])
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
            .appendingPathComponent("tiny-text-lora.safetensors")

        let report = try TextLoRATrainer.train(
            model: model,
            loraLayers: layers,
            examples: [
                TextSFTTokenizedExample(
                    inputTokenIds: [0, 1, 2],
                    labelTokenIds: [1, 2, 3],
                    lossMask: [0, 1, 1]
                ),
            ],
            config: TextLoRATrainingConfig(
                trainingSteps: 3,
                batchSize: 1,
                learningRate: 0.05
            ),
            outputURL: outputURL,
            metadata: ["format": "mererun.gemma4.text-lora.test"]
        ) { model, inputIds in
            model(inputIds)
        }

        XCTAssertEqual(report.steps, 3)
        XCTAssertEqual(report.layerCount, 1)
        XCTAssertNotNil(report.initialLoss)
        XCTAssertNotNil(report.finalLoss)
        XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: outputURL.deletingPathExtension().appendingPathExtension("loss").appendingPathExtension("csv").path
            )
        )
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: outputURL.deletingPathExtension().appendingPathExtension("loss").appendingPathExtension("html").path
            )
        )

        let layer = try XCTUnwrap(layers["proj"])
        let upNorm = MLX.sqrt((layer.loraUp * layer.loraUp).sum()).item(Float.self)
        XCTAssertGreaterThan(upNorm, 0)
    }

    func testNativeTrainerReportsProgress() throws {
        let model = TinyCausalLM()
        let layers = try Gemma4TextLoRAInjector.inject(into: model, rank: 2, targetSuffixes: ["proj"])
        let recorder = TextTrainingProgressRecorder()

        _ = try TextLoRATrainer.train(
            model: model,
            loraLayers: layers,
            examples: [
                TextSFTTokenizedExample(
                    inputTokenIds: [0, 1, 2],
                    labelTokenIds: [1, 2, 3],
                    lossMask: [0, 1, 1]
                ),
            ],
            config: TextLoRATrainingConfig(
                trainingSteps: 2,
                batchSize: 1,
                learningRate: 0.05
            ),
            progressHandler: { progress in
                switch progress.stage {
                case .training(let step, _, _):
                    recorder.record(step: step)
                case .saving:
                    recorder.recordSaving()
                }
            }
        ) { model, inputIds in
            model(inputIds)
        }

        XCTAssertEqual(recorder.steps, [1, 2])
        XCTAssertTrue(recorder.savingSeen)
    }

    func testNativeTrainerRunsOnTinyGemma4TextModel() throws {
        let config = try tinyGemma4TextConfig()
        let model = Gemma4TextCausalLM(config: config)
        let layers = try Gemma4TextLoRAInjector.inject(into: model, rank: 2)

        let report = try TextLoRATrainer.train(
            model: model,
            loraLayers: layers,
            examples: [
                TextSFTTokenizedExample(
                    inputTokenIds: [3, 4, 5],
                    labelTokenIds: [4, 5, 6],
                    lossMask: [0, 1, 1]
                ),
            ],
            config: TextLoRATrainingConfig(
                trainingSteps: 2,
                batchSize: 1,
                learningRate: 0.01
            )
        ) { model, inputIds in
            model(inputIds)
        }

        XCTAssertEqual(report.steps, 2)
        XCTAssertGreaterThan(report.layerCount, 0)
        XCTAssertNotNil(report.finalLoss)
    }

    private func tinyGemma4TextConfig() throws -> Gemma4TextConfig {
        let object: [String: Any] = [
            "model_type": "gemma4_text",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 128,
            "rms_norm_eps": 0.000001,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
            "hidden_size_per_layer_input": 0,
            "rope_parameters": [
                "sliding_attention": [
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                ],
                "full_attention": [
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 1.0,
                ],
            ],
            "sliding_window": 32,
            "layer_types": ["full_attention"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attention_k_eq_v": false,
            "use_double_wide_mlp": false,
            "enable_moe_block": false,
            "num_kv_shared_layers": 0,
            "tie_word_embeddings": true,
        ]
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        return try JSONDecoder().decode(Gemma4TextConfig.self, from: data)
    }
}

private final class TextTrainingProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recordedSteps: [Int] = []
    private var recordedSaving = false

    var steps: [Int] {
        lock.lock()
        defer { lock.unlock() }
        return recordedSteps
    }

    var savingSeen: Bool {
        lock.lock()
        defer { lock.unlock() }
        return recordedSaving
    }

    func record(step: Int) {
        lock.lock()
        defer { lock.unlock() }
        recordedSteps.append(step)
    }

    func recordSaving() {
        lock.lock()
        defer { lock.unlock() }
        recordedSaving = true
    }
}

private final class TinyCausalLM: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "proj") var proj: Linear

    override init() {
        self._embed.wrappedValue = Embedding(embeddingCount: 8, dimensions: 6)
        self._proj.wrappedValue = Linear(6, 8, bias: false)
        super.init()
    }

    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        proj(embed(inputIds))
    }
}
