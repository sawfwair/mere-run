import Foundation
import MLX
import MLXNN
import MLXRandom
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

    func testNativeTrainerReportsBeforeAndAfterEvaluationLoss() throws {
        MLXRandom.seed(31)
        let model = TinyCausalLM()
        let layers = try Gemma4TextLoRAInjector.inject(
            into: model,
            rank: 2,
            targetSuffixes: ["proj"]
        )
        let example = TextSFTTokenizedExample(
            inputTokenIds: [0, 1, 2],
            labelTokenIds: [1, 2, 3],
            lossMask: [0, 1, 1]
        )

        let report = try TextLoRATrainer.train(
            model: model,
            loraLayers: layers,
            examples: [example],
            evaluationExamples: [example],
            config: TextLoRATrainingConfig(
                trainingSteps: 20,
                batchSize: 1,
                learningRate: 0.05
            )
        ) { model, inputIds in
            model(inputIds)
        }

        let initial = try XCTUnwrap(report.initialEvaluationLoss)
        let final = try XCTUnwrap(report.finalEvaluationLoss)
        XCTAssertEqual(report.evaluationExampleCount, 1)
        XCTAssertEqual(report.evaluationTargetTokenCount, 2)
        XCTAssertLessThan(final, initial)
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

    func testGatheredBatchMatchesMaskPositions() throws {
        let examples = [
            TextSFTTokenizedExample(
                inputTokenIds: [3, 4, 5, 6],
                labelTokenIds: [4, 5, 6, 7],
                lossMask: [0, 1, 0, 1]
            ),
            TextSFTTokenizedExample(
                inputTokenIds: [1, 2],
                labelTokenIds: [2, 3],
                lossMask: [0, 1]
            ),
        ]
        let batch = try TextSFTTrainingBatchBuilder.makeGatheredBatch(examples)

        XCTAssertEqual(batch.inputIds.shape, [2, 4])
        // Row 0 targets columns 1 and 3; row 1 (padded to length 4) column 1.
        XCTAssertEqual(batch.targetPositions.asArray(Int32.self), [1, 3, 5])
        XCTAssertEqual(batch.targetLabels.asArray(Int32.self), [5, 7, 3])
    }

    func testGatheredLossMatchesMaskedLoss() throws {
        MLXRandom.seed(11)
        let logits = MLXRandom.normal([2, 4, 8])
        let labels = MLXArray([Int32]([1, 2, 3, 4, 5, 6, 7, 0]), [2, 4])
        let mask = MLXArray([Float]([0, 1, 1, 0, 1, 0, 0, 1]), [2, 4])

        let legacy = TextSFTTrainingLoss.maskedNextTokenCrossEntropy(
            logits: logits,
            labels: labels,
            lossMask: mask
        ).item(Float.self)

        let flatLogits = logits.reshaped([-1, 8])
        let positions = MLXArray([Int32]([1, 2, 4, 7]))
        let gathered = TextSFTTrainingLoss.gatheredNextTokenCrossEntropy(
            logits: take(flatLogits, positions, axis: 0),
            labels: MLXArray([Int32]([2, 3, 5, 0]))
        ).item(Float.self)

        XCTAssertEqual(legacy, gathered, accuracy: 1e-5)
    }

    func testTrainingLogitsMatchesFullForwardAtTargets() throws {
        MLXRandom.seed(5)
        let model = Gemma4TextCausalLM(config: try tinyGemma4TextConfig())
        let inputIds = MLXArray([Int32]([3, 4, 5, 6, 1, 2, 0, 0]), [2, 4])
        let positions = MLXArray([Int32]([1, 3, 5]))

        let full = model(inputIds).reshaped([-1, 32])
        let expected = take(full, positions, axis: 0)
        let gathered = model.trainingLogits(inputIds: inputIds, flatTargetPositions: positions)

        XCTAssertEqual(gathered.shape, expected.shape)
        let maxDiff = MLX.abs(gathered - expected).max().item(Float.self)
        XCTAssertLessThan(maxDiff, 1e-5)
    }

    func testGatheredTrainingMatchesLegacyTrajectory() throws {
        let examples = [
            TextSFTTokenizedExample(
                inputTokenIds: [3, 4, 5, 6],
                labelTokenIds: [4, 5, 6, 7],
                lossMask: [0, 1, 1, 1]
            ),
            TextSFTTokenizedExample(
                inputTokenIds: [7, 8, 9],
                labelTokenIds: [8, 9, 10],
                lossMask: [0, 0, 1]
            ),
        ]
        let config = TextLoRATrainingConfig(trainingSteps: 3, batchSize: 1, learningRate: 0.02)

        func trainOnce(gathered: Bool) throws -> Float {
            MLXRandom.seed(21)
            let model = Gemma4TextCausalLM(config: try tinyGemma4TextConfig())
            let layers = try Gemma4TextLoRAInjector.inject(into: model, rank: 2)
            let report = try TextLoRATrainer.train(
                model: model,
                loraLayers: layers,
                examples: examples,
                config: config,
                gatheredForward: gathered
                    ? { model, inputIds, positions in
                        model.trainingLogits(inputIds: inputIds, flatTargetPositions: positions)
                    }
                    : nil
            ) { model, inputIds in
                model(inputIds)
            }
            return try XCTUnwrap(report.finalLoss)
        }

        let legacyLoss = try trainOnce(gathered: false)
        let gatheredLoss = try trainOnce(gathered: true)
        XCTAssertEqual(legacyLoss, gatheredLoss, accuracy: 2e-4)
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
