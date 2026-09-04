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

    func testNativeTrainerResumesLegacyCheckpointAtGlobalStep() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let example = TextSFTTokenizedExample(
            inputTokenIds: [0, 1, 2],
            labelTokenIds: [1, 2, 3],
            lossMask: [0, 1, 1]
        )
        let metadata = [
            "base_model": "tiny-causal-lm",
            "dataset_fingerprint": "tiny-dataset",
            "format": "mererun.text-lora.test",
            "max_sequence_length": "128",
        ]

        MLXRandom.seed(73)
        let continuousModel = TinyCausalLM()
        let continuousLayers = try Gemma4TextLoRAInjector.inject(
            into: continuousModel,
            rank: 2,
            targetSuffixes: ["proj"]
        )
        let continuousReport = try TextLoRATrainer.train(
            model: continuousModel,
            loraLayers: continuousLayers,
            examples: [example],
            config: TextLoRATrainingConfig(
                trainingSteps: 4,
                batchSize: 1,
                learningRate: 0.02,
                seed: 19
            ),
            metadata: metadata
        ) { model, inputIds in
            model(inputIds)
        }

        MLXRandom.seed(73)
        let interruptedModel = TinyCausalLM()
        let interruptedLayers = try Gemma4TextLoRAInjector.inject(
            into: interruptedModel,
            rank: 2,
            targetSuffixes: ["proj"]
        )
        _ = try TextLoRATrainer.train(
            model: interruptedModel,
            loraLayers: interruptedLayers,
            examples: [example],
            config: TextLoRATrainingConfig(
                trainingSteps: 2,
                batchSize: 1,
                learningRate: 0.02,
                seed: 19
            ),
            metadata: metadata
        ) { model, inputIds in
            model(inputIds)
        }
        let legacyCheckpointURL = directory.appendingPathComponent("legacy.partial.safetensors")
        try LoRASafetensorsWriter.save(
            loraLayers: interruptedLayers,
            to: legacyCheckpointURL,
            includeOptimizerState: true,
            metadata: metadata
        )
        let (legacyArrays, _) = try MLX.loadArraysAndMetadata(url: legacyCheckpointURL)
        XCTAssertEqual(legacyArrays["proj.lora_down.weight"]?.dtype, .float16)
        XCTAssertEqual(legacyArrays["proj.lora_down.m"]?.dtype, .float32)
        XCTAssertEqual(legacyArrays["proj.lora_down.v"]?.dtype, .float32)

        MLXRandom.seed(73)
        let resumedModel = TinyCausalLM()
        let resumedLayers = try Gemma4TextLoRAInjector.inject(
            into: resumedModel,
            rank: 2,
            targetSuffixes: ["proj"]
        )
        let recorder = TextTrainingProgressRecorder()
        let resumedOutputURL = directory.appendingPathComponent("resumed.safetensors")
        let report = try TextLoRATrainer.train(
            model: resumedModel,
            loraLayers: resumedLayers,
            examples: [example],
            config: TextLoRATrainingConfig(
                trainingSteps: 4,
                batchSize: 1,
                learningRate: 0.02,
                seed: 19,
                resumeFrom: legacyCheckpointURL,
                resumeStep: 2
            ),
            outputURL: resumedOutputURL,
            metadata: metadata,
            progressHandler: { progress in
                if case .training(let step, _, _) = progress.stage {
                    recorder.record(step: step)
                }
            }
        ) { model, inputIds in
            model(inputIds)
        }

        XCTAssertEqual(report.steps, 4)
        XCTAssertEqual(recorder.steps, [4])
        let continuousLayer = try XCTUnwrap(continuousLayers["proj"])
        let resumedLayer = try XCTUnwrap(resumedLayers["proj"])
        XCTAssertLessThan(
            MLX.abs(continuousLayer.loraDown - resumedLayer.loraDown).max().item(Float.self),
            0.02
        )
        XCTAssertLessThan(
            MLX.abs(continuousLayer.loraUp - resumedLayer.loraUp).max().item(Float.self),
            0.02
        )
        XCTAssertEqual(
            try XCTUnwrap(report.finalLoss),
            try XCTUnwrap(continuousReport.finalLoss),
            accuracy: 0.003
        )

        let (_, savedMetadata) = try MLX.loadArraysAndMetadata(url: resumedOutputURL)
        XCTAssertEqual(savedMetadata["checkpoint_schema"], TextLoRACheckpointLoader.checkpointSchema)
        XCTAssertEqual(savedMetadata["completed_steps"], "4")
        let sidecar = try XCTUnwrap(LoRATrainingCheckpointState.load(nextTo: resumedOutputURL))
        XCTAssertEqual(sidecar.step, 4)
        XCTAssertEqual(sidecar.totalSteps, 4)
    }

    func testResumeCheckpointRequiresCompleteOptimizerState() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        MLXRandom.seed(79)
        let model = TinyCausalLM()
        let layers = try Gemma4TextLoRAInjector.inject(
            into: model,
            rank: 2,
            targetSuffixes: ["proj"]
        )
        let checkpointURL = directory.appendingPathComponent("weights-only.safetensors")
        let metadata = [
            "base_model": "tiny-causal-lm",
            "dataset_fingerprint": "tiny-dataset",
            "format": "mererun.text-lora.test",
            "max_sequence_length": "128",
        ]
        try LoRASafetensorsWriter.save(
            loraLayers: layers,
            to: checkpointURL,
            metadata: metadata
        )

        XCTAssertThrowsError(
            try TextLoRACheckpointLoader.load(
                from: checkpointURL,
                into: layers,
                config: TextLoRATrainingConfig(
                    trainingSteps: 4,
                    batchSize: 1,
                    learningRate: 0.02,
                    resumeFrom: checkpointURL,
                    resumeStep: 2
                ),
                expectedMetadata: metadata
            )
        ) { error in
            guard case TextLoRACheckpointError.optimizerStateRequired = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testResumeCheckpointRejectsLossyOptimizerState() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        MLXRandom.seed(81)
        let model = TinyCausalLM()
        let layers = try Gemma4TextLoRAInjector.inject(
            into: model,
            rank: 2,
            targetSuffixes: ["proj"]
        )
        TextLoRATrainer.initializeAdamStateIfNeeded(for: layers)
        let checkpointURL = directory.appendingPathComponent("lossy.safetensors")
        let metadata = [
            "base_model": "tiny-causal-lm",
            "dataset_fingerprint": "tiny-dataset",
            "format": "mererun.text-lora.test",
            "has_optimizer_state": "true",
            "lora_alpha": String(try XCTUnwrap(layers["proj"]).loraAlpha),
            "lora_rank": "2",
            "max_sequence_length": "128",
        ]
        var arrays: [String: MLXArray] = [:]
        for (path, layer) in layers {
            arrays["\(path).lora_down.weight"] = layer.loraDown.asType(.float16)
            arrays["\(path).lora_up.weight"] = layer.loraUp.asType(.float16)
            arrays["\(path).lora_down.m"] = try XCTUnwrap(layer.loraDownM).asType(.float16)
            arrays["\(path).lora_down.v"] = try XCTUnwrap(layer.loraDownV).asType(.float16)
            arrays["\(path).lora_up.m"] = try XCTUnwrap(layer.loraUpM).asType(.float16)
            arrays["\(path).lora_up.v"] = try XCTUnwrap(layer.loraUpV).asType(.float16)
        }
        try MLX.save(arrays: arrays, metadata: metadata, url: checkpointURL)

        XCTAssertThrowsError(
            try TextLoRACheckpointLoader.load(
                from: checkpointURL,
                into: layers,
                config: TextLoRATrainingConfig(
                    trainingSteps: 4,
                    batchSize: 1,
                    learningRate: 0.02,
                    seed: 23,
                    resumeFrom: checkpointURL,
                    resumeStep: 2
                ),
                expectedMetadata: metadata
            )
        ) { error in
            guard case TextLoRACheckpointError.optimizerStateDTypeMismatch = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testSelfDescribingResumeCheckpointLoadsWithoutStepOverride() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        MLXRandom.seed(83)
        let sourceModel = TinyCausalLM()
        let sourceLayers = try Gemma4TextLoRAInjector.inject(
            into: sourceModel,
            rank: 2,
            targetSuffixes: ["proj"]
        )
        TextLoRATrainer.initializeAdamStateIfNeeded(for: sourceLayers)
        let checkpointURL = directory.appendingPathComponent("step-2.safetensors")
        let metadata = [
            "base_model": "tiny-causal-lm",
            "dataset_fingerprint": "tiny-dataset",
            "format": "mererun.text-lora.test",
            "max_sequence_length": "128",
        ]
        let config = TextLoRATrainingConfig(
            trainingSteps: 4,
            batchSize: 1,
            learningRate: 0.02,
            seed: 23,
            resumeFrom: checkpointURL
        )
        try LoRASafetensorsWriter.save(
            loraLayers: sourceLayers,
            to: checkpointURL,
            includeOptimizerState: true,
            metadata: TextLoRACheckpointLoader.checkpointMetadata(
                base: metadata,
                config: config,
                loraLayers: sourceLayers,
                completedSteps: 2
            )
        )
        try LoRATrainingCheckpointState(
            format: "mererun.text-lora.test",
            baseModel: "tiny-causal-lm",
            checkpointFile: checkpointURL.lastPathComponent,
            step: 2,
            totalSteps: 4,
            seed: 23,
            rngState: nil,
            datasetFingerprint: "tiny-dataset",
            configFingerprint: TextLoRACheckpointLoader.configFingerprint(
                config: config,
                loraLayers: sourceLayers,
                metadata: metadata
            )
        ).write(nextTo: checkpointURL)

        MLXRandom.seed(89)
        let destinationModel = TinyCausalLM()
        let destinationLayers = try Gemma4TextLoRAInjector.inject(
            into: destinationModel,
            rank: 2,
            targetSuffixes: ["proj"]
        )
        let report = try TextLoRACheckpointLoader.load(
            from: checkpointURL,
            into: destinationLayers,
            config: config,
            expectedMetadata: metadata
        )

        XCTAssertEqual(report.completedSteps, 2)
        XCTAssertEqual(report.layerCount, 1)
        XCTAssertFalse(report.usedLegacyStepOverride)
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

    func testMultimodalTrainerUpdatesTinyGemma4UnifiedLoRA() throws {
        MLXRandom.seed(37)
        let model = try Gemma4UnifiedCausalLM(config: tinyGemma4UnifiedConfig())
        let layers = try Gemma4TextLoRAInjector.inject(
            into: model,
            rank: 2,
            targetSuffixes: ["q_proj"]
        )
        let example = TextSFTTokenizedExample(
            inputTokenIds: [6, 5, 5, 7, 8],
            labelTokenIds: [5, 5, 7, 8, 9],
            lossMask: [0, 0, 0, 0, 1],
            multimodalInputs: TextSFTMultimodalInputs(
                imageReferences: ["synthetic"],
                imageSHA256: ["synthetic"],
                softTokenCounts: [2],
                mmTokenTypeIds: [0, 1, 1, 0, 0],
                mmTokenTypeShape: [1, 5]
            )
        )

        let report = try TextLoRATrainer.train(
            model: model,
            loraLayers: layers,
            examples: [example],
            evaluationExamples: [example],
            config: TextLoRATrainingConfig(
                trainingSteps: 2,
                batchSize: 1,
                learningRate: 0.01
            ),
            multimodalBatchBuilder: { examples in
                let gathered = try TextSFTTrainingBatchBuilder.makeGatheredBatch(examples)
                return TextLoRAMultimodalTrainingBatch(
                    modelInputs: [
                        gathered.inputIds,
                        MLXArray(Array(repeating: Float(0.25), count: 24), [1, 2, 12]),
                        MLXArray([Int32(0), 0, 1, 0], [1, 2, 2]),
                        MLXArray([Int32(2)]),
                        MLXArray([Int32(0), 1, 1, 0, 0], [1, 5]),
                        gathered.targetPositions,
                    ],
                    targetLabels: gathered.targetLabels
                )
            },
            multimodalGatheredForward: { model, inputs in
                model.trainingLogits(
                    inputIds: inputs[0],
                    pixelValues: inputs[1],
                    imagePositionIds: inputs[2],
                    softTokenCounts: inputs[3],
                    mmTokenTypeIds: inputs[4],
                    flatTargetPositions: inputs[5]
                )
            }
        ) { model, inputIds in
            model.forward(inputIds: inputIds)
        }

        XCTAssertEqual(report.steps, 2)
        XCTAssertGreaterThan(report.layerCount, 0)
        XCTAssertNotNil(report.initialEvaluationLoss)
        XCTAssertNotNil(report.finalEvaluationLoss)
        let upNorm = layers.values.reduce(Float(0)) { partial, layer in
            partial + MLX.sqrt((layer.loraUp * layer.loraUp).sum()).item(Float.self)
        }
        XCTAssertGreaterThan(upNorm, 0)
    }

    private func tinyGemma4TextConfig() throws -> Gemma4TextConfig {
        let data = try JSONSerialization.data(
            withJSONObject: tinyGemma4TextConfigObject(),
            options: []
        )
        return try JSONDecoder().decode(Gemma4TextConfig.self, from: data)
    }

    private func tinyGemma4UnifiedConfig() throws -> Gemma4Config {
        let object: [String: Any] = [
            "model_type": "gemma4_unified",
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "tie_word_embeddings": true,
            "eos_token_id": [1, 2],
            "image_token_id": 5,
            "boi_token_id": 6,
            "eoi_token_id": 7,
            "text_config": tinyGemma4TextConfigObject(),
            "vision_config": [
                "model_type": "gemma4_unified_vision",
                "patch_size": 1,
                "pooling_kernel_size": 2,
                "model_patch_size": 2,
                "mm_embed_dim": 8,
                "mm_posemb_size": 4,
                "num_soft_tokens": 2,
                "rms_norm_eps": 0.000001,
                "output_proj_dims": 8,
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        return try JSONDecoder().decode(Gemma4Config.self, from: data)
    }

    private func tinyGemma4TextConfigObject() -> [String: Any] {
        [
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
