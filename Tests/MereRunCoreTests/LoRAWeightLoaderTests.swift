import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class LoRAWeightLoaderTests: MereRunCoreTestCase {
    func testFlux2LoRAStackerCombinesOrderedScaledUpdates() throws {
        let first = LoRAWeights(
            weights: [
                "transformer_blocks.0.attn.to_q": (
                    down: MLXArray([1, 2, 3, 4] as [Float]).reshaped([2, 2]),
                    up: MLXArray([1, 2, 3, 4, 5, 6] as [Float]).reshaped([3, 2])
                ),
            ],
            rank: 2,
            alpha: 4
        )
        let second = LoRAWeights(
            weights: [
                "transformer_blocks.0.attn.to_q": (
                    down: MLXArray([8, 12] as [Float]).reshaped([1, 2]),
                    up: MLXArray([10, 20, 30] as [Float]).reshaped([3, 1])
                ),
            ],
            rank: 1,
            alpha: 1
        )

        let stacked = try Flux2LoRAStacker.stack([
            Flux2LoRAStackInput(label: "turbo", scale: 0.5, weights: first),
            Flux2LoRAStackInput(label: "style", scale: 0.25, weights: second),
        ])
        let pair = try XCTUnwrap(stacked.weights["transformer_blocks.0.attn.to_q"])

        XCTAssertEqual(pair.down.shape, [3, 2])
        XCTAssertEqual(pair.up.shape, [3, 3])
        XCTAssertEqual(pair.down.asArray(Float.self), [1, 2, 3, 4, 2, 3])
        XCTAssertEqual(pair.up.asArray(Float.self), [1, 2, 10, 3, 4, 20, 5, 6, 30])
        XCTAssertEqual(stacked.targetRanks["transformer_blocks.0.attn.to_q"], 3)
        XCTAssertEqual(stacked.alpha, 3)
    }

    func testFlux2LoRAStackerRejectsIncompatibleLayerDimensions() {
        let first = LoRAWeights(
            weights: [
                "transformer_blocks.0.attn.to_q": (
                    down: MLXArray.zeros([2, 4], dtype: .float32),
                    up: MLXArray.zeros([6, 2], dtype: .float32)
                ),
            ],
            rank: 2
        )
        let second = LoRAWeights(
            weights: [
                "transformer_blocks.0.attn.to_q": (
                    down: MLXArray.zeros([1, 5], dtype: .float32),
                    up: MLXArray.zeros([6, 1], dtype: .float32)
                ),
            ],
            rank: 1
        )

        XCTAssertThrowsError(
            try Flux2LoRAStacker.stack([
                Flux2LoRAStackInput(label: "first", scale: 1, weights: first),
                Flux2LoRAStackInput(label: "second", scale: 1, weights: second),
            ])
        ) { error in
            XCTAssertEqual(
                error as? Flux2LoRAStacker.StackError,
                .incompatiblePair(
                    label: "second",
                    path: "transformer_blocks.0.attn.to_q",
                    expectedInput: 4,
                    actualInput: 5,
                    expectedOutput: 6,
                    actualOutput: 6
                )
            )
        }
    }

    func testFlux2DevLoRAValidationRejectsFlux1HiddenWidth() {
        XCTAssertThrowsError(
            try Flux2KleinGenerator.validateLoRAWeightShapes(
                path: "single_transformer_blocks.0.attn.to_q",
                down: MLXArray.zeros([16, 3_072], dtype: .float32),
                up: MLXArray.zeros([3_072, 16], dtype: .float32),
                expectedDown: [16, 6_144],
                expectedUp: [6_144, 16]
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("not FLUX.1"))
            XCTAssertTrue(error.localizedDescription.contains("6144"))
            XCTAssertTrue(error.localizedDescription.contains("3072"))
        }
    }

    func testExternalFlux2DevLoRAWhenConfigured() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_TEST_FLUX2_DEV_LORA"] else {
            throw XCTSkip("Set MERERUN_TEST_FLUX2_DEV_LORA to inspect a real FLUX.2-dev adapter.")
        }

        let weights = try LoRAWeightLoader.load(from: URL(fileURLWithPath: path))
        let pair = try XCTUnwrap(weights.weights["transformer_blocks.0.attn.to_q"])
        XCTAssertEqual(pair.down.shape[1], 6_144)
        XCTAssertEqual(pair.up.shape[0], 6_144)
    }

    func testExternalFlux1LoRAIsRejectedWhenConfigured() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_TEST_FLUX1_LORA"] else {
            throw XCTSkip("Set MERERUN_TEST_FLUX1_LORA to inspect a FLUX.1 adapter.")
        }

        let weights = try LoRAWeightLoader.load(from: URL(fileURLWithPath: path))
        let pair = try XCTUnwrap(weights.weights["single_transformer_blocks.0.attn.to_q"])
        XCTAssertThrowsError(
            try Flux2KleinGenerator.validateLoRAWeightShapes(
                path: "single_transformer_blocks.0.attn.to_q",
                down: pair.down,
                up: pair.up,
                expectedDown: [pair.down.shape[0], 6_144],
                expectedUp: [6_144, pair.up.shape[1]]
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("not FLUX.1"))
        }
    }

    func testFlux1ArchitecturePreservesDiffusersSingleBlockOutputTarget() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let adapterURL = temp.appendingPathComponent("flux1.safetensors")
        try MLX.save(
            arrays: [
                "transformer.single_transformer_blocks.0.proj_out.lora_A.weight":
                    MLXArray.zeros([16, 15_360], dtype: .float32),
                "transformer.single_transformer_blocks.0.proj_out.lora_B.weight":
                    MLXArray.zeros([3_072, 16], dtype: .float32),
            ],
            url: adapterURL
        )

        let weights = try LoRAWeightLoader.load(from: adapterURL, architecture: .flux1)
        let pair = try XCTUnwrap(weights.weights["single_transformer_blocks.0.proj_out"])
        XCTAssertEqual(pair.down.shape, [16, 15_360])
        XCTAssertEqual(pair.up.shape, [3_072, 16])
        XCTAssertNil(weights.weights["single_transformer_blocks.0.attn.to_out"])
    }

    func testExternalFlux1LoRALoadsWithNativeTargetsWhenConfigured() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_TEST_FLUX1_LORA"] else {
            throw XCTSkip("Set MERERUN_TEST_FLUX1_LORA to inspect a real FLUX.1 adapter.")
        }

        let weights = try LoRAWeightLoader.load(
            from: URL(fileURLWithPath: path),
            architecture: .flux1
        )
        let query = try XCTUnwrap(weights.weights["single_transformer_blocks.0.attn.to_q"])
        let projection = try XCTUnwrap(weights.weights["single_transformer_blocks.0.proj_out"])
        XCTAssertEqual(query.down.shape, [16, 3_072])
        XCTAssertEqual(query.up.shape, [3_072, 16])
        XCTAssertEqual(projection.down.shape, [16, 15_360])
        XCTAssertEqual(projection.up.shape, [3_072, 16])
    }

    func testLoadInfersPerLayerRanks() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let adapterURL = temp.appendingPathComponent("mixed-rank.safetensors")
        try MLX.save(
            arrays: [
                "transformer_blocks.0.attn.to_q.lora_down.weight": MLXArray.zeros([8, 4], dtype: .float32),
                "transformer_blocks.0.attn.to_q.lora_up.weight": MLXArray.zeros([4, 8], dtype: .float32),
                "transformer_blocks.0.ff.linear_in.lora_down.weight": MLXArray.zeros([16, 4], dtype: .float32),
                "transformer_blocks.0.ff.linear_in.lora_up.weight": MLXArray.zeros([4, 16], dtype: .float32),
            ],
            metadata: ["lora_alpha": "4"],
            url: adapterURL
        )

        let weights = try LoRAWeightLoader.load(from: adapterURL)

        XCTAssertEqual(weights.targetRanks["transformer_blocks.0.attn.to_q"], 8)
        XCTAssertEqual(weights.targetRanks["transformer_blocks.0.ff.linear_in"], 16)
        XCTAssertEqual(weights.alpha, 4)
    }

    func testLoadPreservesKrea2NativeGlobalKeys() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let adapterURL = temp.appendingPathComponent("krea-official-style.safetensors")
        try MLX.save(
            arrays: [
                "transformer.img_in.lora_A.weight": MLXArray.zeros([4, 64], dtype: .float32),
                "transformer.img_in.lora_B.weight": MLXArray.zeros([6144, 4], dtype: .float32),
                "transformer.final_layer.linear.lora_A.weight": MLXArray.zeros([4, 6144], dtype: .float32),
                "transformer.final_layer.linear.lora_B.weight": MLXArray.zeros([64, 4], dtype: .float32),
            ],
            url: adapterURL
        )

        let weights = try LoRAWeightLoader.load(from: adapterURL)

        XCTAssertNotNil(weights.weights["img_in"])
        XCTAssertNotNil(weights.weights["final_layer.linear"])
        XCTAssertNil(weights.weights["x_embedder"])
        XCTAssertNil(weights.weights["proj_out"])
    }

    func testLoadPreservesRepresentativeKrea2PublishedAdapterSurface() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let paths = [
            "img_in",
            "txt_in.linear_1",
            "txt_in.linear_2",
            "text_fusion.projector",
            "text_fusion.layerwise_blocks.0.attn.to_q",
            "text_fusion.layerwise_blocks.0.attn.to_k",
            "text_fusion.layerwise_blocks.0.attn.to_v",
            "text_fusion.layerwise_blocks.0.attn.to_gate",
            "text_fusion.layerwise_blocks.0.attn.to_out.0",
            "text_fusion.layerwise_blocks.0.ff.gate",
            "text_fusion.layerwise_blocks.0.ff.up",
            "text_fusion.layerwise_blocks.0.ff.down",
            "text_fusion.refiner_blocks.0.attn.to_q",
            "text_fusion.refiner_blocks.0.attn.to_k",
            "text_fusion.refiner_blocks.0.attn.to_v",
            "text_fusion.refiner_blocks.0.attn.to_gate",
            "text_fusion.refiner_blocks.0.attn.to_out.0",
            "text_fusion.refiner_blocks.0.ff.gate",
            "text_fusion.refiner_blocks.0.ff.up",
            "text_fusion.refiner_blocks.0.ff.down",
            "time_embed.linear_1",
            "time_embed.linear_2",
            "time_mod_proj",
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.attn.to_k",
            "transformer_blocks.0.attn.to_v",
            "transformer_blocks.0.attn.to_gate",
            "transformer_blocks.0.attn.to_out.0",
            "transformer_blocks.0.ff.gate",
            "transformer_blocks.0.ff.up",
            "transformer_blocks.0.ff.down",
            "final_layer.linear",
        ]
        let adapterURL = temp.appendingPathComponent("krea-published-surface.safetensors")
        var arrays: [String: MLXArray] = [:]
        for path in paths {
            arrays["transformer.\(path).lora_A.weight"] = MLXArray.zeros([4, 8], dtype: .float32)
            arrays["transformer.\(path).lora_B.weight"] = MLXArray.zeros([8, 4], dtype: .float32)
        }
        try MLX.save(arrays: arrays, url: adapterURL)

        let weights = try LoRAWeightLoader.load(from: adapterURL)

        XCTAssertEqual(Set(weights.weights.keys), Set(paths))
        XCTAssertNil(weights.weights["x_embedder"])
        XCTAssertNil(weights.weights["proj_out"])
        XCTAssertEqual(weights.targetRanks["transformer_blocks.0.attn.to_q"], 4)
        XCTAssertEqual(weights.targetRanks["text_fusion.layerwise_blocks.0.ff.down"], 4)
    }

    func testLoadMapsFALKrea2AdapterKeys() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let adapterURL = temp.appendingPathComponent("fal-krea2-lora.safetensors")
        let sourceToExpected = [
            "base_model.model.first": "img_in",
            "base_model.model.txtmlp.1": "txt_in.linear_1",
            "base_model.model.txtmlp.3": "txt_in.linear_2",
            "base_model.model.txtfusion.projector": "text_fusion.projector",
            "base_model.model.tmlp.0": "time_embed.linear_1",
            "base_model.model.tmlp.2": "time_embed.linear_2",
            "base_model.model.tproj.1": "time_mod_proj",
            "base_model.model.last.linear": "final_layer.linear",
            "base_model.model.blocks.0.attn.wq": "transformer_blocks.0.attn.to_q",
            "base_model.model.blocks.0.attn.wk": "transformer_blocks.0.attn.to_k",
            "base_model.model.blocks.0.attn.wv": "transformer_blocks.0.attn.to_v",
            "base_model.model.blocks.0.attn.gate": "transformer_blocks.0.attn.to_gate",
            "base_model.model.blocks.0.attn.wo": "transformer_blocks.0.attn.to_out.0",
            "base_model.model.blocks.0.mlp.gate": "transformer_blocks.0.ff.gate",
            "base_model.model.blocks.0.mlp.up": "transformer_blocks.0.ff.up",
            "base_model.model.blocks.0.mlp.down": "transformer_blocks.0.ff.down",
            "base_model.model.txtfusion.layerwise_blocks.0.attn.wq": "text_fusion.layerwise_blocks.0.attn.to_q",
            "base_model.model.txtfusion.refiner_blocks.1.mlp.down": "text_fusion.refiner_blocks.1.ff.down",
        ]
        var arrays: [String: MLXArray] = [:]
        for source in sourceToExpected.keys {
            arrays["\(source).lora_A.weight"] = MLXArray.zeros([4, 8], dtype: .float32)
            arrays["\(source).lora_B.weight"] = MLXArray.zeros([8, 4], dtype: .float32)
        }
        try MLX.save(arrays: arrays, metadata: ["lora_alpha": "32"], url: adapterURL)

        let weights = try LoRAWeightLoader.load(from: adapterURL)

        XCTAssertEqual(Set(weights.weights.keys), Set(sourceToExpected.values))
        XCTAssertNil(weights.weights["blocks.0.attn.wq"])
        XCTAssertNil(weights.weights["txtfusion.layerwise_blocks.0.attn.wq"])
        XCTAssertEqual(weights.alpha, 32)
        XCTAssertEqual(weights.targetRanks["transformer_blocks.0.attn.to_q"], 4)
        XCTAssertEqual(weights.targetRanks["text_fusion.refiner_blocks.1.ff.down"], 4)
    }

    func testLoadMapsDottedBFLKleinAdapterKeys() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let adapterURL = temp.appendingPathComponent("fal-klein-lora.safetensors")
        try MLX.save(
            arrays: [
                "base_model.model.double_blocks.0.img_attn.qkv.lora_A.weight": MLXArray.zeros([4, 8], dtype: .float32),
                "base_model.model.double_blocks.0.img_attn.qkv.lora_B.weight": MLXArray.zeros([24, 4], dtype: .float32),
                "base_model.model.double_blocks.1.txt_attn.proj.lora_A.weight": MLXArray.zeros([4, 8], dtype: .float32),
                "base_model.model.double_blocks.1.txt_attn.proj.lora_B.weight": MLXArray.zeros([8, 4], dtype: .float32),
                "base_model.model.single_blocks.2.linear1.lora_A.weight": MLXArray.zeros([4, 8], dtype: .float32),
                "base_model.model.single_blocks.2.linear1.lora_B.weight": MLXArray.zeros([32, 4], dtype: .float32),
                "base_model.model.single_blocks.3.linear2.lora_A.weight": MLXArray.zeros([4, 8], dtype: .float32),
                "base_model.model.single_blocks.3.linear2.lora_B.weight": MLXArray.zeros([8, 4], dtype: .float32),
            ],
            url: adapterURL
        )

        let weights = try LoRAWeightLoader.load(from: adapterURL)

        XCTAssertNotNil(weights.weights["transformer_blocks.0.attn.to_q"])
        XCTAssertNotNil(weights.weights["transformer_blocks.0.attn.to_k"])
        XCTAssertNotNil(weights.weights["transformer_blocks.0.attn.to_v"])
        XCTAssertNotNil(weights.weights["transformer_blocks.1.attn.to_add_out"])
        XCTAssertNotNil(weights.weights["single_transformer_blocks.2.attn.to_qkv_mlp_proj"])
        XCTAssertNotNil(weights.weights["single_transformer_blocks.3.attn.to_out"])
        XCTAssertNil(weights.weights["double_blocks.0.img_attn.qkv"])
        XCTAssertEqual(weights.targetRanks["transformer_blocks.0.attn.to_q"], 4)
        XCTAssertEqual(weights.targetRanks["single_transformer_blocks.2.attn.to_qkv_mlp_proj"], 4)
    }
}
