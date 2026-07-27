import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

final class ACEStepAdapterTests: MereRunCoreTestCase {
    func testLoKrFactorizationMatchesLyCORISRules() {
        XCTAssertEqual(
            ACEStepAdapterTrainer.factorization(128, factor: -1).0,
            8
        )
        XCTAssertEqual(
            ACEStepAdapterTrainer.factorization(128, factor: -1).1,
            16
        )
        XCTAssertEqual(
            ACEStepAdapterTrainer.factorization(360, factor: 8).0,
            8
        )
        XCTAssertEqual(
            ACEStepAdapterTrainer.factorization(360, factor: 8).1,
            45
        )
    }

    func testTrainingFlowMatchingObjectiveMatchesUpstream() {
        let clean = MLXArray([Float(2), Float(4)], [1, 1, 2])
        let noise = MLXArray([Float(10), Float(20)], [1, 1, 2])
        let sample = ACEStepAdapterTrainer.flowMatchingSample(
            clean: clean,
            noise: noise,
            timestep: MLXArray([Float(0.25)])
        )

        MLX.eval(sample.noisy, sample.target)
        XCTAssertEqual(sample.noisy.asArray(Float.self), [4, 8])
        XCTAssertEqual(sample.target.asArray(Float.self), [8, 16])
    }

    func testPEFTLoRAChangesOnlyMatchedProjection() throws {
        let directory = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: directory) }
        MLXRandom.seed(12)
        let model = ACEStepDiT(config: tinyConfig)
        let input = MLXRandom.normal([1, 3, 32]).asType(.float32)
        let original = model.layers[0].selfAttn.qProj(input)
        let down = MLXArray.ones([2, 32], dtype: .float32) * 0.02
        let up = MLXArray.ones([32, 2], dtype: .float32) * 0.03
        let adapterURL = directory
            .appendingPathComponent("ace-lora.safetensors")
        try MLX.save(
            arrays: [
                "base_model.model.decoder.layers.0.self_attn.q_proj.lora_A.default.weight": down,
                "base_model.model.decoder.layers.0.self_attn.q_proj.lora_B.default.weight": up,
            ],
            metadata: ["lora_alpha": "4"],
            url: adapterURL
        )

        let report = try ACEStepAdapterLoader.load(
            from: adapterURL,
            kind: .auto,
            scale: 0.5,
            into: model
        )
        let actual = model.layers[0].selfAttn.qProj(input)
        let expected = original
            + MLX.matmul(MLX.matmul(input, down.T), up.T)

        MLX.eval(actual, expected)
        XCTAssertEqual(report.kind, ACEStepAdapterKind.lora)
        XCTAssertEqual(report.matchedLayers, ["layers.0.self_attn.q_proj"])
        XCTAssertLessThan(
            MLX.max(MLX.abs(actual - expected)).item(Float.self),
            1e-6
        )
    }

    func testLyCORISLoKRDecomposedFactorsAndAlpha() throws {
        let directory = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: directory) }
        MLXRandom.seed(14)
        let model = ACEStepDiT(config: tinyConfig)
        let input = MLXRandom.normal([1, 2, 32]).asType(.float32)
        let original = model.layers[0].crossAttn.vProj(input)
        let w1 = MLXArray.ones([4, 4], dtype: .float32) * 0.01
        let w2A = MLXArray.ones([8, 2], dtype: .float32) * 0.02
        let w2B = MLXArray.ones([2, 8], dtype: .float32) * 0.03
        let delta = MLX.kron(w1, MLX.matmul(w2A, w2B))
        let adapterURL = directory
            .appendingPathComponent("ace-lokr.safetensors")
        let key = "lycoris_layers_0_cross_attn_v_proj"
        try MLX.save(
            arrays: [
                "\(key).lokr_w1": w1,
                "\(key).lokr_w2_a": w2A,
                "\(key).lokr_w2_b": w2B,
                "\(key).alpha": MLXArray(Float(4)),
            ],
            metadata: ["algo": "lokr", "format": "lycoris"],
            url: adapterURL
        )

        let report = try ACEStepAdapterLoader.load(
            from: adapterURL,
            kind: .auto,
            scale: 0.25,
            into: model
        )
        let actual = model.layers[0].crossAttn.vProj(input)
        let expected = original
            + MLX.matmul(input, delta.T) * MLXArray(Float(0.5))

        MLX.eval(actual, expected)
        XCTAssertEqual(report.kind, ACEStepAdapterKind.lokr)
        XCTAssertEqual(report.matchedLayers, ["layers.0.cross_attn.v_proj"])
        XCTAssertLessThan(
            MLX.max(MLX.abs(actual - expected)).item(Float.self),
            1e-6
        )
    }

    func testAdaptersStackOnOneProjection() throws {
        let directory = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: directory) }
        MLXRandom.seed(16)
        let model = ACEStepDiT(config: tinyConfig)
        let input = MLXRandom.normal([1, 2, 32]).asType(.float32)
        let original = model.layers[0].selfAttn.oProj(input)
        let down = MLXArray.ones([1, 32], dtype: .float32) * 0.01
        let up = MLXArray.ones([32, 1], dtype: .float32) * 0.02
        let adapterURL = directory
            .appendingPathComponent("stacked-lora.safetensors")
        try MLX.save(
            arrays: [
                "layers.0.self_attn.o_proj.lora_down.weight": down,
                "layers.0.self_attn.o_proj.lora_up.weight": up,
            ],
            url: adapterURL
        )

        _ = try ACEStepAdapterLoader.load(
            from: adapterURL,
            kind: .lora,
            scale: 1,
            into: model
        )
        _ = try ACEStepAdapterLoader.load(
            from: adapterURL,
            kind: .lora,
            scale: 0.5,
            into: model
        )
        let actual = model.layers[0].selfAttn.oProj(input)
        let contribution = MLX.matmul(
            MLX.matmul(input, down.T),
            up.T
        )
        let expected = original + contribution * MLXArray(Float(1.5))

        MLX.eval(actual, expected)
        XCTAssertLessThan(
            MLX.max(MLX.abs(actual - expected)).item(Float.self),
            1e-6
        )
    }

    private var tinyConfig: ACEStepConfig {
        ACEStepConfig(
            hiddenSize: 32,
            intermediateSize: 64,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            numKeyValueHeads: 4,
            encoderHiddenSize: 32,
            encoderIntermediateSize: 64,
            encoderNumAttentionHeads: 4,
            encoderNumKeyValueHeads: 4,
            headDim: 8,
            audioAcousticHiddenDim: 4,
            inChannels: 12
        )
    }
}
