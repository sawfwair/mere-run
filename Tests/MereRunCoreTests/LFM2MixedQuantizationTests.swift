import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import MereRunCore

final class LFM2MixedQuantizationTests: MereRunCoreTestCase {
    private func config(quantized: Bool, small: Bool = false) throws -> LFM2Config {
        let (gateBits, upBits, downBits, upGroup) = small ? (4, 5, 6, 64) : (6, 6, 8, 32)
        var text = """
            {"model_type":"lfm2_moe","mererun_fp32_activations":true,"vocab_size":32,"hidden_size":64,
             "intermediate_size":64,"moe_intermediate_size":64,"num_hidden_layers":1,
             "num_dense_layers":0,"num_experts":4,"num_experts_per_tok":2,
             "num_attention_heads":4,"num_key_value_heads":2,"max_position_embeddings":128,
             "norm_eps":0.00001,"conv_L_cache":3,"layer_types":["full_attention"]}
            """
        if quantized {
            text = text.replacingOccurrences(of: "\"model_type\"", with: """
                "quantization":{"bits":8,"group_size":64,"mode":"affine"},
                "mererun_quantization":{
                  "model.layers.0.feed_forward.switch_mlp.gate_proj":{"bits":\(gateBits),"group_size":32,"mode":"affine"},
                  "model.layers.0.feed_forward.switch_mlp.up_proj":{"bits":\(upBits),"group_size":\(upGroup),"mode":"affine"},
                  "model.layers.0.feed_forward.switch_mlp.down_proj":{"bits":\(downBits),"group_size":64,"mode":"affine"}},
                "model_type"
                """)
        }
        return try JSONDecoder().decode(LFM2Config.self, from: Data(text.utf8))
    }

    func testMixedProjectionMetadataControlsNativePackingAndKernelEligibility() throws {
        let model = LFM2SwitchGLU(config: try config(quantized: true), layerIndex: 0)
        XCTAssertEqual(model.gateProj.bits, 6)
        XCTAssertEqual(model.gateProj.groupSize, 32)
        XCTAssertEqual(model.upProj.bits, 6)
        XCTAssertEqual(model.downProj.bits, 8)
        XCTAssertEqual(model.downProj.groupSize, 64)
        let fallback = LFM2SwitchGLU(config: try config(quantized: true), layerIndex: 1)
        XCTAssertEqual(fallback.gateProj.bits, 8)
        XCTAssertEqual(fallback.gateProj.groupSize, 64)
        XCTAssertTrue(try config(quantized: false).moduleQuantization.isEmpty)
        XCTAssertTrue(try config(quantized: true).useFP32Activations)
        XCTAssertEqual(try config(quantized: true).prefillChunkSize(promptTokens: 8_192), 128)
        XCTAssertEqual(try config(quantized: true).prefillChunkSize(promptTokens: 131_072), 32)
    }

    func testMixedSixAndEightBitGatherMatchesDequantizedExpertReference() throws {
        try verifyMixedGather(small: false)
    }

    func testMixedFourFiveAndSixBitGatherMatchesDequantizedExpertReference() throws {
        try verifyMixedGather(small: true)
    }

    private func verifyMixedGather(small: Bool) throws {
        MLXRandom.seed(20261013)
        let actual = LFM2SwitchGLU(config: try config(quantized: true, small: small), layerIndex: 0)
        let expected = LFM2SwitchGLU(config: try config(quantized: false), layerIndex: 0)
        var packed: [(String, MLXArray)] = []
        var restored: [(String, MLXArray)] = []
        let precisions = small
            ? [("gate_proj", 4, 32), ("up_proj", 5, 64), ("down_proj", 6, 64)]
            : [("gate_proj", 6, 32), ("up_proj", 6, 32), ("down_proj", 8, 64)]
        for (name, bits, group) in precisions {
            let original = MLXRandom.uniform(low: -0.2, high: 0.2, [4, 64, 64])
            let quantized = MLX.quantized(original, groupSize: group, bits: bits, mode: .affine)
            let biases = try XCTUnwrap(quantized.biases)
            packed += [(name + ".weight", quantized.wq), (name + ".scales", quantized.scales), (name + ".biases", biases)]
            restored.append((name + ".weight", MLX.dequantized(
                quantized.wq, scales: quantized.scales, biases: biases, groupSize: group, bits: bits, mode: .affine
            )))
        }
        try actual.update(parameters: ModuleParameters.unflattened(packed), verify: .noUnusedKeys)
        try expected.update(parameters: ModuleParameters.unflattened(restored), verify: .all)
        let input = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 3, 64])
        let indices = MLXArray([Int32(0), 1, 2, 3, 3, 0], [1, 3, 2])
        let a = actual(input, indices: indices)
        let b = expected(input, indices: indices)
        MLX.eval(a, b)
        XCTAssertLessThan(abs(a - b).max().item(Float.self), 0.0001)
    }
    func testNativeLoaderPreservesBF16TrunkAndHighPrecisionAffineMetadata() throws {
        for metadataDType in [DType.float16, .float32] {
            MLXRandom.seed(20261013)
            let expected = LFM2Model(config: try config(quantized: false))
            let actual = LFM2Model(config: try config(quantized: true))
            var checkpoint: [String: MLXArray] = [:]
            var restored: [(String, MLXArray)] = []
            for (key, value) in expected.parameters().flattened() {
                let original = value.asType(key.hasSuffix("expert_bias") ? .float32 : .bfloat16)
                if key.contains(".switch_mlp."), key.hasSuffix(".weight") {
                    let down = key.contains(".down_proj.")
                    let bits = down ? 8 : 6
                    let group = down ? 64 : 32
                    let packed = MLX.quantized(original.asType(.float32), groupSize: group, bits: bits)
                    let biases = try XCTUnwrap(packed.biases).asType(metadataDType)
                    let scales = packed.scales.asType(metadataDType)
                    let prefix = String(key.dropLast(".weight".count))
                    checkpoint[key] = packed.wq
                    checkpoint[prefix + ".scales"] = scales
                    checkpoint[prefix + ".biases"] = biases
                    restored.append((key, MLX.dequantized(
                        packed.wq, scales: scales.asType(.float32), biases: biases.asType(.float32), groupSize: group, bits: bits
                    )))
                } else {
                    checkpoint[key] = original
                    restored.append((key, original))
                }
            }
            try expected.update(parameters: ModuleParameters.unflattened(restored), verify: .all)
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                checkpoint, to: actual, groupSize: 64, bits: 8, mapper: LFM2Resources.mapWeight(key:value:)
            )
            let parameters = Dictionary(uniqueKeysWithValues: actual.parameters().flattened())
            XCTAssertEqual(parameters["model.embed_tokens.weight"]?.dtype, .bfloat16)
            XCTAssertEqual(parameters["model.layers.0.feed_forward.switch_mlp.gate_proj.scales"]?.dtype, metadataDType)
            let input = MLXArray([Int32(1), 2, 3], [1, 3])
            let a = actual.logits(from: actual.model(input, cache: nil))
            let b = expected.logits(from: expected.model(input, cache: nil))
            MLX.eval(a, b)
            XCTAssertEqual(a.dtype, .float32)
            XCTAssertLessThan(abs(a - b).max().item(Float.self), 0.001)
        }
    }

}
