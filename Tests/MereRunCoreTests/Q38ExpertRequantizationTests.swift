import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import MereRunCore

final class Q38ExpertRequantizationTests: MereRunCoreTestCase {
    func testPackingPreservesLayoutAndDoesNotMutateQ4Source() throws {
        MLXRandom.seed(730)
        let dense = MLXRandom.normal([4, 64, 128]).asType(.bfloat16)
        let packed = MLX.quantized(dense, groupSize: 64, bits: 4)
        let source = Q35SwitchLinear(weight: packed.0, scales: packed.1, biases: packed.2,
                                    bias: nil, groupSize: 64, bits: 4)
        let sourceBiases = try XCTUnwrap(packed.2)
        let original = packed.0.asArray(UInt32.self)
        for recipe in [Q38ExpertRequantization.q3Group64, .q2Group32] {
            let result = try recipe.arrays(from: source)
            XCTAssertEqual(result.0.shape, [4, 64, 128 * recipe.bits / 32])
            XCTAssertEqual(result.1.shape, [4, 64, 128 / recipe.groupSize])
            XCTAssertEqual(result.1.dtype, .bfloat16)
            let reconstructed = MLX.dequantized(result.0, scales: result.1, biases: result.2,
                                               groupSize: recipe.groupSize, bits: recipe.bits)
            XCTAssertEqual(reconstructed.shape, dense.shape)
            XCTAssertTrue(reconstructed.asArray(Float.self).allSatisfy(\.isFinite))
            XCTAssertLessThan(result.0.nbytes + result.1.nbytes + result.2.nbytes,
                              packed.0.nbytes + packed.1.nbytes + sourceBiases.nbytes)
        }
        XCTAssertEqual(source.weight.asArray(UInt32.self), original)
    }

    func testQ3ExpertsPreserveFusedAndSerialArithmetic() throws {
        try qualify(.q3Group64)
    }

    func testSmallGroupQ2ExpertsPreserveFusedAndSerialArithmetic() throws {
        try qualify(.q2Group32)
    }

    private func qualify(_ recipe: Q38ExpertRequantization) throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu,
                          "Routed expert parity requires MERERUN_TEST_MLX_DEVICE=gpu.")
        MLXRandom.seed(731)
        let layer = Q35SwitchGLU(config: try configuration())
        for (path, module) in layer.leafModules().flattened() {
            let expert = try XCTUnwrap(module as? Q35SwitchLinear)
            let q4 = MLX.quantized(expert.weight.asType(.bfloat16), groupSize: 64, bits: 4)
            let source = Q35SwitchLinear(weight: q4.0, scales: q4.1, biases: q4.2,
                                        bias: nil, groupSize: 64, bits: 4)
            let result = try recipe.arrays(from: source)
            // Keep Q4 fallback metadata, as the loaded candidate does. The
            // runtime must infer actual precision from the packed tensor shapes.
            layer.update(modules: ModuleChildren.unflattened([(path, Q35SwitchLinear(
                weight: result.0, scales: result.1, biases: result.2,
                bias: nil, groupSize: 64, bits: 4
            ))]))
        }
        let input = MLXRandom.normal([1, 4, 2_560]).asType(.bfloat16)
        let indices = MLXArray((0..<40).map { Int32($0 % 16) }).reshaped(1, 4, 10)
        let expected = layer.downProj(MLXNN.silu(layer.gateProj(input, indices: indices))
                                     * layer.upProj(input, indices: indices), indices: indices)
        MLX.eval(expected)
        XCTAssertTrue(layer.prepareFusedGateUpAndReleaseSources())
        XCTAssertEqual(layer.sourceGateUpElementCount, 0)
        assertExact(layer(input, indices: indices), expected, "fusion")
        for count in [2, 3, 4] {
            let rows = input[0..., 0..<count, 0...]
            let selected = indices[0..., 0..<count, 0...]
            let serial = (0..<count).map { row in
                let output = layer(rows[0..., row..<(row + 1), 0...],
                                   indices: selected[0..., row..<(row + 1), 0...])
                MLX.eval(output)
                return output
            }
            assertExact(layer(rows, indices: selected), MLX.concatenated(serial, axis: 1),
                        "\(recipe.rawValue) verification width \(count)")
        }
    }

    private func assertExact(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
        XCTAssertEqual((actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self),
                       0, label)
    }

    private func configuration() throws -> Q35Config {
        try JSONDecoder().decode(Q35Config.self, from: Data(#"""
        {"model_type":"qwen4_exp","text_config":{
          "model_type":"qwen4_exp_text","hidden_size":2560,"intermediate_size":640,
          "num_hidden_layers":1,"layer_types":["linear_attention"],
          "num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,
          "num_experts":16,"num_experts_per_tok":10,"norm_topk_prob":true,
          "moe_intermediate_size":640,"shared_expert_intermediate_size":640,
          "linear_num_key_heads":16,"linear_num_value_heads":48,
          "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
          "max_position_embeddings":262144,"vocab_size":32,"rms_norm_eps":0.000001,
          "attention_bias":false,"attention_dropout":0,
          "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25}}}
        """#.utf8))
    }
}
