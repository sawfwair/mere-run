import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import MereRunCore

final class Q38MoERoutingTests: MereRunCoreTestCase {
    func testExpertSelectionRetainsFP32ProbabilitiesUntilAfterNormalization() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("BF16 expert routing parity requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }
        for topK in [1, 10] {
            try qualifyExpertSelection(topK: topK)
        }
    }

    private func qualifyExpertSelection(topK: Int) throws {
        let config = try JSONDecoder().decode(Q35Config.self, from: Data(#"""
        {"model_type":"qwen4_exp","text_config":{
          "model_type":"qwen4_exp_text","hidden_size":64,"intermediate_size":64,
          "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
          "layer_types":["linear_attention"],"num_experts":512,"num_experts_per_tok":\#(topK),
          "moe_intermediate_size":64,"shared_expert_intermediate_size":64,"norm_topk_prob":true,
          "linear_num_value_heads":1,"linear_num_key_heads":1,
          "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
          "max_position_embeddings":262144,"vocab_size":32,"rms_norm_eps":0.000001,
          "attention_bias":false,"attention_dropout":0,
          "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25}}}
        """#.utf8))
        MLXRandom.seed(113)
        let feedForward = Q35FeedForward(config: config)
        feedForward.update(parameters: feedForward.parameters().mapValues { $0.asType(.bfloat16) })
        // Distinct BF16 logits become identical if softmax probabilities are
        // rounded before top-k. The best experts are deliberately first, not
        // at the tail selected from an all-equal sorted array.
        let logits = MLXArray((0..<512).map { Float(max(10 - $0, 0)) * 0.00005 }).asType(.bfloat16)
        let gateWeight = MLX.concatenated([
            logits.reshaped(512, 1), MLXArray.zeros([512, 63], dtype: .bfloat16),
        ], axis: 1)
        feedForward.update(modules: ModuleChildren.unflattened([
            ("gate", Linear(weight: gateWeight, bias: nil)),
        ]))
        let input = MLXArray([Float(1)] + [Float](repeating: 0, count: 63))
            .reshaped(1, 1, 64).asType(.bfloat16)
        let gate = try XCTUnwrap(feedForward.gate)
        let experts = try XCTUnwrap(feedForward.switchMLP)
        let shared = try XCTUnwrap(feedForward.sharedExpert)
        let sharedGate = try XCTUnwrap(feedForward.sharedExpertGate)
        let routerLogits = gate(input)
        let rounded = MLX.softmax(routerLogits, axis: -1, precise: true)
        XCTAssertEqual((rounded.max() - rounded.min()).item(Float.self), 0)

        let probabilities = MLX.softmax(routerLogits.asType(.float32), axis: -1, precise: true)
        let indices = MLX.argPartition(probabilities, kth: -topK, axis: -1)[.ellipsis, (-topK)...].asType(.int32)
        XCTAssertEqual(Set(indices.asArray(Int32.self)), Set((0..<topK).map(Int32.init)))
        let selected = MLX.takeAlong(probabilities, indices, axis: -1)
        let scores = (selected / selected.sum(axis: -1, keepDims: true)).asType(routerLogits.dtype)
        let expected = (experts(input, indices: indices) * scores.expandedDimensions(axis: -1)).sum(axis: -2)
            + MLX.sigmoid(sharedGate(input)) * shared(input)
        let actual = feedForward(input)
        XCTAssertEqual(
            (actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self), 0,
            "Flash-Next must select and normalize experts in FP32 before restoring the model dtype"
        )
    }
}
