import XCTest
import MLX
import MLXNN
import MereRunMLXTestSupport
@testable import MereRunGemmaModel

final class Gemma4LowBitExpertTests: MLXTestCase {
    func testExpandedQ6CodesPreserveWeightsAndVerificationArithmetic() throws {
        try XCTSkipUnless(Gemma4MetalExperts.isSupported, "Requires Metal.")
        for width in [64, 704, 2816] {
            let weight = sin(MLXArray(0..<(4 * 16 * width)).asType(.float32) * 0.017)
                .reshaped(4, 16, width).asType(.bfloat16)
            let linear = Gemma4SwitchLinear(packing: weight, groupSize: 64, bits: 6,
                prefillExpander: Gemma4ExpandedExpertWeights())
            let decoded6 = dequantized(linear.weight, scales: try XCTUnwrap(linear.scales), biases: linear.biases, groupSize: 64, bits: 6)
            let decoded8 = dequantized(try XCTUnwrap(linear.prefillWeight), scales: try XCTUnwrap(linear.scales),
                biases: linear.biases, groupSize: 64, bits: 8)
            XCTAssertEqual(abs(decoded6 - decoded8).max().item(Float.self), 0)
            let input = cos(MLXArray(0..<(16 * width)).asType(.float32) * 0.021).reshaped(16, 1, width).asType(.bfloat16)
            let indices = MLXArray((0..<16).map { Int32($0 / 4) })
            let original = gatherQuantizedMM(input, linear.weight, scales: try XCTUnwrap(linear.scales), biases: linear.biases,
                rhsIndices: indices, transpose: true, groupSize: 64, bits: 6, sortedIndices: true)
            XCTAssertEqual(abs(linear.applyFlat(input, indices: indices, sortedIndices: true) - original).max().item(Float.self), 0)
            let prefill = concatenated([input, input, input, input], axis: 0)
            let routes = MLXArray((0..<64).map { Int32($0 / 16) })
            let expected = gatherQuantizedMM(prefill, try XCTUnwrap(linear.prefillWeight), scales: try XCTUnwrap(linear.scales),
                biases: linear.biases, rhsIndices: routes, transpose: true, groupSize: 64, bits: 8, sortedIndices: true)
            XCTAssertEqual(abs(linear.applyFlat(prefill, indices: routes, sortedIndices: true) - expected).max().item(Float.self), 0)
            let oldExpanded = try XCTUnwrap(linear.prefillWeight)
            try linear.update(parameters: ModuleParameters.unflattened([
                ("weight", MLXArray.zeros(linear.weight.shape, dtype: .uint32))
            ]), verify: .none)
            let changed = linear.applyFlat(prefill, indices: routes, sortedIndices: true)
            eval(changed)
            XCTAssertFalse(linear.prefillWeight === oldExpanded)
            XCTAssertEqual(try XCTUnwrap(linear.prefillWeight).max().item(UInt32.self), 0)
            XCTAssertFalse(linear.parameters().flattened().contains { $0.0.contains("prefill") })
        }
    }

    func testLayerRecipePacksDistinctPrecisionsAndRejectsIncompleteSchedules() throws {
        let data = Data(#"{"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":3,"intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"max_position_embeddings":128,"rms_norm_eps":0.000001,"vocab_size":32,"enable_moe_block":true,"num_experts":4,"top_k_experts":2,"moe_intermediate_size":64}"#.utf8)
        let model = Gemma4TextCausalLM(config: try JSONDecoder().decode(Gemma4TextConfig.self, from: data))
        model.prepareForFullPrecisionExpertWeights()
        let original = model.languageModel.layers[0].experts!.switchGLU
        XCTAssertThrowsError(try model.prepareExperimentalExperts(groupSize: 64, layerBits: [4, 6]))
        XCTAssertTrue(model.languageModel.layers[0].experts!.switchGLU === original)
        try model.prepareExperimentalExperts(groupSize: 64, layerBits: [4, 6, 8])
        for (index, bits) in [4, 6, 8].enumerated() {
            let packed = model.languageModel.layers[index].experts!.switchGLU
            XCTAssertEqual(packed.gateUpProj!.weight.shape, [4, 128, 64 * bits / 32])
            XCTAssertEqual(packed.downProj.weight.shape, [4, 64, 64 * bits / 32])
            let ids = MLXArray([Int32(0), 1, 2, 3, 0, 1, 2, 3]).reshaped(1, 4, 2)
            let x = MLXArray.ones([1, 4, 64], dtype: .bfloat16)
            XCTAssertTrue(packed(x, indices: ids).asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
        }
    }

    func testPackedPrecisionsUseCurrentParametersAndTheirOwnArithmetic() throws {
        for bits in [4, 6, 8] {
            let weight = sin(MLXArray(0..<(4 * 16 * 704)).asType(.float32) * 0.031)
                .reshaped(4, 16, 704).asType(.bfloat16)
            let projection = Gemma4SwitchLinear(packing: weight, groupSize: 64, bits: bits)
            let x = cos(MLXArray(0..<(16 * 704)).asType(.float32) * 0.017).reshaped(16, 1, 704).asType(.bfloat16)
            let indices = MLXArray((0..<16).map { Int32($0 / 4) })
            let packed = quantized(weight, groupSize: 64, bits: bits)
            let expected = gatherQuantizedMM(x, packed.wq, scales: packed.scales, biases: packed.biases,
                rhsIndices: indices, transpose: true, groupSize: 64, bits: bits, sortedIndices: true)
            if Gemma4MetalExperts.isSupported { projection.metalVerification = try Gemma4MetalExperts() }
            let actual = projection.applyFlat(x, indices: indices, sortedIndices: true)
            XCTAssertEqual(abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self), 0)
            XCTAssertEqual(projection.weight.size, weight.size * bits / 32)
            try projection.update(parameters: ModuleParameters.unflattened([
                ("scales", MLXArray.zeros(try XCTUnwrap(projection.scales).shape, dtype: .bfloat16)),
                ("biases", MLXArray.zeros(try XCTUnwrap(projection.biases).shape, dtype: .bfloat16))
            ]), verify: .none)
            XCTAssertEqual(abs(projection.applyFlat(x, indices: indices, sortedIndices: true)).max().item(Float.self), 0)
        }
    }
}
