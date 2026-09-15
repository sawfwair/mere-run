import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class MarigoldV2QuantizationTests: MereRunCoreTestCase {
    func testQuantizationPreservesFirstImageModulationWeightsAndResponse() throws {
        let config = try JSONDecoder().decode(QwenImageEditTransformerConfig.self, from: Data("""
        {"num_attention_heads":2,"attention_head_dim":64,"num_layers":2,
         "joint_attention_dim":192,"axes_dims_rope":[16,24,24]}
        """.utf8))
        let model = MMDiT(config: config)
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        let path = MarigoldV2TransformerQuantization.firstImageModulationPath
        let original = try XCTUnwrap(model.leafModules().flattened().first { $0.0 == path }?.1 as? Linear)
        let weight = original.weight
        let input = MLXArray(0..<128).asType(.bfloat16).reshaped(1, 128) / 128
        let expected = original(input)
        MLX.eval(weight, expected)

        MarigoldV2TransformerQuantization.apply(to: model)

        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let preserved = try XCTUnwrap(modules[path] as? Linear)
        XCTAssertFalse(preserved is QuantizedLinear)
        XCTAssertEqual(preserved.weight.dtype, .bfloat16)
        XCTAssertEqual(MLX.max(MLX.abs(preserved.weight - weight)).item(Float.self), 0)
        XCTAssertEqual(MLX.max(MLX.abs(preserved(input) - expected)).item(Float.self), 0)
        for quantizedPath in [
            "transformer_blocks.0.adaLN_modulation_context.linear",
            "transformer_blocks.1.adaLN_modulation.linear",
            "transformer_blocks.0.attn.to_q"
        ] {
            XCTAssertTrue(modules[quantizedPath] is QuantizedLinear, quantizedPath)
        }
    }

    func testPrequantizedCheckpointRejectsTheExcludedProjection() throws {
        let path = MarigoldV2TransformerQuantization.firstImageModulationPath
        let factory = DenseLayerFactory(
            arrays: ["\(path).scales": MLXArray.ones([1, 1]), "\(path).biases": MLXArray.zeros([1, 1])],
            quantConfig: QuantizationConfig(bits: 4, groupSize: 64)
        )
        XCTAssertThrowsError(try MarigoldV2TransformerQuantization.validate(factory: factory))
        XCTAssertNoThrow(try MarigoldV2TransformerQuantization.validate(factory: .standard))
    }
}
