import Foundation
import XCTest
import MLX
import MLXNN
import MLXRandom
import MereRunMLXTestSupport
@testable import MereRunGemmaModel

final class Gemma4AdapterBoundaryTests: MLXTestCase {
    func testReplacingProjectionInvalidatesRetainedFusionAndChangesOutput() throws {
        guard Gemma4FusedProjectionPolicy.enabled else {
            throw XCTSkip("Projection fusion is disabled by MERERUN_GEMMA4_FUSED_PROJ.")
        }
        let configJSON = """
            {
              "model_type": "gemma4_text", "hidden_size": 64,
              "num_hidden_layers": 1, "intermediate_size": 64,
              "num_attention_heads": 2, "num_key_value_heads": 1,
              "head_dim": 32, "max_position_embeddings": 128,
              "rms_norm_eps": 0.000001, "vocab_size": 32
            }
            """
        let config = try JSONDecoder().decode(Gemma4TextConfig.self, from: Data(configJSON.utf8))
        MLXRandom.seed(83)
        let layer = Gemma4MLP(config: config, layerIndex: 0)
        let gate = QuantizedLinear(weight: MLXRandom.normal([64, 64]) * 0.1, bias: nil, groupSize: 64, bits: 4)
        let up = QuantizedLinear(weight: MLXRandom.normal([64, 64]) * 0.1, bias: nil, groupSize: 64, bits: 4)
        layer.update(modules: ModuleChildren.unflattened([
            ("gate_proj", gate as Module), ("up_proj", up as Module)
        ]))
        let input = MLXRandom.normal([1, 1, 64])
        let original = layer(input)
        MLX.eval(original)
        XCTAssertNotNil(layer.resolvedFusedGateUp())
        XCTAssertGreaterThan(MLX.max(MLX.abs(original)).item(Float.self), 1e-5)

        // Adapter injection replaces modules through the same Module update boundary.
        let replacement = Linear(weight: MLXArray.zeros([64, 64]), bias: nil)
        layer.update(modules: ModuleChildren.unflattened([("gate_proj", replacement as Module)]))
        let adapted = layer(input)
        MLX.eval(adapted)
        XCTAssertNil(layer.resolvedFusedGateUp())
        XCTAssertEqual(MLX.max(MLX.abs(adapted)).item(Float.self), 0, accuracy: 1e-6)
    }
}
