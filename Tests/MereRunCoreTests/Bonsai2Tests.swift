import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore
@testable import MereRunQwenModel

final class Bonsai2Tests: MereRunCoreTestCase {
    func testCatalogPinsDistinctMLXPackAndDefaults() throws {
        let id = Q35Resources.bonsai2ModelId
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))
        XCTAssertEqual(spec.upstreamRepoId, "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit")
        XCTAssertEqual(spec.upstreamRevision, "3f926b415992eaa2ae9dd7b573706494d6bbf787")
        XCTAssertEqual(spec.hubFallback?.revision, spec.upstreamRevision)
        XCTAssertEqual(Q35Resources.defaultContextLength(forModelId: id), 262_144)
        XCTAssertTrue(Q35Resources.thinkingDefault(forModelId: id))
        XCTAssertNil(Q35Resources.recommendedSampling(forModelId: id))
        let profile = try XCTUnwrap(Q35Resources.profile(for: id))
        XCTAssertTrue(profile.snapshotPatterns.contains("hadamard.json"))
        XCTAssertTrue(profile.snapshotPatterns.contains("preprocessor_config.json"))
        let manifest = MereRunModelManifest.template(for: .bonsai2, createdAt: Date(timeIntervalSince1970: 0))
        XCTAssertEqual(manifest.quantization?.scheme, "prism-hadamard-affine-ternary")
        XCTAssertEqual(manifest.quantization?.bits, 2)
        XCTAssertNotEqual(spec.upstreamRevision, Q35Resources.bonsai27B2BitUpstreamRevision)
    }

    func testSignedTransformMatchesIndependentSylvesterSum() {
        let width = 512
        let values = (0..<width).map { Float($0 % 17 - 8) / 16 }
        let signValues = (0..<width).map { $0 % 3 == 0 ? Float(-1) : Float(1) }
        let output = Q35PrismTransform.apply(MLXArray(values), block: width, signs: MLXArray(signValues))
        let actual = output.asArray(Float.self)
        for row in 0..<width {
            var expected: Float = 0
            for column in 0..<width {
                let sign: Float = (row & column).nonzeroBitCount.isMultiple(of: 2) ? 1 : -1
                expected += sign * values[column] * signValues[column]
            }
            XCTAssertEqual(actual[row], expected / Float(width).squareRoot(), accuracy: 0.00002)
        }
        let restored = Q35PrismTransform.apply(output, block: width, signs: MLXArray(signValues), inverse: true)
        XCTAssertLessThan(MLX.max(MLX.abs(restored - MLXArray(values))).item(Float.self), 0.00001)
    }

    func testPackedProjectionAndInverseEmbeddingPreserveRotation() {
        let width = 512
        let weights = MLXArray(Array(repeating: UInt32(0xAAAA_AAAA), count: width / 16), [1, width / 16])
        let scales = MLXArray.ones([1, width / 128])
        let biases = -MLXArray.ones([1, width / 128])
        let signs = MLXArray((0..<width).map { $0 == 0 ? Float(-1) : Float(1) })
        let projection = Q35PrismLinear(weight: weights, scales: scales, biases: biases, signs: signs, block: width)
        let input = MLXArray.ones([1, 2, width])
        // A row of ones in the rotated basis corresponds to sqrt(width) at input zero.
        let logits = projection(input)
        XCTAssertEqual(logits.shape, [1, 2, 1])
        XCTAssertEqual(logits[0, 0, 0].item(Float.self), -Float(width).squareRoot(), accuracy: 0.0001)
        let embedding = Q35PrismEmbedding(weight: weights, scales: scales, biases: biases, signs: signs, block: width)
        let rows = embedding(MLXArray([Int32(0), Int32(0)], [1, 2]))
        XCTAssertEqual(rows.dtype, .float16)
        XCTAssertEqual(rows.shape, [1, 2, width])
        XCTAssertEqual(rows[0, 0, 0].item(Float.self), -Float(width).squareRoot(), accuracy: 0.01)
        XCTAssertEqual(MLX.max(MLX.abs(rows[0, 0, 1...])).item(Float.self), 0)
    }

    func testPackValidationRejectsMissingSignsDuplicatesAndWrongLayout() throws {
        let model = Fixture()
        var arrays: [String: MLXArray] = [
            "projection.weight": MLXArray.zeros([2, 32], dtype: .uint32),
            "projection.scales": MLXArray.ones([2, 4]),
            "projection.biases": MLXArray.zeros([2, 4]),
            "projection.signs": MLXArray.ones([512]),
        ]
        let contract = try configuration()
        XCTAssertEqual(try contract.replacements(model: model, arrays: arrays).count, 1)
        arrays.removeValue(forKey: "projection.signs")
        XCTAssertThrowsError(try contract.replacements(model: model, arrays: arrays))
        arrays["projection.signs"] = MLXArray.zeros([512])
        XCTAssertThrowsError(try contract.replacements(model: model, arrays: arrays))
        arrays["projection.signs"] = MLXArray.ones([512])
        XCTAssertThrowsError(try configuration(duplicate: true).replacements(model: model, arrays: arrays))
        XCTAssertThrowsError(try configuration(layout: "interleaved").replacements(model: model, arrays: arrays))
        arrays["unlisted.scales"] = MLXArray.ones([1, 4])
        XCTAssertThrowsError(try contract.replacements(model: model, arrays: arrays))
    }

    func testNativeLoaderInstallsRotatedModulesAndRejectsMissingOrdinaryWeights() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let configData = Data("""
        {"schema_version":2,"model_type":"prism_hadamard_qwen35",
         "tensor_namespace":"mlx-vlm-qwen3_5","gdn_activation_layout":"grouped",
         "quantization":{"bits":2,"group_size":128},"tie_word_embeddings":false,
         "text_config":{"model_type":"qwen3_5_text","hidden_size":512,"num_hidden_layers":0,
          "intermediate_size":512,"layer_types":[],"num_attention_heads":4,"num_key_value_heads":1,
          "head_dim":128,"vocab_size":8,"max_position_embeddings":1024,"rms_norm_eps":0.000001,
          "linear_num_value_heads":4,"linear_num_key_heads":1,"linear_key_head_dim":128,
          "linear_value_head_dim":128,"linear_conv_kernel_dim":4,"attention_bias":false,
          "attention_dropout":0.0,"rope_parameters":{"rope_theta":10000,"partial_rotary_factor":0.25}},
         "modules":[
          {"path":"model.embed_tokens","block":512,"embedding":true,"dtype":"float16"},
          {"path":"lm_head","block":512,"embedding":false,"dtype":"float16"}]}
        """.utf8)
        try configData.write(to: root.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Q35Config.self, from: configData)
        var arrays: [String: MLXArray] = ["language_model.model.norm.weight": MLXArray.ones([512])]
        for path in ["model.embed_tokens", "lm_head"] {
            let base = "language_model." + path
            arrays[base + ".weight"] = MLXArray.zeros([8, 32], dtype: .uint32)
            arrays[base + ".scales"] = MLXArray.ones([8, 4])
            arrays[base + ".biases"] = MLXArray.zeros([8, 4])
            arrays[base + ".signs"] = MLXArray.ones([512])
        }
        let weightsURL = root.appendingPathComponent("model.safetensors")
        try MLX.save(arrays: arrays, url: weightsURL)
        let model = Q35Model(config: config)
        let generator = Q35Generator(modelId: Q35Resources.bonsai2ModelId)
        try generator.loadTextWeights(into: model, from: Q35Resources(rootURL: root), groupSize: 128, bits: 2)
        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        XCTAssertTrue(leaves["model.embed_tokens"] is Q35PrismEmbedding)
        XCTAssertTrue(leaves["lm_head"] is Q35PrismLinear)
        let norm = try XCTUnwrap(model.parameters().flattened().first { $0.0 == "model.norm.weight" }?.1)
        XCTAssertEqual(MLX.max(MLX.abs(norm)).item(Float.self), 0)
        arrays.removeValue(forKey: "language_model.model.norm.weight")
        try MLX.save(arrays: arrays, url: weightsURL)
        do {
            try generator.loadTextWeights(
                into: Q35Model(config: config), from: Q35Resources(rootURL: root), groupSize: 128, bits: 2
            )
            XCTFail("A missing normalization weight must fail before generation")
        } catch Q35Error.generationFailed(let message) {
            XCTAssertTrue(message.contains("model.norm.weight"))
        }
    }

    func testHybridInventoryRequiresOnlyActiveAttentionAndCheckpointParameters() throws {
        let data = Data("""
        {"model_type":"prism_hadamard_qwen35","text_config":{
         "model_type":"qwen3_5_text","hidden_size":512,"num_hidden_layers":2,
         "intermediate_size":512,"layer_types":["linear_attention","full_attention"],
         "num_attention_heads":4,"num_key_value_heads":1,"head_dim":128,"vocab_size":8,
         "max_position_embeddings":1024,"rms_norm_eps":0.000001,
         "linear_num_value_heads":4,"linear_num_key_heads":1,"linear_key_head_dim":128,
         "linear_value_head_dim":128,"linear_conv_kernel_dim":4,"attention_bias":false,
         "attention_dropout":0.0,"rope_parameters":{"rope_theta":10000,"partial_rotary_factor":0.25}}}
        """.utf8)
        let model = Q35Model(config: try JSONDecoder().decode(Q35Config.self, from: data))
        let names = try configuration().checkpointParameterNames(model: model)
        XCTAssertTrue(names.contains("model.layers.0.linear_attn.in_proj_a.weight"))
        XCTAssertTrue(names.contains("model.layers.0.linear_attn.conv1d.weight"))
        XCTAssertTrue(names.contains("model.layers.1.self_attn.q_norm.weight"))
        XCTAssertTrue(names.contains("model.layers.1.self_attn.q_proj.weight"))
        XCTAssertFalse(names.contains { $0.hasPrefix("model.layers.0.self_attn.") })
        XCTAssertFalse(names.contains { $0.hasPrefix("model.layers.1.linear_attn.") })
        XCTAssertFalse(names.contains { $0.hasSuffix("qkNormWeightBF16") })
        XCTAssertTrue(names.contains("model.layers.0.input_layernorm.weight"))
        XCTAssertTrue(names.contains("model.layers.1.mlp.gate_proj.weight"))
        XCTAssertTrue(names.contains("model.norm.weight"))
    }

    private final class Fixture: Module {
        @ModuleInfo var projection = Linear(512, 2, bias: false)
    }

    private func configuration(duplicate: Bool = false, layout: String = "grouped") throws -> Q35PrismConfiguration {
        let record = #"{"path":"projection","block":512,"embedding":false,"dtype":"float16"}"#
        let records = duplicate ? "\(record),\(record)" : record
        return try JSONDecoder().decode(Q35PrismConfiguration.self, from: Data("""
        {"schema_version":2,"model_type":"prism_hadamard_qwen35",
         "tensor_namespace":"mlx-vlm-qwen3_5","gdn_activation_layout":"\(layout)",
         "quantization":{"bits":2,"group_size":128},"tie_word_embeddings":false,
         "modules":[\(records)]}
        """.utf8))
    }
}
