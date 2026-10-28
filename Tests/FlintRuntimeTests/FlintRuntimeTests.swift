import Foundation
import MLX
import MLXNN
import XCTest
import MereRunMLXTestSupport
@testable import MereRunFlintModel

final class FlintRuntimeTests: MLXTestCase {
    func testPublishedConfiguration() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")
        let config = try JSONDecoder().decode(FlintConfiguration.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        try config.validate()
        XCTAssertEqual(config.numHiddenLayers, 49)
        XCTAssertEqual(config.layerTypes.filter { $0 == .sliding }.count, 35)
        XCTAssertEqual(config.eosTokenID, [150001, 150003, 150005, 150018])
    }

    private var fixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")
    }

    private func configuration(_ name: String = "tiny-config.json") throws -> FlintConfiguration {
        try JSONDecoder().decode(FlintConfiguration.self, from: Data(contentsOf: fixtures.appendingPathComponent(name)))
    }

    func testPublishedINT4ConfigurationAndEmbeddingPolicy() throws {
        let config = try configuration("int4-config.json")
        try config.validate()
        XCTAssertEqual(config.policy(for: "model.embed_tokens")?.bits, 8)
        XCTAssertEqual(config.policy(for: "lm_head")?.bits, 4)
        XCTAssertEqual(config.policy(for: "model.layers.2.mlp.switch_mlp")?.groupSize, 64)
    }

    func testIndependentScalarReferenceAndChunkedCacheParity() throws {
        struct Reference: Decodable { let tokens: [Int32]; let logits: [Float] }
        let config = try configuration()
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: fixtures.appendingPathComponent("reference.json")))
        let model = try FlintCausalLM(config: config)
        let weights = try loadArrays(url: fixtures.appendingPathComponent("weights.safetensors"))
        XCTAssertEqual(Set(weights.keys), Set(model.parameters().flattened().map(\.0)))
        try model.update(parameters: .unflattened(weights), verify: [.allModelKeysSet, .shapeMismatch])
        let expected = MLXArray(reference.logits).reshaped(1, reference.tokens.count, config.vocabSize)
        let full = model(MLXArray(reference.tokens).reshaped(1, -1))
        XCTAssertLessThan(abs(full - expected).max().item(Float.self), 2e-5)
        for chunkSize in [1, 3, 8, 13] {
            let cache = model.makeCache()
            var parts: [MLXArray] = []
            for start in stride(from: 0, to: reference.tokens.count, by: chunkSize) {
                let end = min(start + chunkSize, reference.tokens.count)
                parts.append(model(MLXArray(Array(reference.tokens[start..<end])).reshaped(1, -1), cache: cache))
            }
            XCTAssertLessThan(abs(concatenated(parts, axis: 1) - expected).max().item(Float.self), 2e-5)
            XCTAssertEqual(cache[1].keys?.dim(2), config.slidingWindow - 1)
            XCTAssertEqual(cache[1].offset, reference.tokens.count)
            XCTAssertEqual(cache[0].keys?.dim(2), reference.tokens.count)
        }
    }

    func testRouterBiasChangesProbabilitySelectionOnly() throws {
        let router = FlintRouter(config: try configuration())
        let logits: [Float] = [-2, -1, 1, 2]
        try router.update(parameters: .unflattened([
            "weight": concatenated([MLXArray(logits).reshaped(4, 1), MLXArray.zeros([4, 7])], axis: 1),
            "expert_bias": MLXArray([Float(0.8), 0.6, 0, 0])
        ]), verify: [.allModelKeysSet, .shapeMismatch])
        let input = MLXArray([Float(1), 0, 0, 0, 0, 0, 0, 0]).reshaped(1, 1, 8)
        let routed = router(input)
        let ids = routed.indices.asArray(Int32.self)
        XCTAssertEqual(Set(ids), Set([0, 3]))
        let denominator = 1 / (1 + exp(Float(2))) + 1 / (1 + exp(Float(-2)))
        for (index, weight) in zip(ids, routed.weights.asArray(Float.self)) {
            XCTAssertEqual(weight, 1 / (1 + exp(-logits[Int(index)])) / denominator, accuracy: 1e-6)
        }
    }

    func testFixedCapacityCacheMatchesReferenceAcrossSlidingBoundaries() throws {
        struct Reference: Decodable { let tokens: [Int32]; let logits: [Float] }
        let config = try configuration()
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: fixtures.appendingPathComponent("reference.json")))
        let model = try FlintCausalLM(config: config)
        try model.update(parameters: .unflattened(loadArrays(url: fixtures.appendingPathComponent("weights.safetensors"))),
                         verify: [.allModelKeysSet, .shapeMismatch])
        let expected = MLXArray(reference.logits).reshaped(1, reference.tokens.count, config.vocabSize)
        for chunkSize in [1, 3, 8, 13] {
            let cache = model.makeCache(maxTokens: reference.tokens.count + 5)
            var parts: [MLXArray] = []
            for start in stride(from: 0, to: reference.tokens.count, by: chunkSize) {
                let end = min(start + chunkSize, reference.tokens.count)
                parts.append(model(MLXArray(Array(reference.tokens[start..<end])).reshaped(1, -1), cache: cache))
                XCTAssertEqual(cache[0].keys?.dim(2), end)
            }
            XCTAssertLessThan(abs(concatenated(parts, axis: 1) - expected).max().item(Float.self), 2e-5)
            XCTAssertEqual(cache[0].capacity, reference.tokens.count + 5)
            XCTAssertNil(cache[1].capacity)
            XCTAssertEqual(cache[1].keys?.dim(2), config.slidingWindow - 1)
        }
    }

    func testAsynchronousDecodeMatchesLayerwiseEvaluation() throws {
        let config = try configuration()
        let model = try FlintCausalLM(config: config)
        try model.update(parameters: .unflattened(loadArrays(url: fixtures.appendingPathComponent("weights.safetensors"))),
                         verify: [.allModelKeysSet, .shapeMismatch])
        let tokens: [Int32] = [1, 3, 2, 5, 1, 4, 2, 1]
        let run = { (asynchronous: Bool) -> MLXArray in
            model.asynchronousDecode = asynchronous
            let cache = model.makeCache(maxTokens: tokens.count)
            let parts = tokens.map { token in
                let logits = model(MLXArray([token]).reshaped(1, 1), cache: cache)
                eval(logits)
                return logits
            }
            return concatenated(parts, axis: 1)
        }
        XCTAssertLessThan(abs(run(false) - run(true)).max().item(Float.self), 2e-5)
    }

    func testSinkAddsOnlyToSoftmaxDenominator() throws {
        let config = try configuration()
        let attention = FlintAttention(config: config, index: 0)
        var weights = Dictionary(uniqueKeysWithValues: attention.parameters().flattened())
        for (key, value) in weights { weights[key] = MLXArray.zeros(value.shape, dtype: .float32) }
        weights["q_norm.weight"] = MLXArray.ones([4])
        weights["k_norm.weight"] = MLXArray.ones([4])
        weights["v_proj.weight"] = MLXArray.eye(8)
        weights["o_proj.weight"] = concatenated([MLXArray.eye(8), MLXArray.zeros([8, 8])], axis: 1)
        try attention.update(parameters: .unflattened(weights), verify: [.allModelKeysSet, .shapeMismatch])
        let input = MLXArray.ones([1, 1, 8])
        // Zero Q/K scores and a zero sink split the mass equally; the sink has no value.
        XCTAssertLessThan(abs(attention(input, cache: nil) - 0.5).max().item(Float.self), 1e-6)
        XCTAssertNil(attention.rope)
        XCTAssertNotNil(FlintAttention(config: config, index: 1).rope)
    }

    func testPackedProjectionAgreesWithDequantizedWeights() throws {
        let policy = FlintConfiguration.Quantization(bits: 4, groupSize: 64)
        let projection = FlintProjection(input: 64, output: 8, policy: policy)
        let dense = sin(MLXArray(0..<512).asType(.float32)).reshaped(8, 64)
        let (weight, scales, optionalBiases) = quantized(dense, groupSize: 64, bits: 4)
        let biases = try XCTUnwrap(optionalBiases)
        try projection.update(parameters: .unflattened(["weight": weight, "scales": scales, "biases": biases]), verify: .shapeMismatch)
        let input = MLXArray.ones([1, 64])
        let expected = matmul(input, dequantized(weight, scales: scales, biases: biases, groupSize: 64, bits: 4).T)
        XCTAssertLessThan(abs(projection(input) - expected).max().item(Float.self), 1e-5)
        let bytes = weight.view(dtype: .uint8)
        XCTAssertEqual(bytes.view(dtype: .uint32).asArray(UInt32.self), weight.asArray(UInt32.self))
    }

    func testRoutedBF16QuantizedProjectionIsRepeatableOnGPU() throws {
        try XCTSkipIf(Device.defaultDevice().deviceType != .gpu, "Exercises the GPU quantized matrix kernel.")
        let projection = FlintProjection(input: 2048, output: 512, experts: 8,
                                         policy: .init(bits: 4, groupSize: 64))
        let dense = sin(MLXArray(0..<(8 * 512 * 2048)).asType(.float32) / 1000)
            .reshaped(8, 512, 2048).asType(.bfloat16)
        let (weight, scales, optionalBiases) = quantized(dense, groupSize: 64, bits: 4)
        let biases = try XCTUnwrap(optionalBiases)
        try projection.update(parameters: .unflattened(["weight": weight, "scales": scales, "biases": biases]),
                              verify: [.allModelKeysSet, .shapeMismatch])
        eval(weight, scales, biases)
        let input = cos(MLXArray(0..<(5 * 2048)).asType(.float32) / 100)
            .reshaped(1, 5, 1, 2048).asType(.bfloat16)
        let indices = (MLXArray(0..<30) % 8).reshaped(1, 5, 6)
        let expected = projection(input, indices: indices)
        eval(expected)
        for _ in 0..<20 {
            let actual = projection(input, indices: indices)
            eval(actual)
            XCTAssertEqual(abs(actual - expected).max().item(Float.self), 0)
        }
    }

    func testGPUBF16QuantizedProjectionAccumulatesCancellationInFloat32() throws {
        try XCTSkipIf(Device.defaultDevice().deviceType != .gpu, "Exercises GPU quantized accumulation precision.")
        let projection = FlintProjection(input: 2048, output: 8, policy: .init(bits: 4, groupSize: 64))
        let weight = MLXArray(Array(repeating: UInt32(0x11111111), count: 8 * 256)).reshaped(8, 256)
        try projection.update(parameters: .unflattened([
            "weight": weight,
            "scales": MLXArray.ones([8, 32], dtype: .bfloat16),
            "biases": MLXArray.zeros([8, 32], dtype: .bfloat16)
        ]), verify: [.allModelKeysSet, .shapeMismatch])
        let phases: [Float] = [256, 1, -256, 1]
        let values = phases.flatMap { Array(repeating: $0, count: 512) }
        let input = MLXArray(values).reshaped(1, 2048).asType(.bfloat16)
        // Each weight is one: 512 * (256 + 1 - 256 + 1) = 1024.
        // BF16 partial sums round 256 + 1 to 256 and incorrectly return 512.
        XCTAssertEqual(projection(input).asArray(Float.self), Array(repeating: Float(1024), count: 8))
    }

    func testGPUBF16AffineDequantizationRoundsAfterAddingBias() throws {
        try XCTSkipIf(Device.defaultDevice().deviceType != .gpu, "Exercises GPU affine dequantization precision.")
        let projection = FlintProjection(input: 2048, output: 128, policy: .init(bits: 4, groupSize: 64))
        try projection.update(parameters: .unflattened([
            "weight": MLXArray(Array(repeating: UInt32(0x77777777), count: 128 * 256)).reshaped(128, 256),
            "scales": MLXArray(Array(repeating: Float(0.10009765625), count: 128 * 32)).reshaped(128, 32).asType(.bfloat16),
            "biases": MLXArray(Array(repeating: Float(-0.69921875), count: 128 * 32)).reshaped(128, 32).asType(.bfloat16)
        ]), verify: [.allModelKeysSet, .shapeMismatch])
        for rows in [1, 16] {
            let input = MLXArray.ones([rows, 2048], dtype: .bfloat16)
            // 2048 * (7 * 0.10009765625 - 0.69921875) = 3 exactly.
            // Rounding the product to BF16 before adding bias erases each weight.
            XCTAssertEqual(projection(input).asArray(Float.self), Array(repeating: Float(3), count: rows * 128))
        }
    }

    func testEveryPublishedINT4TensorMatchesNativeParameterLayout() throws {
        struct Tensor: Decodable { let shape: [Int]; let dtype: String }
        let config = try configuration("int4-config.json")
        let model = try FlintCausalLM(config: config)
        let expected = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let layout = try JSONDecoder().decode([String: Tensor].self, from: Data(contentsOf: fixtures.appendingPathComponent("int4-layout.json")))
        let optional = Set(expected.keys.filter { $0.hasSuffix(".gate.expert_bias") })
        XCTAssertEqual(Set(layout.keys), Set(expected.keys).subtracting(optional))
        for (key, tensor) in layout {
            let native = try XCTUnwrap(expected[key], key)
            var shape = tensor.shape
            if tensor.dtype == "U8" {
                shape[shape.count - 1] /= 4
                XCTAssertEqual(native.dtype, .uint32, key)
            } else {
                XCTAssertEqual(tensor.dtype, "BF16", key)
                XCTAssertEqual(native.dtype, .bfloat16, key)
            }
            XCTAssertEqual(shape, native.shape, key)
        }
    }

    func testINT8EmbeddingPreservesRepeatedTokenOrder() throws {
        let config = try configuration("tiny-int4-config.json")
        try config.validate()
        let embedding = FlintEmbedding(config: config)
        let dense = sin(MLXArray(0..<(17 * 64)).asType(.float32)).reshaped(17, 64)
        let (weight, scales, optionalBiases) = quantized(dense, groupSize: 64, bits: 8)
        let biases = try XCTUnwrap(optionalBiases)
        try embedding.update(parameters: .unflattened(["weight": weight, "scales": scales, "biases": biases]),
                             verify: [.allModelKeysSet, .shapeMismatch])
        let tokens = MLXArray([Int32(3), 1, 3]).reshaped(1, 3)
        let expected = dequantized(weight, scales: scales, biases: biases, groupSize: 64, bits: 8).take(tokens, axis: 0)
        XCTAssertEqual(embedding(tokens).shape, [1, 3, 64])
        XCTAssertLessThan(abs(embedding(tokens) - expected).max().item(Float.self), 1e-6)
    }
}
