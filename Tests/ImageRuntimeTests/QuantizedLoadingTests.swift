import Foundation
import MLX
import MLXNN
import MereRunMLXTestSupport
import XCTest
@testable import MereRunTensor

final class QuantizedLoadingTests: MLXTestCase {
    private final class Layers: Module {
        @ModuleInfo var layers: [Linear]
        @ParameterInfo var gain: MLXArray

        override init() {
            _layers.wrappedValue = [Linear(32, 4, bias: true), Linear(4, 2, bias: false)]
            _gain.wrappedValue = MLXArray(Float(1))
            super.init()
        }
    }

    func testShardedAndArrayLoadingPreserveArraySiblingsAndNumericalOutput() throws {
        let root = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let dense = MLXArray((0..<128).map { Float($0 % 17 - 8) / 16 }, [4, 32])
        let (weight, scales, biases) = MLX.quantized(dense, groupSize: 32, bits: 4)
        let bias = MLXArray([Float(1), 2, 3, 4])
        let siblingWeight = MLXArray([Float(1), 0, 0, 1, 0, 1, 1, 0], [2, 4])
        let arrays = [
            "source.layers.0.weight": weight,
            "source.layers.0.scales": scales,
            "source.layers.0.biases": try XCTUnwrap(biases),
            "source.layers.0.bias": bias,
            "source.layers.1.weight": siblingWeight,
            "source.checkpoint_gain": MLXArray(Float(2))
        ]
        let shard = root.appendingPathComponent("weights.safetensors")
        try MLX.save(arrays: arrays, url: shard)
        let index = HFSafetensorsIndex(
            metadata: nil,
            weightMap: arrays.mapValues { _ in shard.lastPathComponent }
        )
        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        try JSONEncoder().encode(index).write(to: indexURL)
        let keyMapper: (String) -> String = { String($0.dropFirst("source.".count)) }
        let mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            [(key == "checkpoint_gain" ? "gain" : key, value)]
        }
        let fromArray = Layers()
        let fromShard = Layers()
        let sibling = fromArray.layers[1]
        try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
            arrays, to: fromArray, groupSize: 64, bits: 8,
            keyMapper: keyMapper, mapper: mapper
        )
        try HFSafetensorsWeightsLoader.applyQuantizedWeights(
            indexURL: indexURL, to: fromShard, groupSize: 64, bits: 8,
            keyMapper: keyMapper, mapper: mapper
        )

        XCTAssertTrue(fromArray.layers[1] === sibling)
        XCTAssertEqual(fromArray.layers.count, 2)
        let quantized = try XCTUnwrap(fromArray.layers[0] as? PortableQuantizedLinear)
        XCTAssertEqual(quantized.bits, 4)
        XCTAssertEqual(quantized.groupSize, 32)
        XCTAssertEqual(fromArray.gain.item(Float.self), 2)
        let input = MLXArray((0..<64).map { Float($0 % 7) / 8 }, [2, 32])
        let restoredWeight = MLX.dequantized(weight, scales: scales, biases: biases, groupSize: 32, bits: 4)
        let expected = MLX.matmul(MLX.matmul(input, restoredWeight.T) + bias, siblingWeight.T)
        for model in [fromArray, fromShard] {
            let output = model.layers[1](model.layers[0](input))
            XCTAssertLessThan(MLX.max(MLX.abs(output - expected)).item(Float.self), 1e-5)
        }
    }

    func testResidualLoadingMatchesDenseCorrectionAndCanBeDisabled() throws {
        let dense = MLXArray((0..<128).map { Float($0 % 11 - 5) / 8 }, [4, 32])
        let (weight, scales, biases) = MLX.quantized(dense, groupSize: 32, bits: 4)
        let down = MLXArray.ones([1, 32]) * 0.125
        let up = MLXArray([Float(1), 2, 3, 4], [4, 1])
        let arrays = [
            "layers.0.weight": weight, "layers.0.scales": scales,
            "layers.0.biases": try XCTUnwrap(biases),
            "layers.0.svd_down": down, "layers.0.svd_up": up
        ]
        let corrected = Layers()
        let plain = Layers()
        try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
            arrays, to: corrected, applySVDResiduals: true
        )
        try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
            arrays, to: plain, applySVDResiduals: false
        )
        XCTAssertTrue(corrected.layers[0] is ResidualQuantizedLinear)
        XCTAssertTrue(plain.layers[0] is PortableQuantizedLinear)
        let input = MLXArray.ones([1, 32])
        let correction = MLX.matmul(MLX.matmul(input, down.T), up.T)
        let actualCorrection = corrected.layers[0](input) - plain.layers[0](input)
        XCTAssertLessThan(MLX.max(MLX.abs(actualCorrection - correction)).item(Float.self), 1e-5)
    }

    func testQuantizedEmbeddingRestoresLookupAndTiedProjection() throws {
        final class Model: Module {
            @ModuleInfo var embedding: Embedding = Embedding(embeddingCount: 4, dimensions: 32)
        }
        let dense = MLXArray((0..<128).map { Float($0 % 13) / 16 }, [4, 32])
        let (weight, scales, biases) = MLX.quantized(dense, groupSize: 32, bits: 4)
        let model = Model()
        try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays([
            "embedding.weight": weight, "embedding.scales": scales,
            "embedding.biases": try XCTUnwrap(biases)
        ], to: model)
        XCTAssertTrue(model.embedding is PreQuantizedEmbedding)
        let restoredWeight = MLX.dequantized(weight, scales: scales, biases: biases, groupSize: 32, bits: 4)
        let ids = MLXArray([Int32(3), 1])
        let expected = MLX.take(restoredWeight, ids, axis: 0)
        XCTAssertEqual(model.embedding(ids).asArray(Float.self), expected.asArray(Float.self))
        let input = MLXArray.ones([1, 32])
        let output = model.embedding.asLinear(input)
        XCTAssertLessThan(MLX.max(MLX.abs(output - MLX.matmul(input, restoredWeight.T))).item(Float.self), 1e-5)
    }

    func testMissingQuantizedWeightReportsItsMappedKey() {
        let model = Layers()
        XCTAssertThrowsError(try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
            ["layers.0.scales": MLXArray.ones([4, 1])], to: model
        )) { error in
            guard case HFSafetensorsWeightsLoader.LoaderError.missingQuantizedWeight(let key) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(key, "layers.0.weight")
        }
    }
}
