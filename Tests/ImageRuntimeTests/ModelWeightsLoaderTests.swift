import Foundation
import MLX
import MLXNN
import MereRunMLXTestSupport
import XCTest
import MereRunTensor

final class ModelWeightsLoaderTests: MLXTestCase {
    func testIndexTakesPrecedenceOverSingleFileAndAppliesDtypeAndMapping() throws {
        let root = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let single = root.appendingPathComponent("model.safetensors")
        let shard = root.appendingPathComponent("shard.safetensors")
        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        try MLX.save(arrays: ["checkpoint.weight": MLXArray.ones([2, 2])], url: single)
        let indexedWeight = MLXArray([Float(2), 3, 4, 5], [2, 2])
        try MLX.save(arrays: ["checkpoint.weight": indexedWeight], url: shard)
        let index = HFSafetensorsIndex(
            metadata: nil, weightMap: ["checkpoint.weight": shard.lastPathComponent]
        )
        try JSONEncoder().encode(index).write(to: indexURL)
        let indexed = Linear(2, 2, bias: false)
        let fromSingle = Linear(2, 2, bias: false)
        let mapper: (String, MLXArray) -> [(String, MLXArray)] = { _, value in [("weight", value)] }
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: indexURL, singleURL: single, to: indexed,
            dtype: .float16, verify: [.noUnusedKeys, .shapeMismatch], mapper: mapper
        )
        try FileManager.default.removeItem(at: indexURL)
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: indexURL, singleURL: single, to: fromSingle,
            dtype: .float16, verify: [.noUnusedKeys, .shapeMismatch], mapper: mapper
        )
        XCTAssertEqual(indexed.weight.dtype, .float16)
        XCTAssertEqual(fromSingle.weight.dtype, .float16)
        XCTAssertEqual(indexed.weight.asArray(Float.self), [2, 3, 4, 5])
        XCTAssertEqual(fromSingle.weight.asArray(Float.self), [1, 1, 1, 1])
    }

    func testQuantizedCheckpointRequiresExplicitMetadataForSingleAndIndexedFiles() throws {
        let root = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let single = root.appendingPathComponent("model.safetensors")
        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        let model = Linear(32, 4, bias: false)
        try MLX.save(arrays: ["layer.scales": MLXArray.ones([4, 1])], url: single)
        func loadWithoutMetadata() throws {
            try ModelWeightsLoader.applyHFSafetensors(indexURL: indexURL, singleURL: single, to: model)
        }
        for indexed in [false, true] {
            if indexed {
                let index = HFSafetensorsIndex(
                    metadata: nil, weightMap: ["layer.scales": single.lastPathComponent]
                )
                try JSONEncoder().encode(index).write(to: indexURL)
            }
            XCTAssertThrowsError(try loadWithoutMetadata()) { error in
                guard case ModelWeightsLoader.LoaderError.invalidQuantizationMetadata = error else {
                    return XCTFail("Unexpected error: \(error)")
                }
            }
        }
    }

    func testMissingIndexAndShardKeepTypedErrors() throws {
        let root = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        XCTAssertThrowsError(try HFSafetensorsWeightsLoader.loadShardedArrays(indexURL: indexURL)) { error in
            guard case HFSafetensorsWeightsLoader.LoaderError.indexFileMissing(let url) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(url, indexURL)
        }
        let index = HFSafetensorsIndex(metadata: nil, weightMap: ["weight": "missing.safetensors"])
        try JSONEncoder().encode(index).write(to: indexURL)
        XCTAssertThrowsError(try HFSafetensorsWeightsLoader.loadShardedArrays(indexURL: indexURL)) { error in
            guard case HFSafetensorsWeightsLoader.LoaderError.shardFileMissing(let url) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(url.lastPathComponent, "missing.safetensors")
        }
    }
}
