import Foundation
import MLX
import MLXNN
import XCTest
import MereRunMLXTestSupport
@testable import MereRunTensor
@testable import MereRunImageModels

final class HFSafetensorsWeightsLoaderTests: MLXTestCase {
    func testAppliedWeightsRetainLoadedValuesAfterCheckpointChanges() throws {
        for dtype in [DType.float32, .bfloat16] {
            let root = try makeTempDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let weights = root.appendingPathComponent("model.safetensors")
            try MLX.save(
                arrays: ["weight": MLXArray(Array(repeating: Float(1), count: 4_096), [64, 64])],
                url: weights
            )
            let model = Linear(64, 64, bias: false)

            try HFSafetensorsWeightsLoader.applyWeights(url: weights, to: model, dtype: dtype)

            let writer = try FileHandle(forWritingTo: weights)
            let end = try writer.seekToEnd()
            try writer.seek(toOffset: end - UInt64(MemoryLayout<UInt32>.size))
            var replacement = Float(9).bitPattern.littleEndian
            try writer.write(contentsOf: withUnsafeBytes(of: &replacement) { Data($0) })
            try writer.close()

            XCTAssertEqual(model.weight.dtype, dtype)
            XCTAssertEqual(
                model.weight.asType(.float32).asArray(Float.self).last, 1,
                "Applied weights must finish checkpoint reads before returning (\(dtype))"
            )
        }
    }

    func testShardedArraysRespectIndexOwnershipAndExcludeUnindexedTensors() throws {
        let root = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let index = try makeIndexedShards(root: root)

        let arrays = try HFSafetensorsWeightsLoader.loadShardedArrays(indexURL: index)

        XCTAssertEqual(Set(arrays.keys), ["weight", "other.weight"])
        XCTAssertEqual(try XCTUnwrap(arrays["weight"]).asArray(Float.self), [1, 2, 3, 4])
        XCTAssertEqual(try XCTUnwrap(arrays["other.weight"]).asArray(Float.self), [7])
    }

    func testShardedModuleUpdatesRespectIndexBeforeMapping() throws {
        let root = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let index = try makeIndexedShards(root: root)
        let model = Linear(2, 2, bias: false)
        var mappedKeys: [String] = []

        try HFSafetensorsWeightsLoader.applyShardedWeights(
            indexURL: index, to: model, dtype: .float32, verify: [.shapeMismatch],
            mapper: { key, value in
                mappedKeys.append(key)
                return key == "weight" ? [(key, value)] : []
            }
        )

        XCTAssertEqual(model.weight.asArray(Float.self), [1, 2, 3, 4])
        XCTAssertEqual(mappedKeys.filter { $0 == "weight" }.count, 1)
        XCTAssertFalse(mappedKeys.contains("unindexed.weight"))
    }

    func testIndexCanSelectTensorFromLaterShard() throws {
        let root = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let index = try makeIndexedShards(root: root, weightOwner: "b.safetensors")

        let arrays = try HFSafetensorsWeightsLoader.loadShardedArrays(indexURL: index)

        XCTAssertEqual(try XCTUnwrap(arrays["weight"]).asArray(Float.self), [9, 9, 9, 9])
        XCTAssertNil(arrays["unindexed.weight"])
    }

    private func makeIndexedShards(root: URL, weightOwner: String = "a.safetensors") throws -> URL {
        try MLX.save(
            arrays: ["weight": MLXArray([Float(1), 2, 3, 4], [2, 2]),
                     "unindexed.weight": MLXArray([Float(5)])],
            url: root.appendingPathComponent("a.safetensors")
        )
        try MLX.save(
            arrays: ["weight": MLXArray([Float(9), 9, 9, 9], [2, 2]),
                     "other.weight": MLXArray([Float(7)])],
            url: root.appendingPathComponent("b.safetensors")
        )
        struct Index: Encodable {
            let weightMap: [String: String]

            enum CodingKeys: String, CodingKey {
                case weightMap = "weight_map"
            }
        }
        let index = Index(weightMap: ["weight": weightOwner, "other.weight": "b.safetensors"])
        let url = root.appendingPathComponent("model.safetensors.index.json")
        try JSONEncoder().encode(index).write(to: url)
        return url
    }
}
