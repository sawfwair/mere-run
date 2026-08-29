import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class HFSafetensorsWeightsLoaderTests: MereRunCoreTestCase {
    func testShardedArraysRespectIndexOwnershipAndExcludeUnindexedTensors() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let index = try makeIndexedShards(root: root)

        let arrays = try HFSafetensorsWeightsLoader.loadShardedArrays(indexURL: index)

        XCTAssertEqual(Set(arrays.keys), ["weight", "other.weight"])
        XCTAssertEqual(try XCTUnwrap(arrays["weight"]).asArray(Float.self), [1, 2, 3, 4])
        XCTAssertEqual(try XCTUnwrap(arrays["other.weight"]).asArray(Float.self), [7])
    }

    func testShardedModuleUpdatesRespectIndexBeforeMapping() throws {
        let root = try TestFileSystem.makeTempDir()
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
        let root = try TestFileSystem.makeTempDir()
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
