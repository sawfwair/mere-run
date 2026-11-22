import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunInkling2Model

final class Inkling2Q8ProjectionTests: MLXTestCase {
    func testPackedCheckpointRoundTrip() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let matrix = MLXArray((0..<256).map { Float($0 % 47) / 47 }, [4, 64]).asType(.bfloat16)
        let (packed, scales, maybeBiases) = quantized(matrix, groupSize: 64, bits: 8, mode: .affine)
        let biases = try XCTUnwrap(maybeBiases)
        let key = Inkling2Q8Projection.outputKey
        let keys = Inkling2Q8Projection.keys(key)
        let tensors = Dictionary(uniqueKeysWithValues: zip(keys, [packed, scales, biases]))
        let filename = "model-00001-of-00001.safetensors"
        try save(arrays: tensors, url: root.appendingPathComponent(filename))
        struct Index: Encodable { let weight_map: [String: String] }
        let index = Index(weight_map: tensors.mapValues { _ in filename })
        try JSONEncoder().encode(index).write(to: root.appendingPathComponent("model.safetensors.index.json"))

        let weights = try Inkling2Weights(root: root)
        let restored = try Inkling2Q8Projection(weights: weights, key: key, rows: 4, columns: 64)
        let input = matrix[0..<2]
        let expected = quantizedMM(input, packed, scales: scales, biases: biases, groupSize: 64, bits: 8, mode: .affine)
        XCTAssertTrue(arrayEqual(restored(input), expected).item(Bool.self))
        XCTAssertThrowsError(try Inkling2Q8Projection(weights: weights, key: key, rows: 5, columns: 64))
    }
}
