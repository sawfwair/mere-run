import Foundation
import MLX
import MLXNN
import XCTest
import MereRunMLXTestSupport
@testable import MereRunCore

final class H3ImageCheckpointTests: MLXTestCase {
    private let fixtureRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/H3Image")

    func testOriginalImageVAEHeaderCoversEveryNativeParameterWithCorrectShape() throws {
        struct Tensor: Decodable { let shape: [Int] }
        let header = try JSONDecoder().decode([String: Tensor].self, from: Data(contentsOf: fixtureRoot.appending(path: "vae-header.json")))
        let configuration = try JSONDecoder().decode(MiniMaxH3ImageVAEConfiguration.self,
            from: Data(contentsOf: fixtureRoot.appending(path: "vae.json")))
        let model = MiniMaxH3ImageVAE(configuration: configuration)
        // Shape-only lazy zeros exercise mapping and strict coverage without allocating checkpoint storage.
        let mapped = header.flatMap { MiniMaxH3ImageLoader.vaeWeight(key: $0.key, value: MLXArray.zeros($0.value.shape)) }
        try model.update(parameters: ModuleParameters.unflattened(mapped), verify: .all)
        XCTAssertEqual(Set(model.parameters().flattened().map(\.0)), Set(mapped.map(\.0)))
        XCTAssertFalse(model.parameters().flattened().contains { $0.0 == "mean" || $0.0 == "standardDeviation" })
    }

    func testOfficialConditionerIndexMatchesTruncatedLayer50Encoder() throws {
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self,
            from: Data(contentsOf: fixtureRoot.appending(path: "text-index.json")))
        let encoder = MiniMaxH3ModelLoader.makeConditioner()
        let required = Set(encoder.parameters().flattened().map(\.0)).subtracting(["textEncoder.encoder.norm.weight"])
        let dummy = MLXArray.zeros([1])
        let mapped = index.weightMap.keys.flatMap { key in
            MiniMaxH3ModelLoader.conditionerWeight(key: MiniMaxH3ModelLoader.conditionerWeightKey(key), value: dummy).map(\.0)
        }
        XCTAssertEqual(Set(mapped), required)
        XCTAssertEqual(mapped.count, required.count)
    }

    func testVAEQKVMappingUsesSixteenHeadsAndBothWeightAndBias() throws {
        let input = MLXArray(0..<(16 * 3 * 64)).asType(.float32)
        let mapped = try XCTUnwrap(MiniMaxH3ImageLoader.vaeWeight(key: "decoder.transformer_blocks.0.attn.to_qkv.bias", value: input).first)
        let values = mapped.1.asArray(Float.self)
        XCTAssertEqual(Array(values.prefix(64)), Array(0..<64).map(Float.init))
        XCTAssertEqual(values[64], 192)
        XCTAssertEqual(values[1_024], 64)
        XCTAssertEqual(values[2_048], 128)
    }
}
