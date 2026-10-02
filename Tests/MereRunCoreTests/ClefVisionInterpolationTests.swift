import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunTextEncoder

final class ClefVisionInterpolationTests: MereRunCoreTestCase {
    func testCheckpointDTypePositionInterpolationMatchesIndependentReference() throws {
        struct Fixture: Decodable { let position_grids: [[Int]] }
        let root = Bundle.module.resourceURL!.appending(path: "Fixtures/Clef")
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: root.appending(path: "media-reference.json")))
        let arrays = try MLX.loadArrays(url: root.appending(path: "vision-interpolation.safetensors"))
        let embedding = Embedding(embeddingCount: 2304, dimensions: 8)
        try embedding.update(parameters: ModuleParameters.unflattened([("weight", try XCTUnwrap(arrays["weight"]))]), verify: [.all])
        let tower = QwenVisionTower(configuration: QwenVisionConfiguration(
            depth: 0, embedDim: 8, mlpHiddenDim: 16, numHeads: 2, outHiddenDim: 8,
            numPositionEmbeddings: 2304, useLearnedPosEmbed: true))
        for (index, grid) in fixture.position_grids.enumerated() {
            let geometry = [(t: grid[0], h: grid[1], w: grid[2])]
            let expected = try XCTUnwrap(arrays["bf16_\(index)"])
            tower.positionEmbeddingArithmetic = .embeddingDType
            let actual = tower.fastPosEmbedInterpolate(gridThw: geometry, posEmbed: embedding)
            XCTAssertEqual(actual.dtype, .bfloat16)
            XCTAssertEqual(actual.shape, expected.shape)
            XCTAssertEqual(abs(actual - expected).max().item(Float.self), 0, "grid \(grid)")
            // Preserve the existing FP32 policy for other runtime callers.
            tower.positionEmbeddingArithmetic = .float32
            let ordinary = tower.fastPosEmbedInterpolate(gridThw: geometry, posEmbed: embedding)
            let expectedFP32 = try XCTUnwrap(arrays["fp32_\(index)"])
            XCTAssertEqual(ordinary.dtype, .float32)
            // CPU coordinates and reference MLX linspace differ by a few FP32 ULPs.
            XCTAssertLessThanOrEqual(abs(ordinary - expectedFP32).max().item(Float.self), 5e-6)
        }
    }
}
