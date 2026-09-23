import Foundation
import XCTest
import MLX
import MLXNN
@testable import MereRunCore

final class FalconPerceptionNormalizationTests: MereRunCoreTestCase {
    private struct Fixture: Decodable {
        let input: [Float]
        let wqkv: [Float]
        let w13: [Float]
        let w2: [Float]
        let normalized: [Float]
        let queries: [Float]
        let keys: [Float]
        let mlp: [Float]
        let finalNorm: [Float]
    }

    private func fixture() throws -> Fixture {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/falcon-functional-normalization.json")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    private func config() -> FalconPerceptionTextConfig {
        FalconPerceptionTextConfig(
            hiddenSize: 4, numHiddenLayers: 1, numAttentionHeads: 2,
            headDim: 4, numKeyValueHeads: 1, vocabSize: 16,
            intermediateSize: 3, rmsNormEps: 0.1, maxPositionEmbeddings: 32
        )
    }

    private func assertValues(
        _ actual: MLXArray, _ expected: [Float],
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let values = actual.asType(.float32).asArray(Float.self)
        XCTAssertEqual(values.count, expected.count, file: file, line: line)
        for (value, reference) in zip(values, expected) {
            XCTAssertEqual(value, reference, accuracy: 2e-6, file: file, line: line)
        }
    }

    func testAttentionFunctionalNormsMatchTorchInsteadOfFinalLayerEpsilon() throws {
        let data = try fixture()
        let attention = FalconPerceptionAttention(config: config())
        try attention.update(
            parameters: ModuleParameters.unflattened([
                ("wqkv.weight", MLXArray(data.wqkv, [16, 4]))
            ]), verify: [.shapeMismatch]
        )
        attention.captureDebugStages = true
        let output = attention(
            MLXArray(data.input, [1, 2, 4]), mask: nil, cache: nil,
            cos1D: nil, sin1D: nil, cos2D: nil, sin2D: nil
        )
        MLX.eval(output)
        let captured = try XCTUnwrap(attention.lastDebugCapture)
        assertValues(try XCTUnwrap(captured.normalizedInput), data.normalized)
        assertValues(try XCTUnwrap(captured.queriesBeforeRoPE), data.queries)
        assertValues(try XCTUnwrap(captured.keysBeforeRoPE), data.keys)
    }

    func testFeedForwardFunctionalNormMatchesTorch() throws {
        let data = try fixture()
        let mlp = FalconPerceptionMLP(config: config())
        try mlp.update(
            parameters: ModuleParameters.unflattened([
                ("w13.weight", MLXArray(data.w13, [6, 4])),
                ("w2.weight", MLXArray(data.w2, [4, 3]))
            ]), verify: [.shapeMismatch]
        )
        assertValues(mlp(MLXArray(data.input, [1, 2, 4])), data.mlp)
    }

    func testFinalNormalizationRetainsConfiguredEpsilon() throws {
        let data = try fixture()
        let model = FalconPerceptionTransformerModel(
            config: FalconPerceptionModelConfig(textConfig: config())
        )
        assertValues(model.norm(MLXArray(data.input, [1, 2, 4])), data.finalNorm)
    }
}
