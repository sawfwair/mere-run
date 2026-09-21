import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class QwenImage21ConditioningTests: MereRunCoreTestCase {
    func testFinalUnnormalizedMultimodalActivationMatchesTransformers() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "ImageRuntimeTests/Fixtures/QwenImage21")
        let model = QwenVLEncoder(
            textEncoderConfig: QwenTextEncoderConfiguration(
                vocabSize: 128, hiddenSize: 16, numHiddenLayers: 3, numAttentionHeads: 2,
                numKeyValueHeads: 1, intermediateSize: 32, ropeTheta: 5_000_000, rmsNormEps: 1e-6,
                promptDropIndex: 0, headDim: 8, mropeSection: [2, 1, 1], mropeInterleaved: true,
                useFloat32Activations: true
            ),
            visionConfig: QwenVisionConfiguration(
                depth: 2, embedDim: 16, mlpHiddenDim: 32, hiddenAct: .geluApproximate,
                numHeads: 2, patchSize: 16, temporalPatchSize: 2, spatialMergeSize: 2,
                inChannels: 3, outHiddenDim: 16, patchEmbedBias: true,
                numPositionEmbeddings: 16, useLearnedPosEmbed: true, deepstackVisualIndexes: [0, 1]
            )
        )
        let arrays = try MLX.loadArrays(url: root.appending(path: "text-encoder.safetensors"))
        let mapped = arrays.filter { $0.key != "lm_head.weight" }.flatMap { Qwen3VLEmbeddingWeights.mapWeight($0.key, $0.value) }
        try model.update(parameters: ModuleParameters.unflattened(mapped), verify: [.all])
        let data = try MLX.loadArrays(url: root.appending(path: "text-results.safetensors"))
        let visual = try model.visionTower(patchInputs: data["patches"]!.reshaped(1, 4, -1), grid: [QwenVisionGrid(temporal: 1, height: 2, width: 2)])
        XCTAssertLessThan(max(abs(visual.hiddenStates - data["visual"]!)).item(Float.self), 2e-5)
        for index in 0..<2 {
            XCTAssertLessThan(max(abs(visual.deepstackFeatures[index] - data["deep\(index)"]!)).item(Float.self), 2e-5)
        }
        let actual = try XCTUnwrap(model.forwardMultimodalActivationHiddenState(
            inputIds: data["ids"]!, attentionMask: ones([1, 6], dtype: .int32),
            images: [.init(pixelValues: data["pixels"]!, tokenRange: 2..<3, heightPatchCount: 2, widthPatchCount: 2)],
            activationLayer: 2
        ))
        XCTAssertLessThan(max(abs(actual - data["expected"]!)).item(Float.self), 2e-5)
    }
}
