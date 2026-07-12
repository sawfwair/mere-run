import MLX
import XCTest
@testable import MereRunCore

final class Wan2TransformerTests: MereRunCoreTestCase {
    func testPatchLayoutMatchesSceneWorksSourceOrder() {
        let channel0 = MLXArray([Float(0), 1, 2, 3], [1, 1, 2, 2])
        let channel1 = MLXArray([Float(10), 11, 12, 13], [1, 1, 2, 2])
        let latent = MLX.concatenated([channel0, channel1], axis: 0)
        let patch = Wan2PatchLayout.flatten(latent, patchSize: [1, 2, 2])
        eval(patch.value)

        XCTAssertEqual(patch.grid, Wan2GridSize(frames: 1, height: 1, width: 1))
        XCTAssertEqual(patch.value.asArray(Float.self), [0, 1, 2, 3, 10, 11, 12, 13])
    }

    func testTinyTransformerPreservesVideoShapeForTokenTimesteps() {
        let configuration = Wan2TransformerConfiguration(
            patchSize: [1, 2, 2],
            textLength: 4,
            inputChannels: 4,
            hiddenSize: 16,
            feedForwardSize: 32,
            timestepFrequencySize: 8,
            textEmbeddingSize: 6,
            outputChannels: 4,
            headCount: 2,
            layerCount: 2
        )
        let model = Wan2TransformerModel(configuration: configuration)
        let latent = MLX.zeros([4, 2, 4, 4])
        let text = MLX.zeros([1, 4, 6])
        let context = model.embedText(text)
        let timesteps = MLX.ones([1, 8]) * 500
        let output = model(latents: [latent], timesteps: timesteps, embeddedContext: context)
        eval(output)

        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(output[0].shape, [4, 2, 4, 4])
        XCTAssertEqual(output[0].dtype, .float32)
        XCTAssertTrue(output[0].asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testCrossAttentionCachesMatchLayerCountAndBatch() {
        let configuration = Wan2TransformerConfiguration(
            textLength: 4,
            inputChannels: 4,
            hiddenSize: 16,
            feedForwardSize: 32,
            timestepFrequencySize: 8,
            textEmbeddingSize: 6,
            outputChannels: 4,
            headCount: 2,
            layerCount: 3
        )
        let model = Wan2TransformerModel(configuration: configuration)
        let context = model.embedText(MLX.zeros([2, 4, 6]))
        let caches = model.prepareCrossAttentionCaches(context: context)
        eval(caches.flatMap { [$0.key, $0.value] })

        XCTAssertEqual(caches.count, 3)
        XCTAssertEqual(caches[0].key.shape, [2, 2, 4, 8])
        XCTAssertEqual(caches[0].value.shape, [2, 2, 4, 8])
    }

    func testTextEmbeddingPadsRawContextBeforeProjection() {
        let configuration = Wan2TransformerConfiguration(
            textLength: 4,
            inputChannels: 4,
            hiddenSize: 16,
            feedForwardSize: 32,
            timestepFrequencySize: 8,
            textEmbeddingSize: 6,
            outputChannels: 4,
            headCount: 2,
            layerCount: 1
        )
        let model = Wan2TransformerModel(configuration: configuration)
        let context = model.embedText(MLX.zeros([1, 2, 6]))
        eval(context)
        XCTAssertEqual(context.shape, [1, 4, 16])
    }
}
