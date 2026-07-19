import MLX
import XCTest
@testable import MereRunCore

final class SCAIL2TransformerTests: MereRunCoreTestCase {
    private let configuration = SCAIL2TransformerConfiguration(
        textLength: 8,
        hiddenSize: 32,
        feedForwardSize: 64,
        timestepFrequencySize: 16,
        textEmbeddingSize: 16,
        imageEmbeddingSize: 8,
        headCount: 4,
        layerCount: 2,
        ropeTableLength: 256,
        poseWidthShift: 16,
        replacementReferenceHeightShift: 16
    )

    func testMixedRoPELayoutMatchesAllTokenStreams() {
        let layout = SCAIL2TokenLayout(
            additionalReferenceGrid: Wan2GridSize(frames: 2, height: 4, width: 4),
            referenceGrid: Wan2GridSize(frames: 1, height: 4, width: 4),
            videoGrid: Wan2GridSize(frames: 3, height: 4, width: 4),
            drivingGrid: Wan2GridSize(frames: 3, height: 2, width: 2)
        )
        let frequencies = Wan2RoPE.frequencies(maxSequence: 256, dimensions: [4, 2, 2])
        let animation = SCAIL2RoPE.prepare(
            layout: layout,
            frequencies: frequencies,
            mode: .animation,
            poseWidthShift: 16,
            replacementReferenceHeightShift: 16
        )
        let replacement = SCAIL2RoPE.prepare(
            layout: layout,
            frequencies: frequencies,
            mode: .replacement,
            poseWidthShift: 16,
            replacementReferenceHeightShift: 16
        )
        eval(animation.cosine, replacement.cosine)

        XCTAssertEqual(animation.cosine.dim(0), layout.totalLength)
        XCTAssertEqual(animation.cosine.shape, replacement.cosine.shape)
        XCTAssertNotEqual(
            animation.cosine.asArray(Float.self),
            replacement.cosine.asArray(Float.self)
        )
    }

    func testTinyTransformerProducesVideoLatentGeometry() {
        let model = SCAIL2TransformerModel(configuration: configuration)
        let input = SCAIL2TransformerInput(
            videoLatent: MLX.zeros([16, 3, 8, 8]),
            referenceLatent: MLX.zeros([16, 1, 8, 8]),
            referenceMask: MLX.zeros([28, 1, 8, 8]),
            drivingLatent: MLX.zeros([16, 3, 4, 4]),
            drivingMask: MLX.zeros([28, 3, 4, 4]),
            textEmbeddings: MLX.zeros([1, 4, 16]),
            imageEmbeddings: MLX.zeros([1, 17, 8]),
            timestep: MLXArray([500]),
            mode: .animation
        )
        let output = model(input)
        eval(output)

        XCTAssertEqual(output.shape, [16, 3, 8, 8])
        XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testCleanHistoryAndAdditionalReferenceStreamsAreAccepted() {
        let model = SCAIL2TransformerModel(configuration: configuration)
        let output = model(SCAIL2TransformerInput(
            videoLatent: MLX.zeros([16, 3, 8, 8]),
            referenceLatent: MLX.zeros([16, 1, 8, 8]),
            referenceMask: MLX.zeros([28, 1, 8, 8]),
            drivingLatent: MLX.zeros([16, 3, 4, 4]),
            drivingMask: MLX.zeros([28, 3, 4, 4]),
            historyMask: MLX.ones([4, 3, 8, 8]),
            additionalReferenceLatents: [MLX.zeros([16, 1, 8, 8])],
            additionalReferenceMasks: [MLX.zeros([28, 1, 8, 8])],
            textEmbeddings: MLX.zeros([1, 8, 16]),
            imageEmbeddings: MLX.zeros([1, 17, 8]),
            timestep: MLXArray([250]),
            mode: .replacement
        ))
        eval(output)
        XCTAssertEqual(output.shape, [16, 3, 8, 8])
    }

    func testPreparedConditioningMatchesDirectForwardPath() {
        let model = SCAIL2TransformerModel(configuration: configuration)
        let input = SCAIL2TransformerInput(
            videoLatent: MLX.zeros([16, 3, 8, 8]),
            referenceLatent: MLX.zeros([16, 1, 8, 8]),
            referenceMask: MLX.zeros([28, 1, 8, 8]),
            drivingLatent: MLX.zeros([16, 3, 4, 4]),
            drivingMask: MLX.zeros([28, 3, 4, 4]),
            textEmbeddings: MLX.zeros([1, 4, 16]),
            imageEmbeddings: MLX.zeros([1, 17, 8]),
            timestep: MLXArray([500]),
            mode: .animation
        )
        let direct = model(input)
        let prepared = model.prepareConditioning(
            textEmbeddings: input.textEmbeddings,
            imageEmbeddings: input.imageEmbeddings
        )
        let cached = model(input, conditioning: prepared)
        eval(direct, cached)

        XCTAssertEqual(direct.asArray(Float.self), cached.asArray(Float.self))
    }
}
