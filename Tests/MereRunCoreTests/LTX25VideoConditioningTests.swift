import MLX
import XCTest
@testable import MereRunCore

final class LTX25VideoConditioningTests: MereRunCoreTestCase {
    func testTemporalRetakeUsesOfficialCausalPixelBounds() {
        let latent = MLX.zeros([1, 2, 4, 1, 2], dtype: .float32)
        let clean = MLX.ones([1, 2, 4, 1, 2], dtype: .float32)
        let positions = MLX.zeros([1, 3, 8, 2], dtype: .float32)
        var state = LTX25VideoTokenState(initialLatent: latent, positions: positions)
        state.applyTemporalRetake(
            cleanVideoLatent: clean,
            startTime: 0.1,
            endTime: 0.5,
            fps: 24,
            regenerate: true
        )
        MLX.eval(state.cleanLatent, state.denoiseMask)

        XCTAssertEqual(state.cleanLatent.asArray(Float.self), [Float](repeating: 1, count: 16))
        XCTAssertEqual(state.denoiseMask.asArray(Float.self), [
            0, 0,
            1, 1,
            1, 1,
            0, 0,
        ])
    }

    func testAudioTemporalRetakeUsesOfficialCausalTimingBounds() {
        let clean = MLX.ones([1, 2, 5, 3], dtype: .float32)
        let state = makeLTXAudioTemporalConditioning(
            cleanLatent: clean,
            startTime: 0.02,
            endTime: 0.1,
            regenerate: true
        )
        MLX.eval(state.denoiseMask)

        // Bounds are [0,.01], [.01,.05], [.05,.09], [.09,.13], [.13,.17].
        XCTAssertEqual(state.denoiseMask.asArray(Float.self), [0, 1, 1, 1, 0])
    }

    func testDubItAudioReferenceIsCleanAppendedAndEndsFortyMillisecondsBeforeZero() {
        let target = MLX.zeros([1, 2, 2, 1], dtype: .float32)
        let reference = MLX.ones([1, 2, 3, 1], dtype: .float32)
        let conditioning = makeLTXAudioReferenceConditioning(
            targetLatent: target,
            referenceLatent: reference,
            frozenTarget: false
        )
        MLX.eval(
            conditioning.state.latent,
            conditioning.state.cleanLatent,
            conditioning.state.denoiseMask,
            conditioning.positions
        )

        XCTAssertEqual(conditioning.state.latent.shape, [1, 2, 5, 1])
        XCTAssertEqual(conditioning.state.denoiseMask.asArray(Float.self), [1, 1, 0, 0, 0])
        XCTAssertEqual(
            conditioning.positions[0, 0, 4, 1].item(Float.self),
            -0.04,
            accuracy: 1e-6
        )
        XCTAssertEqual(conditioning.mainLatent(from: conditioning.state.latent).shape, target.shape)
    }

    func testDubItStageTwoFreezesGeneratedAudioAndItsReference() {
        let stageOne = MLX.ones([1, 2, 3, 1], dtype: .float32)
        let conditioning = makeLTXAudioReferenceConditioning(
            targetLatent: stageOne,
            referenceLatent: stageOne,
            frozenTarget: true
        )
        MLX.eval(conditioning.state.denoiseMask, conditioning.state.cleanLatent)

        XCTAssertEqual(conditioning.state.denoiseMask.asArray(Float.self), [0, 0, 0, 0, 0, 0])
        XCTAssertEqual(
            conditioning.state.cleanLatent.asArray(Float.self),
            [Float](repeating: 1, count: 12)
        )
    }

    func testInitialStateMarksOnlyTheCausalFirstLatentFrame() {
        let latent = MLX.zeros([1, 2, 3, 2, 2], dtype: .float32)
        let positions = MLX.zeros([1, 3, 12, 2], dtype: .float32)
        let state = LTX25VideoTokenState(initialLatent: latent, positions: positions)
        MLX.eval(state.keyframesMask)

        XCTAssertEqual(state.latent.shape, [1, 12, 2])
        XCTAssertEqual(state.keyframesMask.asArray(Float.self), [
            1, 1, 1, 1,
            0, 0, 0, 0,
            0, 0, 0, 0,
        ])
    }

    func testFrameZeroReplacesCleanTokensAndLaterImagesAppendGuidingTokens() {
        let latent = MLX.zeros([1, 2, 2, 1, 2], dtype: .float32)
        let positions = MLX.zeros([1, 3, 4, 2], dtype: .float32)
        var state = LTX25VideoTokenState(initialLatent: latent, positions: positions)
        state.applyImageLatent(
            MLX.ones([1, 2, 1, 1, 2], dtype: .float32),
            pixelFrameIndex: 0,
            strength: 1,
            fps: 24
        )
        state.applyImageLatent(
            MLX.full([1, 2, 1, 1, 2], values: MLXArray(2)),
            pixelFrameIndex: 17,
            strength: 0.75,
            fps: 24
        )
        MLX.eval(state.cleanLatent, state.denoiseMask, state.positions, state.keyframesMask)

        XCTAssertEqual(state.latent.shape, [1, 6, 2])
        XCTAssertEqual(state.cleanLatent[0, 0, 0].item(Float.self), 1)
        XCTAssertEqual(state.cleanLatent[0, 4, 0].item(Float.self), 2)
        XCTAssertEqual(state.denoiseMask[0, 0, 0].item(Float.self), 0)
        XCTAssertEqual(state.denoiseMask[0, 4, 0].item(Float.self), 0.25)
        XCTAssertEqual(state.positions[0, 0, 4, 0].item(Float.self), 17.0 / 24.0, accuracy: 1e-6)
        XCTAssertEqual(state.positions[0, 0, 4, 1].item(Float.self), 18.0 / 24.0, accuracy: 1e-6)
        XCTAssertEqual(state.keyframesMask[0, 4, 0].item(Float.self), 0)
    }

    func testKeyframeInterpolationAppendsFrameZeroInsteadOfReplacingIt() {
        let latent = MLX.zeros([1, 2, 2, 1, 2], dtype: .float32)
        let positions = MLX.zeros([1, 3, 4, 2], dtype: .float32)
        var state = LTX25VideoTokenState(initialLatent: latent, positions: positions)
        state.applyImageLatent(
            MLX.ones([1, 2, 1, 1, 2], dtype: .float32),
            pixelFrameIndex: 0,
            strength: 1,
            fps: 24,
            replaceFirstFrame: false
        )
        MLX.eval(state.latent, state.cleanLatent, state.denoiseMask, state.positions)

        XCTAssertEqual(state.latent.shape, [1, 6, 2])
        XCTAssertEqual(state.cleanLatent[0, 0, 0].item(Float.self), 0)
        XCTAssertEqual(state.cleanLatent[0, 4, 0].item(Float.self), 1)
        XCTAssertEqual(state.denoiseMask[0, 0, 0].item(Float.self), 1)
        XCTAssertEqual(state.denoiseMask[0, 4, 0].item(Float.self), 0)
        XCTAssertEqual(state.positions[0, 0, 4, 0].item(Float.self), 0, accuracy: 1e-6)
        XCTAssertEqual(state.positions[0, 0, 4, 1].item(Float.self), 1.0 / 24.0, accuracy: 1e-6)
    }

    func testGeneratedSlotsUseExactPixelSpansAndLearnedMarkerMask() {
        let latent = MLX.zeros([1, 2, 2, 1, 2], dtype: .float32)
        let positions = MLX.zeros([1, 3, 4, 2], dtype: .float32)
        var state = LTX25VideoTokenState(initialLatent: latent, positions: positions)
        state.appendGeneratedKeyframeSlots(pixelFrameIndices: [8, 23], fps: 24)
        MLX.eval(state.positions, state.keyframesMask)

        XCTAssertEqual(state.latent.shape, [1, 8, 2])
        XCTAssertEqual(
            state.generatedKeyframeLayout,
            LTXGeneratedKeyframeLayout(
                pixelFrameIndices: [8, 23],
                tokensPerKeyframe: 2,
                firstToken: 4
            )
        )
        XCTAssertEqual(state.positions[0, 0, 4, 0].item(Float.self), 8.0 / 24.0, accuracy: 1e-6)
        XCTAssertEqual(state.positions[0, 0, 4, 1].item(Float.self), 9.0 / 24.0, accuracy: 1e-6)
        XCTAssertEqual(state.positions[0, 0, 6, 0].item(Float.self), 23.0 / 24.0, accuracy: 1e-6)
        XCTAssertEqual(state.keyframesMask[0, 4, 0].item(Float.self), 1)
        XCTAssertEqual(state.keyframesMask[0, 7, 0].item(Float.self), 1)
        XCTAssertEqual(state.generatedKeyframes()?.shape, [1, 2, 2, 1, 2])
        XCTAssertEqual(state.mainLatent().shape, latent.shape)
    }

    func testReferenceLatentUsesICLoRATokenAndCoordinateContract() throws {
        let latent = MLX.zeros([1, 2, 2, 1, 2], dtype: .float32)
        let positions = MLX.zeros([1, 3, 4, 2], dtype: .float32)
        var state = LTX25VideoTokenState(initialLatent: latent, positions: positions)
        state.appendReferenceLatent(
            MLX.full([1, 2, 2, 1, 1], values: MLXArray(3)),
            downscaleFactor: 2,
            temporalScaleFactor: 4,
            strength: 1,
            attentionStrength: 0.5,
            fps: 24
        )
        let attentionMask = try XCTUnwrap(state.attentionMask)
        MLX.eval(
            state.latent,
            state.cleanLatent,
            state.denoiseMask,
            state.positions,
            state.keyframesMask,
            attentionMask
        )

        XCTAssertEqual(state.latent.shape, [1, 6, 2])
        XCTAssertEqual(state.latent[0, 4, 0].item(Float.self), 0)
        XCTAssertEqual(state.cleanLatent[0, 4, 0].item(Float.self), 3)
        XCTAssertEqual(state.denoiseMask[0, 4, 0].item(Float.self), 0)
        XCTAssertEqual(state.keyframesMask[0, 4, 0].item(Float.self), 0)
        XCTAssertEqual(state.positions[0, 0, 4, 0].item(Float.self), 0, accuracy: 1e-6)
        XCTAssertEqual(state.positions[0, 0, 4, 1].item(Float.self), 1.0 / 24.0, accuracy: 1e-6)
        XCTAssertEqual(state.positions[0, 0, 5, 0].item(Float.self), 1.0 / 24.0, accuracy: 1e-6)
        XCTAssertEqual(state.positions[0, 0, 5, 1].item(Float.self), 33.0 / 24.0, accuracy: 1e-6)
        XCTAssertEqual(state.positions[0, 1, 4, 1].item(Float.self), 64)
        XCTAssertEqual(attentionMask.shape, [1, 6, 6])
        XCTAssertEqual(attentionMask[0, 0, 4].item(Float.self), 0.5)
        XCTAssertEqual(attentionMask[0, 4, 0].item(Float.self), 0.5)
        XCTAssertEqual(attentionMask[0, 4, 5].item(Float.self), 1)
    }

    func testSeparateReferenceGroupsDoNotAttendToEachOther() throws {
        let latent = MLX.zeros([1, 2, 1, 1, 2], dtype: .float32)
        let positions = MLX.zeros([1, 3, 2, 2], dtype: .float32)
        var state = LTX25VideoTokenState(initialLatent: latent, positions: positions)
        let reference = MLX.ones([1, 2, 1, 1, 1], dtype: .float32)
        state.appendReferenceLatent(reference, attentionStrength: 0.75, fps: 24)
        state.appendReferenceLatent(reference, attentionStrength: 0.25, fps: 24)
        let attentionMask = try XCTUnwrap(state.attentionMask)
        MLX.eval(attentionMask)

        XCTAssertEqual(attentionMask.shape, [1, 4, 4])
        XCTAssertEqual(attentionMask[0, 2, 3].item(Float.self), 0)
        XCTAssertEqual(attentionMask[0, 3, 2].item(Float.self), 0)
        XCTAssertEqual(attentionMask[0, 0, 3].item(Float.self), 0.25)
        XCTAssertEqual(attentionMask[0, 3, 0].item(Float.self), 0.25)
    }

    func testReferenceSpatialAttentionMaskUsesCausalAreaDownsampling() throws {
        let mask = MLXArray([
            Float(1), 1, 0, 0,
            1, 1, 0, 0,
            0, 0, 1, 1,
            0, 0, 1, 1,
            0, 0, 0, 0,
            0, 0, 0, 0,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
        ]).reshaped(1, 1, 3, 4, 4)
        let targetShape = LTXVideoLatentShape(
            batch: 1,
            channels: 2,
            frames: 2,
            height: 2,
            width: 2
        )
        let weights = downsampleLTXReferenceAttentionMask(mask, targetLatentShape: targetShape)
        MLX.eval(weights)
        XCTAssertEqual(weights.asArray(Float.self), [1, 0, 0, 1, 0.5, 0.5, 1, 1])

        let latent = MLX.zeros([1, 2, 1, 1, 2], dtype: .float32)
        let positions = MLX.zeros([1, 3, 2, 2], dtype: .float32)
        var state = LTX25VideoTokenState(initialLatent: latent, positions: positions)
        state.appendReferenceLatent(
            MLX.ones([1, 2, 2, 2, 2], dtype: .float32),
            attentionStrength: 0.5,
            attentionWeights: weights,
            fps: 24
        )
        let attention = try XCTUnwrap(state.attentionMask)
        MLX.eval(attention)
        XCTAssertEqual(attention[0, 0, 2].item(Float.self), 0.5)
        XCTAssertEqual(attention[0, 0, 3].item(Float.self), 0)
        XCTAssertEqual(attention[0, 0, 6].item(Float.self), 0.25)
    }

    func testEvenlySpacedGeneratedKeyframesMatchUpstreamRounding() throws {
        XCTAssertEqual(
            try ltxEvenlySpacedGeneratedKeyframePositions(count: 3, numFrames: 97),
            [24, 48, 72]
        )
        XCTAssertEqual(
            try ltxEvenlySpacedGeneratedKeyframePositions(count: 2, numFrames: 10),
            [3, 6]
        )
        XCTAssertEqual(
            try ltxEvenlySpacedGeneratedKeyframePositions(count: 1, numFrames: 4),
            [2]
        )
        XCTAssertEqual(
            try ltxEvenlySpacedGeneratedKeyframePositions(count: 0, numFrames: 1),
            []
        )
        XCTAssertThrowsError(
            try ltxEvenlySpacedGeneratedKeyframePositions(count: 3, numFrames: 4)
        )
    }
}
