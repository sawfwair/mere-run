import MLX
import XCTest
@testable import MereRunCore

final class Wan2CameraConditioningTests: MereRunCoreTestCase {
    func testDreamXLatentFrameAlignmentMatchesOnePlusFourKContract() {
        XCTAssertEqual(Wan2DreamXCameraTrajectory.latentFrameIndices(pixelFrameCount: 1), [0])
        XCTAssertEqual(Wan2DreamXCameraTrajectory.latentFrameIndices(pixelFrameCount: 17), [0, 4, 8, 12, 16])
        XCTAssertEqual(Wan2DreamXCameraTrajectory.latentFrameIndices(pixelFrameCount: 81).count, 21)
        XCTAssertEqual(Wan2DreamXCameraTrajectory.latentFrameIndices(pixelFrameCount: 81).last, 80)
    }

    func testYawLeftProducesProjectiveViewRotationWithoutTranslation() {
        let conditioning = Wan2DreamXCameraTrajectory.compile(
            segments: [Wan2DreamXTrajectorySegment(action: "j", speed: 1.5)],
            pixelFrameCount: 17
        )
        XCTAssertEqual(conditioning.frameCount, 5)
        XCTAssertEqual(conditioning.viewMatrices.count, 80)
        let first = Array(conditioning.viewMatrices[0..<16])
        XCTAssertEqual(first, [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ])
        let last = Array(conditioning.viewMatrices[64..<80])
        XCTAssertEqual(last[3], 0, accuracy: 1e-6)
        XCTAssertEqual(last[7], 0, accuracy: 1e-6)
        XCTAssertEqual(last[11], 0, accuracy: 1e-6)
        XCTAssertGreaterThan(last[2], 0)
        XCTAssertLessThan(last[8], 0)
    }

    func testForwardTrajectoryUsesRelativeFirstFrameAndPositiveCameraDepth() {
        let conditioning = Wan2DreamXCameraTrajectory.compile(
            segments: [Wan2DreamXTrajectorySegment(action: "w", speed: 1.5)],
            pixelFrameCount: 17
        )
        let first = Array(conditioning.viewMatrices[0..<16])
        let last = Array(conditioning.viewMatrices[64..<80])
        XCTAssertEqual(first[11], 0, accuracy: 1e-6)
        XCTAssertLessThan(last[11], 0)
        XCTAssertEqual(last[11], -1.5 * 16 / 17, accuracy: 1e-5)
    }

    func testProjectiveTransformsRoundTripIdentityCameraFeatures() {
        let conditioning = Wan2DreamXCameraTrajectory.compile(
            segments: [Wan2DreamXTrajectorySegment(action: " ")],
            pixelFrameCount: 5
        )
        let transforms = Wan2ProjectivePositionEncoding.prepare(
            conditioning: conditioning,
            batchSize: 1,
            dtype: .float32
        )
        let features = MLXArray((0..<32).map(Float.init), [1, 1, 4, 8])
        let encoded = Wan2ProjectivePositionEncoding.apply(
            features,
            matrices: transforms.keyValue,
            cameraFrames: conditioning.frameCount
        )
        let decoded = Wan2ProjectivePositionEncoding.apply(
            encoded,
            matrices: transforms.output,
            cameraFrames: conditioning.frameCount
        )
        eval(decoded)
        for (actual, expected) in zip(decoded.asArray(Float.self), features.asArray(Float.self)) {
            XCTAssertEqual(actual, expected, accuracy: 1e-5)
        }
    }

    func testTinyTransformerAcceptsDreamXProjectiveConditioning() {
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
            layerCount: 2,
            projectiveCameraConditioning: true
        )
        let model = Wan2TransformerModel(configuration: configuration)
        let context = model.embedText(MLX.zeros([1, 4, 6]))
        let conditioning = Wan2DreamXCameraTrajectory.compile(
            segments: [Wan2DreamXTrajectorySegment(action: "j")],
            pixelFrameCount: 5
        )
        let output = model(
            latents: [MLX.zeros([4, 2, 4, 4])],
            timesteps: MLX.ones([1, 8]) * 500,
            embeddedContext: context,
            cameraConditioning: conditioning
        )
        eval(output)
        XCTAssertEqual(output[0].shape, [4, 2, 4, 4])
        XCTAssertTrue(output[0].asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testCameraTransformerUsesReleasedDreamXParameterNames() {
        let configuration = Wan2TransformerConfiguration(
            inputChannels: 4,
            hiddenSize: 16,
            feedForwardSize: 32,
            timestepFrequencySize: 8,
            textEmbeddingSize: 6,
            outputChannels: 4,
            headCount: 2,
            layerCount: 3,
            projectiveCameraConditioning: true
        )
        let names = Wan2TransformerModel(configuration: configuration)
            .parameters()
            .flattened()
            .map(\.0)
            .filter { $0.contains(".cam_self_attn.") }
        XCTAssertEqual(names.count, 30)
        XCTAssertTrue(names.contains("blocks.0.cam_self_attn.q_proj.weight"))
        XCTAssertTrue(names.contains("blocks.2.cam_self_attn.out_proj.bias"))
        XCTAssertTrue(names.contains("blocks.1.cam_self_attn.norm_k.weight"))
    }

    func testCausalCameraTransformerUsesReleasedCompressedShapes() {
        let configuration = Wan2TransformerConfiguration(
            hiddenSize: 3_072,
            feedForwardSize: 14_336,
            headCount: 24,
            layerCount: 1,
            projectiveCameraConditioning: true,
            projectiveCameraAttentionCompression: 4
        )
        let parameters = Dictionary(uniqueKeysWithValues: Wan2TransformerModel(configuration: configuration)
            .parameters()
            .flattened())
        XCTAssertEqual(parameters["blocks.0.cam_self_attn.q_proj.weight"]?.shape, [768, 3_072])
        XCTAssertEqual(parameters["blocks.0.cam_self_attn.out_proj.weight"]?.shape, [3_072, 768])
        XCTAssertEqual(parameters["blocks.0.cam_self_attn.norm_q.weight"]?.shape, [768])
    }

    func testCausalTransformerStateAppendsRecomputesAndRolls() {
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
            layerCount: 2,
            projectiveCameraConditioning: true
        )
        let model = Wan2TransformerModel(configuration: configuration)
        let state = Wan2CausalTransformerState(
            layerCount: 2,
            localAttentionFrames: 4,
            sinkFrames: 1,
            cachesProjectiveAttention: true
        )
        let context = model.embedText(MLX.zeros([1, 4, 6]))
        let conditioning = Wan2DreamXCameraTrajectory.compile(
            segments: [Wan2DreamXTrajectorySegment(action: "j")],
            pixelFrameCount: 5
        )
        let latent = MLX.zeros([4, 2, 4, 4])
        let timesteps = MLX.ones([1, 8]) * 500

        let first = model(
            latents: [latent],
            timesteps: timesteps,
            embeddedContext: context,
            cameraConditioning: conditioning,
            causalState: state,
            currentStartToken: 0
        )[0]
        eval(first)
        XCTAssertEqual(state.snapshot(spatialTokensPerFrame: 4).globalFrames, 2)

        let second = model(
            latents: [latent],
            timesteps: timesteps,
            embeddedContext: context,
            cameraConditioning: conditioning,
            causalState: state,
            currentStartToken: 8
        )[0]
        eval(second)
        XCTAssertEqual(state.snapshot(spatialTokensPerFrame: 4).globalFrames, 4)

        let recomputed = model(
            latents: [latent],
            timesteps: timesteps,
            embeddedContext: context,
            cameraConditioning: conditioning,
            causalState: state,
            currentStartToken: 8
        )[0]
        eval(recomputed)
        XCTAssertEqual(state.snapshot(spatialTokensPerFrame: 4).globalFrames, 4)

        let rolled = model(
            latents: [latent],
            timesteps: timesteps,
            embeddedContext: context,
            cameraConditioning: conditioning,
            causalState: state,
            currentStartToken: 16
        )[0]
        eval(rolled)
        let snapshot = state.snapshot(spatialTokensPerFrame: 4)
        XCTAssertEqual(snapshot.globalFrames, 6)
        XCTAssertEqual(snapshot.cachedFrames, 4)
        XCTAssertTrue(rolled.asArray(Float.self).allSatisfy(\.isFinite))
    }
}
