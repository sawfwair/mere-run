import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class Wan2DreamXCameraWeightsTests: MereRunCoreTestCase {
    func testConvertedCausalCheckpointMatchesReleasedInventory() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_DREAMX_CAUSAL_WEIGHTS"] else {
            throw XCTSkip("Set MERERUN_DREAMX_CAUSAL_WEIGHTS to the converted causal checkpoint.")
        }
        let url = URL(fileURLWithPath: path)
        let metadata = try SafetensorsStreamingLoader.metadata(url: url)
        XCTAssertEqual(metadata.count, 1_125)
        XCTAssertEqual(metadata["blocks.0.cam_self_attn.q_proj.weight"]?.shape, [768, 3_072])
        XCTAssertEqual(metadata["blocks.29.cam_self_attn.out_proj.weight"]?.shape, [3_072, 768])
        XCTAssertEqual(metadata["patch_embedding_proj.weight"]?.shape, [3_072, 192])
        let model = try Wan2ModelLoader.loadDreamXCausalTransformer(weightsURL: url)
        let parameterKeys = Set(model.parameters().flattened().map(\.0))
            .subtracting(["inverseTimestepFrequencies", "ropeFrequencies"])
        XCTAssertEqual(parameterKeys, Set(metadata.keys))
    }

    func testExtractedCameraAdapterMatchesReleasedInventory() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_DREAMX_CAMERA_WEIGHTS"] else {
            throw XCTSkip("Set MERERUN_DREAMX_CAMERA_WEIGHTS to the extracted camera adapter.")
        }
        let metadata = try SafetensorsStreamingLoader.metadata(url: URL(fileURLWithPath: path))
        XCTAssertEqual(metadata.count, 300)
        XCTAssertTrue(metadata.keys.allSatisfy { $0.contains(".cam_self_attn.") })
        for block in 0..<30 {
            let prefix = "blocks.\(block).cam_self_attn."
            XCTAssertEqual(metadata.keys.filter { $0.hasPrefix(prefix) }.count, 10)
            XCTAssertEqual(metadata[prefix + "q_proj.weight"]?.shape, [3_072, 3_072])
            XCTAssertEqual(metadata[prefix + "out_proj.bias"]?.shape, [3_072])
            XCTAssertEqual(metadata[prefix + "norm_q.weight"]?.shape, [3_072])
        }
    }

    func testRealBlockCameraAttentionMatchesDreamXFixture() throws {
        guard let weightsPath = ProcessInfo.processInfo.environment["MERERUN_DREAMX_CAMERA_WEIGHTS"],
              let fixturePath = ProcessInfo.processInfo.environment["MERERUN_DREAMX_CAMERA_BLOCK_FIXTURE"] else {
            throw XCTSkip("Set DreamX camera weights and block fixture environment variables.")
        }
        let weightsURL = URL(fileURLWithPath: weightsPath)
        let fixture = try SafetensorsStreamingLoader.loadArrays(url: URL(fileURLWithPath: fixturePath))
        let attention = Wan2ProjectiveSelfAttention(dimensions: 3_072, heads: 24, epsilon: 1e-6)
        let prefix = "blocks.0.cam_self_attn."
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: weightsURL,
            to: attention,
            dtype: .float32,
            verify: .noUnusedKeys,
            include: { $0.hasPrefix(prefix) },
            mapper: { key, value in [(String(key.dropFirst(prefix.count)), value)] },
            batchSize: 10
        )
        guard let input = fixture["input"],
              let views = fixture["view_matrices"],
              let intrinsics = fixture["intrinsics"],
              let expected = fixture["output"] else {
            XCTFail("DreamX camera block fixture is incomplete.")
            return
        }
        let conditioning = Wan2ProjectiveCameraConditioning(
            frameCount: views.dim(1),
            viewMatrices: views.asArray(Float.self),
            intrinsics: intrinsics.asArray(Float.self)
        )
        let actual = attention(input, conditioning: conditioning).asType(.float32)
        eval(actual)
        XCTAssertEqual(actual.shape, expected.shape)
        let actualValues = actual.asArray(Float.self)
        let expectedValues = expected.asArray(Float.self)
        let dot = zip(actualValues, expectedValues).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        let actualNorm = sqrt(actualValues.reduce(Float(0)) { $0 + $1 * $1 })
        let expectedNorm = sqrt(expectedValues.reduce(Float(0)) { $0 + $1 * $1 })
        let meanAbsoluteError = zip(actualValues, expectedValues)
            .reduce(Float(0)) { $0 + abs($1.0 - $1.1) } / Float(actualValues.count)
        XCTAssertGreaterThan(dot / (actualNorm * expectedNorm), 0.9999)
        XCTAssertLessThan(meanAbsoluteError, 2e-4)
    }

    func testFullReleasedTransformerMatchesDreamXFixture() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_WAN2_MODEL_ROOT"],
              let weightsPath = environment["MERERUN_DREAMX_CAMERA_WEIGHTS"],
              let fixturePath = environment["MERERUN_DREAMX_CAMERA_TRANSFORMER_FIXTURE"] else {
            throw XCTSkip("Set the Wan2 model, DreamX camera weights, and full transformer fixture variables.")
        }
        let fixture = try SafetensorsStreamingLoader.loadArrays(
            url: URL(fileURLWithPath: fixturePath)
        )
        guard let input = fixture["input"],
              let timesteps = fixture["timesteps"],
              let context = fixture["context"],
              let views = fixture["view_matrices"],
              let intrinsics = fixture["intrinsics"],
              let expected = fixture["output"] else {
            XCTFail("DreamX full transformer fixture is incomplete.")
            return
        }
        let model = try Wan2ModelLoader.loadDreamXCameraTransformer(
            resources: Wan2Resources(rootURL: URL(fileURLWithPath: root)),
            cameraWeightsURL: URL(fileURLWithPath: weightsPath)
        )
        let conditioning = Wan2ProjectiveCameraConditioning(
            frameCount: views.dim(1),
            viewMatrices: views.asArray(Float.self),
            intrinsics: intrinsics.asArray(Float.self)
        )
        let embeddedContext = model.embedText(context.expandedDimensions(axis: 0))
        let actual = model(
            latents: [input],
            timesteps: timesteps,
            embeddedContext: embeddedContext,
            cameraConditioning: conditioning
        )[0].asType(.float32)
        eval(actual)
        XCTAssertEqual(actual.shape, expected.shape)
        let actualValues = actual.asArray(Float.self)
        let expectedValues = expected.asArray(Float.self)
        let dot = zip(actualValues, expectedValues).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        let actualNorm = sqrt(actualValues.reduce(Float(0)) { $0 + $1 * $1 })
        let expectedNorm = sqrt(expectedValues.reduce(Float(0)) { $0 + $1 * $1 })
        let meanAbsoluteError = zip(actualValues, expectedValues)
            .reduce(Float(0)) { $0 + abs($1.0 - $1.1) } / Float(actualValues.count)
        let cosineSimilarity = dot / (actualNorm * expectedNorm)
        XCTAssertGreaterThan(cosineSimilarity, 0.9995)
        XCTAssertLessThan(meanAbsoluteError, 0.02)
    }

    func testFullReleasedCausalTransformerMatchesDreamXFixture() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let weightsPath = environment["MERERUN_DREAMX_CAUSAL_WEIGHTS"],
              let fixturePath = environment["MERERUN_DREAMX_CAUSAL_TRANSFORMER_FIXTURE"] else {
            throw XCTSkip("Set the DreamX causal weights and transformer fixture variables.")
        }
        let fixture = try SafetensorsStreamingLoader.loadArrays(
            url: URL(fileURLWithPath: fixturePath)
        )
        guard let input = fixture["input"],
              let timesteps = fixture["timesteps"],
              let context = fixture["context"],
              let views = fixture["view_matrices"],
              let intrinsics = fixture["intrinsics"],
              let expected = fixture["output"],
              let recomputeInput = fixture["recompute_input"],
              let recomputeTimesteps = fixture["recompute_timesteps"],
              let recomputeExpected = fixture["recompute_output"],
              let secondInput = fixture["second_input"],
              let secondTimesteps = fixture["second_timesteps"],
              let secondViews = fixture["second_view_matrices"],
              let secondExpected = fixture["second_output"] else {
            XCTFail("DreamX causal transformer fixture is incomplete.")
            return
        }
        let model = try Wan2ModelLoader.loadDreamXCausalTransformer(
            weightsURL: URL(fileURLWithPath: weightsPath)
        )
        let conditioning = Wan2ProjectiveCameraConditioning(
            frameCount: views.dim(1),
            viewMatrices: views.asArray(Float.self),
            intrinsics: intrinsics.asArray(Float.self)
        )
        let state = Wan2CausalTransformerState(
            layerCount: 30,
            localAttentionFrames: 12,
            sinkFrames: 3
        )
        let embeddedContext = model.embedText(context.expandedDimensions(axis: 0))
        let crossCaches = model.prepareCrossAttentionCaches(context: embeddedContext)
        let actual = model(
            latents: [input],
            timesteps: timesteps,
            embeddedContext: embeddedContext,
            crossCaches: crossCaches,
            cameraConditioning: conditioning,
            causalState: state,
            currentStartToken: 0
        )[0].asType(.float32)
        eval(actual)
        assertParity(actual: actual, expected: expected, label: "initial block")

        let recomputeActual = model(
            latents: [recomputeInput],
            timesteps: recomputeTimesteps,
            embeddedContext: embeddedContext,
            crossCaches: crossCaches,
            cameraConditioning: conditioning,
            causalState: state,
            currentStartToken: 0
        )[0].asType(.float32)
        eval(recomputeActual)
        assertParity(actual: recomputeActual, expected: recomputeExpected, label: "recomputed block")

        let secondConditioning = Wan2ProjectiveCameraConditioning(
            frameCount: secondViews.dim(1),
            viewMatrices: secondViews.asArray(Float.self),
            intrinsics: intrinsics.asArray(Float.self)
        )
        let secondActual = model(
            latents: [secondInput],
            timesteps: secondTimesteps,
            embeddedContext: embeddedContext,
            crossCaches: crossCaches,
            cameraConditioning: secondConditioning,
            causalState: state,
            currentStartToken: 6
        )[0].asType(.float32)
        eval(secondActual)
        assertParity(actual: secondActual, expected: secondExpected, label: "appended block")
        let snapshot = state.snapshot(spatialTokensPerFrame: 2)
        XCTAssertEqual(snapshot.cachedFrames, 6)
        XCTAssertEqual(snapshot.globalFrames, 6)
    }

    private func assertParity(
        actual: MLXArray,
        expected: MLXArray,
        label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, label, file: file, line: line)
        let actualValues = actual.asArray(Float.self)
        let expectedValues = expected.asArray(Float.self)
        let dot = zip(actualValues, expectedValues).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        let actualNorm = sqrt(actualValues.reduce(Float(0)) { $0 + $1 * $1 })
        let expectedNorm = sqrt(expectedValues.reduce(Float(0)) { $0 + $1 * $1 })
        let meanAbsoluteError = zip(actualValues, expectedValues)
            .reduce(Float(0)) { $0 + abs($1.0 - $1.1) } / Float(actualValues.count)
        let cosine = dot / (actualNorm * expectedNorm)
        print("DreamX causal parity \(label): cosine=\(cosine), mae=\(meanAbsoluteError)")
        XCTAssertGreaterThan(cosine, 0.995, label, file: file, line: line)
        XCTAssertLessThan(meanAbsoluteError, 0.05, label, file: file, line: line)
    }
}
