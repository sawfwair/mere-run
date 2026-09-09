import Foundation
import MLX
import MLXNN
import MLXRandom
import MereRunKVCache
import MereRunMLXTestSupport
import XCTest
@testable import MereRunImageModels
@testable import MereRunTensor
@testable import MereRunTextEncoder

final class ImageModelNumericsTests: MLXTestCase {
    func testFluxLatentPackingPreservesSpatialOrderAndRoundTrips() {
        let latents = MLXArray((0..<32).map(Float.init), [1, 2, 4, 4])
        let packed = Flux2LatentPacking.patchifyLatents(latents, height: 4, width: 4)
        XCTAssertEqual(packed.shape, [1, 8, 2, 2])
        XCTAssertEqual(packed[0, 0].asArray(Float.self), [0, 2, 8, 10])
        XCTAssertEqual(packed[0, 3].asArray(Float.self), [5, 7, 13, 15])
        let restored = Flux2LatentPacking.unpatchifyPackedLatents(packed, height: 2, width: 2)
        XCTAssertEqual(restored.shape, latents.shape)
        XCTAssertEqual(restored.asArray(Float.self), latents.asArray(Float.self))
    }

    func testQwenCachedDecodeMatchesFullCausalPassAfterCheckpointReload() throws {
        MLXRandom.seed(19)
        let configuration = QwenTextEncoderConfiguration(
            vocabSize: 32, hiddenSize: 8, numHiddenLayers: 2,
            numAttentionHeads: 2, numKeyValueHeads: 1, intermediateSize: 16,
            headDim: 4, useFloat32Activations: true
        )
        let original = QwenTextEncoder(configuration: configuration)
        let restored = QwenTextEncoder(configuration: configuration)
        try reloadWeights(from: original, into: restored)
        let ids = MLXArray([Int32(1), 7, 3, 9], [1, 4])
        let expected = original.encoder.forwardCausal(inputIds: ids, cache: nil)
        let caches: [KVCache] = (0..<2).map { _ in KVCacheSimple(step: 2) }
        var outputs: [MLXArray] = []
        for index in 0..<4 {
            outputs.append(restored.encoder.forwardCausal(
                inputIds: ids[0..., index..<(index + 1)], cache: caches
            ))
        }
        let actual = MLX.concatenated(outputs, axis: 1)
        assertClose(actual, expected, tolerance: 1e-4)
        XCTAssertEqual(caches.map(\.offset), [4, 4])
    }

    func testQwenPromptTrimmingPreservesPaddingAndHiddenStateSelection() throws {
        let hidden = MLXArray((0..<24).map(Float.init), [2, 4, 3])
        let mask = MLXArray([Int32(1), 1, 1, 1, 1, 1, 0, 0], [2, 4])
        let (embeddings, trimmedMask) = QwenTextEncoder.processTextEmbeddings(
            hiddenStates: hidden, attentionMask: mask, dropIndex: 1
        )
        XCTAssertEqual(embeddings.shape, [2, 3, 3])
        XCTAssertEqual(trimmedMask.asArray(Int32.self), [1, 1, 1, 1, 0, 0])
        XCTAssertEqual(embeddings[0].asArray(Float.self), Array(3..<12).map(Float.init))
        XCTAssertEqual(embeddings[1].asArray(Float.self), [15, 16, 17, 0, 0, 0, 0, 0, 0])

        let configuration = QwenTextEncoderConfiguration(
            vocabSize: 16, hiddenSize: 8, numHiddenLayers: 2,
            numAttentionHeads: 2, numKeyValueHeads: 1, intermediateSize: 16,
            headDim: 4, useFloat32Activations: true
        )
        let model = QwenTextEncoder(configuration: configuration)
        let ids = MLXArray([Int32(1), 2, 3, 0], [1, 4])
        let attentionMask = MLXArray([Int32(1), 1, 1, 0], [1, 4])
        let states = try XCTUnwrap(model.forwardWithHiddenStates(
            inputIds: ids, attentionMask: attentionMask
        ).hiddenStates)
        let zImageEmbeddings = try XCTUnwrap(model.encodeForZImage(
            inputIds: ids, attentionMask: attentionMask
        ).first)
        assertClose(zImageEmbeddings, states[states.count - 2][0, 0..<3], tolerance: 1e-6)
    }

    func testFluxTransformerCheckpointReloadAndBatchParity() throws {
        MLXRandom.seed(23)
        let configuration = Flux2TransformerConfiguration(
            hiddenSize: 16, numHeads: 2, headDim: 8, numLayers: 1,
            numSingleLayers: 1, inChannels: 8, contextDim: 8, mlpRatio: 2,
            axesDimsRope: [2, 2, 2, 2]
        )
        let original = Flux2Transformer2DModel(config: configuration)
        let restored = Flux2Transformer2DModel(config: configuration)
        try reloadWeights(from: original, into: restored)
        let latents = MLXRandom.normal([1, 4, 8])
        let prompt = MLXRandom.normal([1, 3, 8])
        let timestep = MLXArray([Float(0.5)])
        let imageIds = Flux2PosEmbed.prepareMultiImageIds(imageCount: 1, height: 2, width: 2, tCoords: [0])
        let textIds = Flux2PosEmbed.prepareTextIds(seqLen: 3, numAxes: 4)
        let expected = original(
            hiddenStates: latents, encoderHiddenStates: prompt,
            timestep: timestep, imgIds: imageIds, txtIds: textIds
        )
        let batch = restored(
            hiddenStates: MLX.concatenated([latents, latents], axis: 0),
            encoderHiddenStates: MLX.concatenated([prompt, prompt], axis: 0),
            timestep: MLX.concatenated([timestep, timestep]),
            imgIds: imageIds, txtIds: textIds
        )
        XCTAssertEqual(expected.shape, [1, 4, 8])
        assertClose(batch[0..<1], expected, tolerance: 1e-4)
        assertClose(batch[1..<2], expected, tolerance: 1e-4)
    }

    func testZImageTransformerCheckpointReloadAndCacheResetParity() throws {
        MLXRandom.seed(29)
        let configuration = ZImageTurboTransformerConfig(
            inChannels: 2, dim: 16, nLayers: 1, nRefinerLayers: 1,
            nHeads: 2, nKVHeads: 2, normEps: 1e-5, qkNorm: true,
            capFeatDim: 8, ropeTheta: 256, tScale: 1000,
            axesDims: [2, 2, 4], axesLens: [128, 32, 32],
            allPatchSize: [2], allFPatchSize: [1]
        )
        let original = ZImageTransformer2DModel(configuration: configuration)
        let restored = ZImageTransformer2DModel(configuration: configuration)
        try reloadWeights(from: original, into: restored)
        let latents = MLXRandom.normal([1, 2, 4, 4])
        let prompt = MLXRandom.normal([1, 3, 8])
        let timestep = MLXArray([Float(0.5)])
        let expected = original.forward(latents: latents, timestep: timestep, promptEmbeds: prompt)
        let actual = restored.forward(latents: latents, timestep: timestep, promptEmbeds: prompt)
        XCTAssertEqual(actual.shape, latents.shape)
        assertClose(actual, expected, tolerance: 1e-5)
        restored.clearCache()
        let afterReset = restored.forward(latents: latents, timestep: timestep, promptEmbeds: prompt)
        assertClose(afterReset, expected, tolerance: 1e-5)
    }

    func testVAEDecodeAppliesLatentScaleAndShiftBeforeDecoder() throws {
        MLXRandom.seed(31)
        let scaled = AutoencoderKL(configuration: VAEConfig(
            latentChannels: 2, scalingFactor: 0.5, shiftFactor: 0.25,
            blockOutChannels: [4, 8], layersPerBlock: 1, normNumGroups: 2,
            sampleSize: 8, midBlockAddAttention: true
        ))
        let plain = AutoencoderKL(configuration: VAEConfig(
            latentChannels: 2, scalingFactor: 1, shiftFactor: 0,
            blockOutChannels: [4, 8], layersPerBlock: 1, normNumGroups: 2,
            sampleSize: 8, midBlockAddAttention: true
        ))
        try reloadWeights(from: scaled, into: plain)
        let latents = MLXRandom.normal([1, 2, 4, 4])
        let actual = scaled.decode(latents).0
        let expected = plain.decode(latents / 0.5 + 0.25).0
        XCTAssertEqual(actual.shape, [1, 3, 8, 8])
        assertClose(actual, expected, tolerance: 1e-5)
        XCTAssertEqual(scaled.encode(actual).shape, [1, 4, 4, 4])
    }

    private func reloadWeights(from original: Module, into restored: Module) throws {
        let root = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("model.safetensors")
        let parameters = Dictionary(uniqueKeysWithValues: original.parameters().flattened())
        try MLX.save(arrays: parameters, url: file)
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: root.appendingPathComponent("model.safetensors.index.json"),
            singleURL: file, to: restored, dtype: nil, verify: [.noUnusedKeys, .shapeMismatch]
        )
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: MLXArray, tolerance: Float,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        let error = MLX.max(MLX.abs(actual.asType(.float32) - expected.asType(.float32))).item(Float.self)
        XCTAssertLessThanOrEqual(error, tolerance, file: file, line: line)
    }
}
