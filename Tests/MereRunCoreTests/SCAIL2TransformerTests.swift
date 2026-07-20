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

    func testBlockwiseEvaluationMatchesLazyForwardPath() {
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
            mode: .replacement
        )
        let prepared = model.prepareConditioning(
            textEmbeddings: input.textEmbeddings,
            imageEmbeddings: input.imageEmbeddings
        )
        let lazy = model(input, conditioning: prepared)
        eval(lazy)
        let blockwise = model(
            input,
            conditioning: prepared,
            evaluationBlockInterval: 1
        )
        eval(blockwise)

        XCTAssertEqual(lazy.asArray(Float.self), blockwise.asArray(Float.self))
    }

    func testQueryChunkedAttentionMatchesFullAttention() {
        let queries = MLXArray(
            (0..<(1 * 2 * 6 * 4)).map { Float($0) / 100 },
            [1, 2, 6, 4]
        )
        let keys = MLXArray(
            (0..<(1 * 2 * 7 * 4)).map { Float($0) / 110 },
            [1, 2, 7, 4]
        )
        let values = MLXArray(
            (0..<(1 * 2 * 7 * 4)).map { Float($0) / 90 },
            [1, 2, 7, 4]
        )
        let full = SCAIL2SelfAttention.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 0.5,
            maximumQueryTokens: nil
        )
        let chunked = SCAIL2SelfAttention.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 0.5,
            maximumQueryTokens: 2,
            maximumKernelsPerEvaluation: 2
        )
        eval(full, chunked)

        let fullValues = full.asArray(Float.self)
        let chunkedValues = chunked.asArray(Float.self)
        XCTAssertEqual(fullValues.count, chunkedValues.count)
        XCTAssertLessThan(
            zip(fullValues, chunkedValues).map { abs($0 - $1) }.max() ?? .infinity,
            1e-5
        )
    }

    func testDistilledAdapterMapsWanTargetsWithoutExternalRuntimeCode() {
        XCTAssertEqual(
            SCAIL2DistilledAdapter.targetBaseKey(
                for: "diffusion_model.blocks.3.self_attn.q"
            ),
            "blocks.3.self_attn.q"
        )
        XCTAssertEqual(
            SCAIL2DistilledAdapter.targetBaseKey(
                for: "diffusion_model.blocks.7.ffn.2"
            ),
            "blocks.7.ffn.fc2"
        )
        XCTAssertEqual(
            SCAIL2DistilledAdapter.targetBaseKey(
                for: "diffusion_model.img_emb.proj.4"
            ),
            "img_emb.layer_4"
        )
        XCTAssertEqual(
            SCAIL2DistilledAdapter.differenceTargetKey(
                for: "diffusion_model.text_embedding.2.diff_b"
            ),
            "text_embedding_1.bias"
        )
        XCTAssertNil(
            SCAIL2DistilledAdapter.targetBaseKey(
                for: "foreign_runtime.blocks.0.self_attn.q"
            )
        )
        XCTAssertTrue(
            SCAIL2DistilledAdapter.isTaskInputProjectionDifference(
                sourceKey: "diffusion_model.patch_embedding.diff",
                sourceShape: [5_120, 36, 1, 2, 2],
                targetShape: [5_120, 80]
            )
        )
        XCTAssertFalse(
            SCAIL2DistilledAdapter.isTaskInputProjectionDifference(
                sourceKey: "diffusion_model.blocks.0.self_attn.q.diff",
                sourceShape: [5_120, 5_120],
                targetShape: [5_120, 5_120]
            )
        )
    }

    func testDistilledAdapterFusesPairsAndDifferenceParameters() throws {
        let model = SCAIL2TransformerModel(configuration: configuration)
        let query = model.blocks[0].selfAttention.query
        let originalWeight = query.weight
        let originalBias = try XCTUnwrap(query.bias)
        eval(originalWeight, originalBias)
        let originalWeightSnapshot = MLXArray(
            originalWeight.asArray(Float.self),
            originalWeight.shape
        )
        let originalBiasSnapshot = MLXArray(
            originalBias.asArray(Float.self),
            originalBias.shape
        )

        let adapterURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("scail2-distilled-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: adapterURL) }
        let down = MLX.full([2, configuration.hiddenSize], values: MLXArray(Float(0.25)))
        let up = MLX.full([configuration.hiddenSize, 2], values: MLXArray(Float(0.5)))
        let biasDifference = MLX.full(
            [configuration.hiddenSize],
            values: MLXArray(Float(0.75))
        )
        try MLX.save(
            arrays: [
                "diffusion_model.blocks.0.self_attn.q.lora_down.weight": down,
                "diffusion_model.blocks.0.self_attn.q.lora_up.weight": up,
                "diffusion_model.blocks.0.self_attn.q.diff_b": biasDifference,
            ],
            metadata: [:],
            url: adapterURL
        )

        let result = try SCAIL2DistilledAdapter.apply(
            url: adapterURL,
            to: model,
            strength: 0.5,
            expectedPairCount: 1,
            expectedDifferenceCount: 1
        )
        let fusedQuery = model.blocks[0].selfAttention.query
        let expectedWeight = originalWeightSnapshot + MLX.full(
            originalWeight.shape,
            values: MLXArray(Float(0.125))
        )
        let expectedBias = originalBiasSnapshot + MLX.full(
            originalBias.shape,
            values: MLXArray(Float(0.375))
        )
        eval(fusedQuery.weight, fusedQuery.bias!, expectedWeight, expectedBias)

        XCTAssertEqual(result, .init(pairCount: 1, differenceCount: 1))
        XCTAssertLessThan(
            MLX.max(MLX.abs(fusedQuery.weight - expectedWeight)).item(Float.self),
            1e-5
        )
        XCTAssertLessThan(
            MLX.max(MLX.abs(fusedQuery.bias! - expectedBias)).item(Float.self),
            1e-5
        )
    }
}
