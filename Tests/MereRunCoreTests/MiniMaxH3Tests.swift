import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import XCTest
@testable import MereRunCore

final class MiniMaxH3Tests: MereRunCoreTestCase {
    func testExactKernelModeEnvironmentIsTypedAndFailsClosed() throws {
        XCTAssertEqual(
            try MiniMaxH3ExactKernelMode.resolve(environmentValue: nil),
            .disabled
        )
        XCTAssertEqual(
            try MiniMaxH3ExactKernelMode.resolve(environmentValue: "disabled"),
            .disabled
        )
        XCTAssertEqual(
            try MiniMaxH3ExactKernelMode.resolve(environmentValue: "BOUNDARY-LAYOUT"),
            .boundaryLayout
        )
        XCTAssertEqual(
            try MiniMaxH3ExactKernelMode.resolve(environmentValue: "AFFINE-Q8"),
            .affineQ8
        )
        XCTAssertThrowsError(
            try MiniMaxH3ExactKernelMode.resolve(environmentValue: "automatic")
        ) { error in
            XCTAssertTrue(
                error.localizedDescription.contains(
                    "must be disabled, boundary-layout, or affine-q8"
                )
            )
        }
    }

    func testExactKernelModeFallsBackForUnsupportedTransformerContracts() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 12,
            layerCount: 1,
            refinerLayerCount: 1,
            attentionHeadCount: 2,
            attentionHeadDimension: 6,
            feedForwardSize: 16,
            videoLatentChannels: 3,
            audioLatentChannels: 4,
            patchSize: [1, 2, 2],
            textDimension: 10,
            timeFrequencyDimension: 4,
            timeEmbeddingHiddenSize: 12,
            timeEmbeddingDimension: 8,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1, 1],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3,
            keyframeAnchors: [.first]
        )
        let video = MLXArray.zeros([1, 12, 12])
        let audio = MLXArray.zeros([1, 6, 4])
        let text = MLXArray.zeros([1, 2, 10])
        let baseline = model(
            videoRows: video,
            audioRows: audio,
            textStates: text,
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(baseline.videoVelocityRows, baseline.audioVelocityRows)

        XCTAssertFalse(model.supportsAffineQ8ExactKernels)
        model.exactKernelMode = .affineQ8
        let fallback = model(
            videoRows: video,
            audioRows: audio,
            textStates: text,
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(fallback.videoVelocityRows, fallback.audioVelocityRows)

        XCTAssertEqual(
            MLX.max(MLX.abs(
                baseline.videoVelocityRows - fallback.videoVelocityRows
            )).item(Float.self),
            0
        )
        XCTAssertEqual(
            MLX.max(MLX.abs(
                baseline.audioVelocityRows - fallback.audioVelocityRows
            )).item(Float.self),
            0
        )
    }

    func testDynamicSparseAttentionPolicyProtectsDenseBoundaries() throws {
        XCTAssertNil(MiniMaxH3AccelerationMode.quality.dynamicSparseAttentionPolicy)
        let balanced = try XCTUnwrap(
            MiniMaxH3AccelerationMode.balanced.dynamicSparseAttentionPolicy
        )
        let maximum = try XCTUnwrap(
            MiniMaxH3AccelerationMode.maximum.dynamicSparseAttentionPolicy
        )
        XCTAssertEqual(balanced.thresholdStandardDeviations, 0.75)
        XCTAssertEqual(maximum.thresholdStandardDeviations, 1)

        XCTAssertNil(maximum.request(
            stepIndex: 1,
            stepCount: 8,
            layerIndex: 2,
            sequenceLength: 12_930,
            prefixTokenCount: 900
        ))
        XCTAssertNil(maximum.request(
            stepIndex: 2,
            stepCount: 8,
            layerIndex: 1,
            sequenceLength: 12_930,
            prefixTokenCount: 900
        ))
        XCTAssertNotNil(maximum.request(
            stepIndex: 2,
            stepCount: 8,
            layerIndex: 2,
            sequenceLength: 12_930,
            prefixTokenCount: 900
        ))
        XCTAssertNil(maximum.request(
            stepIndex: 7,
            stepCount: 8,
            layerIndex: 49,
            sequenceLength: 12_930,
            prefixTokenCount: 900
        ))
        XCTAssertNil(maximum.request(
            stepIndex: 2,
            stepCount: 8,
            layerIndex: 2,
            sequenceLength: 11_999,
            prefixTokenCount: 900
        ))
    }

    func testDynamicSparseAttentionRoutesAboveAdaptiveThreshold() {
        let queryCentroids = MLXArray([Float(1), 0]).reshaped(1, 1, 1, 2)
        let keyCentroids = MLXArray([
            Float(-2), 0,
            -1, 0,
            0, 0,
            3, 0,
        ]).reshaped(1, 1, 4, 2)
        let routes = DynamicSparseAttention.routesForTesting(
            queryCentroids: queryCentroids,
            keyCentroids: keyCentroids,
            thresholdStandardDeviations: 0
        )
        MLX.eval(routes)
        XCTAssertEqual(routes.asArray(UInt8.self), [0, 0, 0, 1])
    }

    func testDynamicSparseMetalDenseRouteMatchesFusedSDPA() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Dynamic sparse attention Metal parity requires a GPU.")
        }
        MLXRandom.seed(41)
        let shape = [1, 2, 256, DynamicSparseAttention.headDimension]
        let queries = (MLXRandom.normal(shape) * Float(0.2)).asType(.bfloat16)
        let keys = (MLXRandom.normal(shape) * Float(0.2)).asType(.bfloat16)
        let values = MLXRandom.normal(shape).asType(.bfloat16)
        MLX.eval(queries, keys, values)

        let gate = try XCTUnwrap(DynamicSparseAttention.denseRouteGate(
            queries: queries,
            keys: keys,
            values: values,
            queryStart: 128,
            scale: 1 / sqrt(Float(DynamicSparseAttention.headDimension))
        ))
        XCTAssertTrue(gate.passed)
        XCTAssertLessThanOrEqual(gate.relativeL2Error, 0.005)
    }

    func testDynamicSparseMetalFloat32InputsPassDenseGate() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Dynamic sparse attention FP32 gate requires a GPU.")
        }
        MLXRandom.seed(2_026_081)
        let shape = [1, 2, 256, DynamicSparseAttention.headDimension]
        let queries = MLXRandom.normal(shape)
        let keys = MLXRandom.normal(shape)
        let values = MLXRandom.normal(shape)
        MLX.eval(queries, keys, values)

        let gate = try XCTUnwrap(DynamicSparseAttention.denseRouteGate(
            queries: queries,
            keys: keys,
            values: values,
            queryStart: 64,
            scale: 1 / sqrt(Float(DynamicSparseAttention.headDimension))
        ))
        XCTAssertTrue(gate.passed)
    }

    func testDynamicSparseMetalRetainsSkippedBlockCentroidContribution() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Dynamic sparse attention Metal correction requires a GPU.")
        }
        MLXRandom.seed(42)
        let heads = 2
        let tokens = 256
        let dimension = DynamicSparseAttention.headDimension
        let shape = [1, heads, tokens, dimension]
        let queries = (MLXRandom.normal(shape) * Float(0.2)).asType(.bfloat16)
        let keys = (MLXRandom.normal(shape) * Float(0.2)).asType(.bfloat16)
        let values = MLXRandom.normal(shape).asType(.bfloat16)
        let routes = MLXArray.zeros([1, heads, 1, 4], dtype: .uint8)
        let scale = 1 / sqrt(Float(dimension))
        let candidate = try XCTUnwrap(
            DynamicSparseAttention.sparseOutputForTesting(
                queries: queries,
                keys: keys,
                values: values,
                routes: routes,
                queryStart: 192,
                queryCount: 1,
                prefixTokenCount: 64,
                scale: scale
            )
        )

        let skippedCentroid = keys[0..., 0..., 64..<128, 0...]
            .mean(axis: 2, keepDims: true)
        let approximatedKeys = MLX.concatenated([
            keys[0..., 0..., 0..<64, 0...],
            MLX.broadcast(skippedCentroid, to: [1, heads, 64, dimension]),
            keys[0..., 0..., 128..., 0...],
        ], axis: 2)
        let reference = MLXFast.scaledDotProductAttention(
            queries: queries[0..., 0..., 192..<193, 0...],
            keys: approximatedKeys,
            values: values,
            scale: scale,
            mask: .none
        )
        let delta = candidate.asType(.float32) - reference.asType(.float32)
        let relativeL2 = MLX.sqrt(
            MLX.sum(delta * delta)
                / MLX.maximum(
                    MLX.sum(reference.asType(.float32) * reference.asType(.float32)),
                    MLXArray(Float(1e-12))
                )
        )
        MLX.eval(candidate, reference, relativeL2)
        XCTAssertLessThanOrEqual(relativeL2.item(Float.self), 0.005)
    }

    func testRuntimeLoRAAppliesDeltaInActivationSpace() {
        let base = Linear(
            weight: MLXArray([Float(1), 0, 0, 1]).reshaped(2, 2),
            bias: nil
        )
        let layer = MiniMaxH3RuntimeLoRALinear(
            base: base,
            loraDown: MLXArray([Float(1), 0]).reshaped(1, 2),
            loraUp: MLXArray([Float(2), 3]).reshaped(2, 1),
            strength: 0.5
        )
        let output = layer(MLXArray([Float(4), 5]).reshaped(1, 2))
        MLX.eval(output)
        XCTAssertEqual(output.asArray(Float.self), [8, 11])
    }

    func testLightX2VPEFTAdapterFusesOnceIntoNativeRuntime() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        let adapterURL = temp.appendingPathComponent("lightx2v-h3.safetensors")
        let rankOneInput = MLXArray.ones([1, 2], dtype: .float32)
        let rankOneAttentionOutput = MLXArray.ones([2, 1], dtype: .float32)
        try MLX.save(
            arrays: [
                "transformer_blocks.0.attn.to_q.lora_A.default.weight": rankOneInput,
                "transformer_blocks.0.attn.to_q.lora_B.default.weight": rankOneAttentionOutput,
                "transformer_blocks.0.attn.to_k.lora_A.default.weight": rankOneInput,
                "transformer_blocks.0.attn.to_k.lora_B.default.weight": rankOneAttentionOutput,
                "transformer_blocks.0.attn.to_v.lora_A.default.weight": rankOneInput,
                "transformer_blocks.0.attn.to_v.lora_B.default.weight": rankOneAttentionOutput,
                "transformer_blocks.0.attn.to_out.0.lora_A.default.weight": rankOneInput,
                "transformer_blocks.0.attn.to_out.0.lora_B.default.weight": rankOneAttentionOutput,
                "transformer_blocks.0.ff.net.0.proj.lora_A.default.weight": rankOneInput,
                "transformer_blocks.0.ff.net.0.proj.lora_B.default.weight": MLXArray.ones(
                    [6, 1],
                    dtype: .float32
                ),
                "transformer_blocks.0.ff.net.2.lora_A.default.weight": MLXArray.ones(
                    [1, 3],
                    dtype: .float32
                ),
                "transformer_blocks.0.ff.net.2.lora_B.default.weight": rankOneAttentionOutput,
            ],
            url: adapterURL
        )

        let transformer = MiniMaxH3Transformer(
            configuration: MiniMaxH3TransformerConfiguration(
                hiddenSize: 2,
                layerCount: 1,
                refinerLayerCount: 0,
                attentionHeadCount: 1,
                attentionHeadDimension: 2,
                feedForwardSize: 3,
                videoLatentChannels: 1,
                audioLatentChannels: 1,
                patchSize: [1, 1, 1],
                textDimension: 2,
                timeFrequencyDimension: 2,
                timeEmbeddingHiddenSize: 2,
                timeEmbeddingDimension: 2,
                ropeFrequencyCount: 1
            ),
            includeAdaLN: false
        )
        let originalLeaves = Dictionary(
            uniqueKeysWithValues: transformer.leafModules().flattened()
        )
        let originalQKV = try XCTUnwrap(
            originalLeaves["blocks.0.attn.qkv_proj"] as? Linear
        ).weight
        MLX.eval(originalQKV)
        let originalQKVValues = originalQKV.asArray(Float.self)
        let count = try MiniMaxH3TurboAdapter.install(
            url: adapterURL,
            into: transformer,
            strength: 1,
            expectedPairCount: 6
        )
        let leaves = Dictionary(uniqueKeysWithValues: transformer.leafModules().flattened())
        let qkv = try XCTUnwrap(leaves["blocks.0.attn.qkv_proj"] as? Linear)
        MLX.eval(qkv.weight)
        let qkvDelta = zip(qkv.weight.asArray(Float.self), originalQKVValues).map {
            $0.0 - $0.1
        }

        XCTAssertEqual(count, 6)
        XCTAssertEqual(qkvDelta.count, 12)
        for value in qkvDelta {
            XCTAssertEqual(value, 8, accuracy: 1e-5)
        }
        XCTAssertFalse(qkv is MiniMaxH3RuntimeLoRALinear)
        XCTAssertFalse(leaves["blocks.0.attn.out_proj"] is MiniMaxH3RuntimeLoRALinear)
        XCTAssertFalse(leaves["blocks.0.mlp.fc1"] is MiniMaxH3RuntimeLoRALinear)
        XCTAssertFalse(leaves["blocks.0.mlp.fc2"] is MiniMaxH3RuntimeLoRALinear)
    }

    func testTurboAdapterUsesFourDenoiseEvaluationsByDefault() throws {
        let options = try MiniMaxH3GenerationOptions(
            prompt: "a cinematic local video",
            width: 256,
            height: 160,
            numFrames: 22,
            adapterURL: URL(fileURLWithPath: "/tmp/minimax-h3-turbo.safetensors")
        )
        XCTAssertEqual(options.steps, 5)
        XCTAssertEqual(options.adapterStrength, 1)

        let sparseTurbo = try MiniMaxH3GenerationOptions(
            prompt: "Turbo with attention-only acceleration",
            width: 256,
            height: 160,
            numFrames: 22,
            accelerationMode: .maximum,
            adapterURL: URL(fileURLWithPath: "/tmp/minimax-h3-turbo.safetensors")
        )
        XCTAssertEqual(sparseTurbo.steps, 5)
        XCTAssertEqual(sparseTurbo.accelerationMode, .maximum)
    }

    func testVelocityReusePolicyPreservesFirstAndFinalDenoiseEvaluations() {
        let policy = MiniMaxH3VelocityReusePolicy(interval: 2)

        XCTAssertFalse(policy.shouldReuse(stepIndex: 0, stepCount: 8, hasCachedVelocity: false))
        XCTAssertTrue(policy.shouldReuse(stepIndex: 1, stepCount: 8, hasCachedVelocity: true))
        XCTAssertFalse(policy.shouldReuse(stepIndex: 2, stepCount: 8, hasCachedVelocity: true))
        XCTAssertTrue(policy.shouldReuse(stepIndex: 5, stepCount: 8, hasCachedVelocity: true))
        XCTAssertFalse(policy.shouldReuse(stepIndex: 6, stepCount: 8, hasCachedVelocity: true))
        XCTAssertFalse(policy.shouldReuse(stepIndex: 7, stepCount: 8, hasCachedVelocity: true))
    }

    func testLayerThinningRanksAdaLNGatesAndProtectsStructuralBlocks() {
        let scores: [Float] = [0, 0, 0.1, 0.9, 0.2, 0]
        let modulations = scores.map { score in
            MLXArray.full(
                [2, 9, 6 * 4],
                values: MLXArray(score).asType(.bfloat16)
            )
        }

        let active = MiniMaxH3LayerThinningPolicy(activeBlockCount: 4)
            .activeBlockIndices(blockModulations: modulations)

        XCTAssertEqual(active, [0, 1, 3, 5])
    }

    func testVelocityReuseAccelerationKeepsQualityScheduleAndHasNoOtherApproximationPolicy() throws {
        let options = try MiniMaxH3GenerationOptions(
            prompt: "an interval two velocity reuse bake-off",
            width: 768,
            height: 448,
            numFrames: 124,
            accelerationMode: .velocityReuse2
        )

        XCTAssertEqual(options.steps, 9)
        XCTAssertEqual(options.accelerationMode.velocityReusePolicy?.interval, 2)
        XCTAssertNil(options.accelerationMode.adaptiveFirstBlockCachePolicy)
        XCTAssertNil(options.accelerationMode.blockReusePolicy)
        XCTAssertNil(options.accelerationMode.dynamicSparseAttentionPolicy)
    }

    func testLayerThinningAccelerationKeepsQualityScheduleAndIsIsolated() throws {
        let options = try MiniMaxH3GenerationOptions(
            prompt: "a gate ranked layer thinning bake-off",
            width: 768,
            height: 448,
            numFrames: 124,
            accelerationMode: .layers45
        )

        XCTAssertEqual(options.steps, 9)
        XCTAssertEqual(options.accelerationMode.layerThinningPolicy?.activeBlockCount, 45)
        XCTAssertNil(options.accelerationMode.velocityReusePolicy)
        XCTAssertNil(options.accelerationMode.adaptiveFirstBlockCachePolicy)
        XCTAssertNil(options.accelerationMode.blockReusePolicy)
        XCTAssertNil(options.accelerationMode.dynamicSparseAttentionPolicy)
    }

    func testTokenReductionPolicyMatchesPinnedH3CBlockSchedule() throws {
        let policy = MiniMaxH3TokenReductionPolicy()
        let options = try MiniMaxH3GenerationOptions(
            prompt: "a horizontal token reduction bake-off",
            width: 768,
            height: 448,
            numFrames: 124,
            accelerationMode: .tokenReduction
        )

        XCTAssertEqual(policy.beginBlock, 4)
        XCTAssertEqual(policy.restoreBeforeBlock(stepIndex: 0), 40)
        XCTAssertEqual(policy.restoreBeforeBlock(stepIndex: 9), 40)
        XCTAssertEqual(policy.restoreBeforeBlock(stepIndex: 10), 30)
        XCTAssertEqual(options.steps, 9)
        XCTAssertNil(options.accelerationMode.velocityReusePolicy)
        XCTAssertNil(options.accelerationMode.layerThinningPolicy)
        XCTAssertNil(options.accelerationMode.adaptiveFirstBlockCachePolicy)
        XCTAssertNil(options.accelerationMode.dynamicSparseAttentionPolicy)
    }

    func testTokenReductionPreservesPairDetailThroughBypassedReconstruction() {
        let layout = MiniMaxH3PackedLayout(
            positions: MLXArray.zeros([7, 3]),
            tokenTags: Array(repeating: Int32(0), count: 7),
            textRows: 0..<1,
            conditionRows: 1..<1,
            conditionSegments: [],
            conditionVideoRowCount: 0,
            conditionAudioRowCount: 0,
            targetAudioRows: 1..<1,
            targetVideoRows: 1..<7,
            videoLatentFrames: 1,
            latentHeight: 4,
            latentWidth: 6,
            audioLatentFrames: 0
        )
        let map = MiniMaxH3TokenReductionMap(layout: layout)
        let hidden = MLXArray([Float(100), 0, 2, 10, 14, 20, 26])
            .reshaped(1, 7, 1)
        let state = map.pool(hidden)
        MLX.eval(state.reducedHidden)

        XCTAssertEqual(state.reducedHidden.shape, [1, 5, 1])
        XCTAssertEqual(state.reducedHidden.asArray(Float.self), [100, 1, 10, 17, 26])

        let processed = MLXArray([Float(101), 4, 13, 21, 30]).reshaped(1, 5, 1)
        let restored = map.restore(processed, state: state, updateScale: 1)
        MLX.eval(restored)
        XCTAssertEqual(restored.asArray(Float.self), [101, 3, 5, 13, 18, 24, 30])
    }

    func testReducedRenderCanvasUsesInternalGeometryAndPreservesOutputGeometry() throws {
        let options = try MiniMaxH3GenerationOptions(
            prompt: "an internal canvas bake-off",
            width: 512,
            height: 512,
            renderWidth: 384,
            renderHeight: 384,
            numFrames: 124
        )

        XCTAssertEqual(options.width, 512)
        XCTAssertEqual(options.height, 512)
        XCTAssertEqual(options.internalWidth, 384)
        XCTAssertEqual(options.internalHeight, 384)
        XCTAssertTrue(options.usesReducedRenderCanvas)
        XCTAssertEqual(options.steps, 9)
    }

    func testReducedRenderCanvasRequiresOracleCompatibleGeometry() {
        XCTAssertThrowsError(try MiniMaxH3GenerationOptions(
            prompt: "missing render height",
            width: 512,
            height: 512,
            renderWidth: 384
        ))
        XCTAssertThrowsError(try MiniMaxH3GenerationOptions(
            prompt: "different aspect",
            width: 512,
            height: 512,
            renderWidth: 384,
            renderHeight: 320
        ))
        XCTAssertThrowsError(try MiniMaxH3GenerationOptions(
            prompt: "larger internal canvas",
            width: 512,
            height: 512,
            renderWidth: 544,
            renderHeight: 544
        ))
        XCTAssertThrowsError(try MiniMaxH3GenerationOptions(
            prompt: "off grid internal canvas",
            width: 512,
            height: 512,
            renderWidth: 400,
            renderHeight: 400
        ))
    }

    func testPositionedFrameInputsAreValidatedAndSorted() throws {
        let later = URL(fileURLWithPath: "/tmp/later.png")
        let earlier = URL(fileURLWithPath: "/tmp/earlier.png")
        let options = try MiniMaxH3GenerationOptions(
            prompt: "a continuous staged scene",
            width: 256,
            height: 160,
            numFrames: 90,
            frameInputs: [
                .init(frameIndex: 72, url: later),
                .init(frameIndex: 24, url: earlier),
            ]
        )

        XCTAssertEqual(options.frameInputs.map(\.frameIndex), [24, 72])
        XCTAssertThrowsError(try MiniMaxH3GenerationOptions(
            prompt: "duplicate frames",
            width: 256,
            height: 160,
            numFrames: 90,
            frameInputs: [
                .init(frameIndex: 24, url: earlier),
                .init(frameIndex: 24, url: later),
            ]
        ))
        XCTAssertThrowsError(try MiniMaxH3GenerationOptions(
            prompt: "out of range frame",
            width: 256,
            height: 160,
            numFrames: 90,
            frameInputs: [.init(frameIndex: 90, url: later)]
        ))
    }

    func testSlidingWindowPlanPreservesExactGlobalTimeline() throws {
        let options = try MiniMaxH3SlidingWindowOptions(
            totalFrameCount: 124,
            windowFrameCount: 39,
            overlapFrameCount: 18
        )
        let plan = MiniMaxH3SlidingWindowPlan(options: options)

        XCTAssertEqual(plan.windows.count, 6)
        XCTAssertEqual(plan.windows[0].generatedFrameCount, 39)
        XCTAssertEqual(plan.windows[0].appendedFrameRange, 0..<39)
        XCTAssertEqual(plan.windows[1].boundaryFrameIndex, 38)
        XCTAssertEqual(plan.windows[1].generatedFrameCount, 22)
        XCTAssertEqual(plan.windows[1].appendedFrameRange, 1..<22)
        XCTAssertEqual(plan.windows[1].outputFrameRange, 39..<60)
        XCTAssertEqual(plan.windows[2].localFrameIndex(for: 72), 13)
        XCTAssertEqual(plan.windows.last?.outputFrameRange, 123..<124)
        XCTAssertEqual(plan.windows.last?.appendedFrameRange, 1..<2)
        XCTAssertEqual(plan.windows.reduce(0) { $0 + $1.appendedFrameCount }, 124)
    }

    func testSlidingWindowOptionsRequireCompatibleTargetGeometry() {
        XCTAssertThrowsError(try MiniMaxH3SlidingWindowOptions(
            totalFrameCount: 124,
            windowFrameCount: 39,
            overlapFrameCount: 17
        ))
        XCTAssertThrowsError(try MiniMaxH3SlidingWindowOptions(
            totalFrameCount: 124,
            windowFrameCount: 39,
            overlapFrameCount: 35
        ))
        XCTAssertNoThrow(try MiniMaxH3SlidingWindowOptions(
            totalFrameCount: 243,
            windowFrameCount: 124,
            overlapFrameCount: 35
        ))
    }

    func testTransformerQKVUsesConvertedGlobalSlabs() {
        let projected = MLXArray((0..<12).map(Float.init)).reshaped(1, 1, 12)
        let parts = miniMaxH3SplitProjectedQKV(projected, heads: 2, headDimension: 2)
        MLX.eval(parts)

        XCTAssertEqual(parts[0].asArray(Float.self), [0, 1, 2, 3])
        XCTAssertEqual(parts[1].asArray(Float.self), [4, 5, 6, 7])
        XCTAssertEqual(parts[2].asArray(Float.self), [8, 9, 10, 11])
    }

    func testReleasedBF16TransformerQKVIsDeinterleavedAtLoadTime() {
        let raw = MLXArray(0..<12).reshaped(12, 1)
        let mapped = MiniMaxH3ModelLoader.releasedBF16TransformerWeight(
            key: "blocks.0.attn.qkv_proj.weight",
            value: raw,
            omitCachedAdaLNWeights: true,
            headCount: 2,
            headDimension: 2
        )
        MLX.eval(mapped.map(\.1))

        XCTAssertEqual(mapped.map(\.0), ["blocks.0.attn.qkv_proj.weight"])
        XCTAssertEqual(mapped[0].1.asArray(Int32.self), [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11])
        XCTAssertTrue(
            MiniMaxH3ModelLoader.releasedBF16TransformerWeight(
                key: "blocks.0.adaln_proj.linear.weight",
                value: raw,
                omitCachedAdaLNWeights: true
            ).isEmpty
        )
    }

    func testPinnedMLXArtifactConfigurationDecodes() throws {
        let data = Data(#"""
        {
          "model_type": "minimax_h3",
          "partition": "fl2va",
          "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
          "quantization": {"group_size": 64, "bits": 8, "mode": "affine"},
          "transformer": {
            "hidden_size": 5376,
            "num_layers": 50,
            "num_attention_heads": 56,
            "attention_head_dim": 128,
            "ffn_hidden_size": 14336,
            "latents_dim": 24,
            "audio_latents_dim": 32,
            "text_dim": 5120,
            "time_embed_dim": 2688,
            "rope_inv_freq_len": 16
          }
        }
        """#.utf8)
        let configuration = try JSONDecoder().decode(MiniMaxH3Configuration.self, from: data)
        XCTAssertEqual(configuration.task, "fl2va")
        XCTAssertEqual(configuration.quantization?.bits, 8)
        XCTAssertEqual(configuration.textEncoderQuantization?.bits, 8)
        XCTAssertEqual(configuration.timeEmbeddingHiddenSize, 5_376)
        XCTAssertEqual(configuration.timeEmbeddingDimension, 2_688)
        XCTAssertTrue(configuration.validationIssues().isEmpty)
    }

    func testMixedTransformerAndTextEncoderQuantizationDecodes() throws {
        let data = Data(#"""
        {
          "model_type": "minimax_h3",
          "partition": "fl2va",
          "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
          "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
          "text_encoder_quantization": {"group_size": 64, "bits": 8, "mode": "affine"},
          "transformer": {
            "hidden_size": 5376,
            "num_layers": 50,
            "num_attention_heads": 56,
            "attention_head_dim": 128,
            "ffn_hidden_size": 14336,
            "latents_dim": 24,
            "audio_latents_dim": 32,
            "text_dim": 5120,
            "time_embed_dim": 2688,
            "rope_inv_freq_len": 16
          }
        }
        """#.utf8)
        let configuration = try JSONDecoder().decode(MiniMaxH3Configuration.self, from: data)
        XCTAssertEqual(configuration.quantization?.bits, 4)
        XCTAssertEqual(configuration.textEncoderQuantization?.bits, 8)
        XCTAssertTrue(configuration.validationIssues().isEmpty)
    }

    func testManagedFL2VAProfileUsesOfficialSourceCompactQ4Artifact() throws {
        XCTAssertEqual(
            MiniMaxH3Resources.artifactRepository,
            "Sawfwair/MiniMax-H3-FL2VA-MLX-4bit"
        )
        XCTAssertEqual(
            MiniMaxH3Resources.sourceRepository,
            "MiniMaxAI/MiniMax-H3"
        )
        XCTAssertEqual(
            MiniMaxH3Resources.sourceRevision,
            "ec19cc6daf5d8add9417c18e86b6b58cc6c55027"
        )
        XCTAssertTrue(MiniMaxH3Resources.compactArtifactFiles.contains("adaln_cache.safetensors"))
        XCTAssertTrue(MiniMaxH3Resources.compactArtifactFiles.contains("SOURCE_MANIFEST.json"))
        XCTAssertTrue(MiniMaxH3Resources.compactArtifactFiles.contains("transformer.conversion.json"))
        XCTAssertTrue(MiniMaxH3Resources.compactArtifactFiles.contains("SHA256SUMS"))

        let manifest = MereRunModelManifest.template(
            for: .miniMaxH3FL2VAMLX,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertEqual(manifest.quantization?.bits, 4)
        XCTAssertEqual(manifest.quantization?.groupSize, 64)
        XCTAssertEqual(manifest.quantization?.scheme, "mlx-affine")
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(MiniMaxH3Resources.artifactRepository)@\(MiniMaxH3Resources.artifactRevision)"
        )

        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: MiniMaxH3Resources.fl2vaModelID))
        XCTAssertEqual(spec.hubFallback?.repoId, MiniMaxH3Resources.artifactRepository)
        XCTAssertEqual(spec.hubFallback?.revision, MiniMaxH3Resources.artifactRevision)
        XCTAssertEqual(spec.hubFallback?.patterns, MiniMaxH3Resources.compactArtifactFiles)
    }

    func testManagedBF16ProfileUsesPinnedExistingMLXArtifact() throws {
        XCTAssertEqual(
            MiniMaxH3Resources.bf16ArtifactRepository,
            "pipenetwork/MiniMax-H3-MLX-bf16"
        )
        XCTAssertEqual(
            MiniMaxH3Resources.bf16ArtifactRevision,
            "1486555759eed9e3037edf29f9e055a0713bab2f"
        )
        XCTAssertEqual(MiniMaxH3Resources.bf16ShardFilenames.count, 13)
        XCTAssertFalse(MiniMaxH3Resources.bf16SupportArtifactFiles.contains("transformer.safetensors"))
        XCTAssertFalse(MiniMaxH3Resources.bf16SupportArtifactFiles.contains("adaln_cache.safetensors"))

        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: MiniMaxH3Resources.fl2vaBF16ModelID)
        )
        XCTAssertEqual(spec.hubFallback?.repoId, MiniMaxH3Resources.artifactRepository)
        XCTAssertEqual(spec.hubFallback?.patterns, MiniMaxH3Resources.bf16SupportArtifactFiles)
        let transformer = try XCTUnwrap(spec.mountedHubFallbacks.first)
        XCTAssertEqual(transformer.destinationPath, MiniMaxH3Resources.bf16TransformerDirectory)
        XCTAssertEqual(transformer.hubFallback.repoId, MiniMaxH3Resources.bf16ArtifactRepository)
        XCTAssertEqual(transformer.hubFallback.revision, MiniMaxH3Resources.bf16ArtifactRevision)

        let manifest = MereRunModelManifest.template(
            for: .miniMaxH3FL2VABF16MLX,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertNil(manifest.quantization)
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(MiniMaxH3Resources.bf16ArtifactRepository)@\(MiniMaxH3Resources.bf16ArtifactRevision)"
        )
    }

    func testManagedRef2VAProfileUsesPinnedPublicEightBitArtifact() throws {
        XCTAssertEqual(
            MiniMaxH3Resources.ref2vaArtifactRepository,
            "Sawfwair/MiniMax-H3-Ref2VA-MLX-8bit"
        )
        XCTAssertEqual(
            MiniMaxH3Resources.ref2vaArtifactRevision,
            "61dc387ef1a7166425cdacd63c2340598dcc364f"
        )
        XCTAssertEqual(
            MiniMaxH3Resources.ref2vaConvertedSHA256,
            "234f22f69f8d40d6ed81cceed8259fa287f3c9417d40fba5274e3a7aa84e18a2"
        )
        XCTAssertEqual(
            MiniMaxH3Resources.ref2vaAdaLNCacheSourceIdentity,
            "sha256:234f22f69f8d40d6ed81cceed8259fa287f3c9417d40fba5274e3a7aa84e18a2"
        )
        XCTAssertEqual(MiniMaxH3Resources.ref2vaConvertedByteCount, 36_024_412_656)
        XCTAssertTrue(MiniMaxH3Resources.ref2vaArtifactFiles.contains("adaln_cache.safetensors"))
        XCTAssertTrue(MiniMaxH3Resources.ref2vaArtifactFiles.contains("SOURCE_MANIFEST.json"))
        XCTAssertTrue(MiniMaxH3Resources.ref2vaArtifactFiles.contains("transformer.conversion.json"))
        XCTAssertTrue(MiniMaxH3Resources.ref2vaArtifactFiles.contains("SHA256SUMS"))

        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: MiniMaxH3Resources.ref2vaModelID)
        )
        XCTAssertEqual(spec.hubFallback?.repoId, MiniMaxH3Resources.ref2vaArtifactRepository)
        XCTAssertEqual(spec.hubFallback?.revision, MiniMaxH3Resources.ref2vaArtifactRevision)
        XCTAssertEqual(spec.hubFallback?.patterns, MiniMaxH3Resources.ref2vaArtifactFiles)
        XCTAssertEqual(spec.estimatedDownloadBytes, 70_941_103_245)

        let manifest = MereRunModelManifest.template(
            for: .miniMaxH3Ref2VAMLX,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(manifest.precision, .int8)
        XCTAssertEqual(manifest.quantization?.bits, 8)
        XCTAssertEqual(manifest.quantization?.groupSize, 64)
        XCTAssertEqual(manifest.quantization?.scheme, "mlx-affine")
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(MiniMaxH3Resources.ref2vaArtifactRepository)@\(MiniMaxH3Resources.ref2vaArtifactRevision)"
        )
    }

    func testManagedRef2VAArtifactValidationPinsConversionReceipt() throws {
        let rootURL = FileManager.default.temporaryDirectory
            .appending(path: "minimax-h3-ref2va-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: rootURL) }

        try Data("{}".utf8).write(to: rootURL.appending(path: "SHA256SUMS"))
        let sourceManifestURL = rootURL.appending(path: "SOURCE_MANIFEST.json")
        let sourceManifest = """
        {
          "adaln_cache": {
            "byte_count": 873820740,
            "format": "mere.run.minimax-h3-adaln-cache",
            "path": "adaln_cache.safetensors",
            "schedule": {
              "audio_flow_shift": 3.0,
              "point_count": 31,
              "video_flow_shift": 12.0
            },
            "schema_version": 2,
            "sha256": "2cbe9e3324ef2cc5108a3ba7f1219d84079ff00a017f604fd86300005cc64fcd",
            "source_identity": "sha256:234f22f69f8d40d6ed81cceed8259fa287f3c9417d40fba5274e3a7aa84e18a2"
          },
          "artifact": {
            "format": "mere.run.minimax-h3-ref2va-mlx-8bit",
            "partition": "ref2va",
            "repository": "Sawfwair/MiniMax-H3-Ref2VA-MLX-8bit"
          },
          "schema_version": 1
        }
        """
        try Data(sourceManifest.utf8).write(to: sourceManifestURL)
        let receiptURL = rootURL.appending(path: "transformer.conversion.json")
        let receipt = """
        {
          "converter": "scripts/model-conversion/convert_minimax_h3_convrot.py",
          "converter_version": 2,
          "output": {
            "byte_count": 36024412656,
            "filename": "transformer.safetensors",
            "sha256": "234f22f69f8d40d6ed81cceed8259fa287f3c9417d40fba5274e3a7aa84e18a2"
          },
          "partition": "ref2va",
          "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
          "source": {
            "byte_count": 34038894550,
            "filename": "minimax_h3_ref2va_int8_convrot.safetensors",
            "repository": "Comfy-Org/MiniMax-H3",
            "revision": "fd70b39279d1ae6eb214c903f53e1bec3af19a77",
            "sha256": "9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9"
          },
          "source_convrot_groups": {"256": 200, "64": 50}
        }
        """
        try Data(receipt.utf8).write(to: receiptURL)

        let metadata = [
            "__metadata__": [
                "quantization": "affine 8-bit g64",
                "source_repository": MiniMaxH3Resources.conversionSourceRepository,
                "source_revision": MiniMaxH3Resources.conversionSourceRevision,
            ],
        ]
        let header = try JSONEncoder().encode(metadata)
        var headerLength = UInt64(header.count).littleEndian
        var transformer = withUnsafeBytes(of: &headerLength) { Data($0) }
        transformer.append(header)
        let transformerURL = rootURL.appending(path: "transformer.safetensors")
        try transformer.write(to: transformerURL)
        let handle = try FileHandle(forWritingTo: transformerURL)
        try handle.truncate(atOffset: UInt64(MiniMaxH3Resources.ref2vaConvertedByteCount))
        try handle.close()

        let cacheMetadata = [
            "__metadata__": [
                "format": "mere.run.minimax-h3-adaln-cache",
                "schema_version": MiniMaxH3AdaLNCache.schemaVersion,
                "source_identity": MiniMaxH3Resources.ref2vaAdaLNCacheSourceIdentity,
            ],
        ]
        let cacheHeader = try JSONEncoder().encode(cacheMetadata)
        var cacheHeaderLength = UInt64(cacheHeader.count).littleEndian
        var cacheFile = withUnsafeBytes(of: &cacheHeaderLength) { Data($0) }
        cacheFile.append(cacheHeader)
        let cacheURL = rootURL.appending(path: MiniMaxH3AdaLNCache.filename)
        try cacheFile.write(to: cacheURL)
        let cacheHandle = try FileHandle(forWritingTo: cacheURL)
        try cacheHandle.truncate(atOffset: UInt64(MiniMaxH3Resources.ref2vaAdaLNCacheByteCount))
        try cacheHandle.close()

        let resources = MiniMaxH3Resources(rootURL: rootURL)
        XCTAssertTrue(resources.validateManagedRef2VAArtifact().isEmpty)

        try Data(sourceManifest.replacingOccurrences(
            of: MiniMaxH3Resources.ref2vaAdaLNCacheSHA256,
            with: String(repeating: "0", count: 64)
        ).utf8).write(to: sourceManifestURL)
        XCTAssertTrue(resources.validateManagedRef2VAArtifact().contains { $0.contains("pinned AdaLN cache") })
        try Data(sourceManifest.utf8).write(to: sourceManifestURL)

        try Data(receipt.replacingOccurrences(of: "\"256\": 200", with: "\"256\": 199").utf8)
            .write(to: receiptURL)
        XCTAssertTrue(resources.validateManagedRef2VAArtifact().contains { $0.contains("source-group counts") })
    }

    func testManagedRef2VAArtifactWhenAvailable() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_H3_REF2VA_MODEL_ROOT"], !root.isEmpty else {
            throw XCTSkip("Set MERERUN_H3_REF2VA_MODEL_ROOT to validate the complete managed Ref2VA artifact.")
        }
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: MiniMaxH3Resources.ref2vaModelID)
        )
        let rootURL = URL(fileURLWithPath: root, isDirectory: true)
        let messages = spec.validationMessages(in: rootURL)
        XCTAssertTrue(messages.isEmpty, messages.joined(separator: "; "))
        XCTAssertTrue(spec.isManagedRootComplete(rootURL))
        XCTAssertTrue(spec.isManagedRuntimeReady(rootURL))
    }

    func testInstalledRef2VAExactKernelInventoryWhenAvailable() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_H3_EXACT_KERNEL_MODEL_ROOT"],
              !root.isEmpty else {
            throw XCTSkip(
                "Set MERERUN_H3_EXACT_KERNEL_MODEL_ROOT to inspect an installed Ref2VA transformer."
            )
        }
        let resources = MiniMaxH3Resources(
            rootURL: URL(fileURLWithPath: root, isDirectory: true)
        )
        let configuration = try resources.loadConfiguration()
        let transformer = try MiniMaxH3ModelLoader.loadInferenceTransformer(
            resources: resources,
            configuration: configuration,
            cachedAdaLN: nil
        )
        XCTAssertEqual(configuration.task, "ref2va")
        XCTAssertEqual(configuration.quantization?.bits, 8)
        XCTAssertEqual(configuration.quantization?.groupSize, 64)
        XCTAssertEqual(
            transformer.affineQ8ExactKernelBlockCount,
            configuration.layerCount
        )
        XCTAssertTrue(transformer.supportsAffineQ8ExactKernels)
    }

    func testInstalledRef2VAExactKernelFullForwardWhenEnabled() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_H3_EXACT_FULL_FORWARD"] == "1",
              let root = environment["MERERUN_H3_EXACT_KERNEL_MODEL_ROOT"],
              !root.isEmpty else {
            throw XCTSkip(
                "Set MERERUN_H3_EXACT_FULL_FORWARD=1 and "
                    + "MERERUN_H3_EXACT_KERNEL_MODEL_ROOT to run the real-weight 50-block gate."
            )
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The real-weight H3 exact-kernel gate requires a Metal GPU.")
        }
        let resources = MiniMaxH3Resources(
            rootURL: URL(fileURLWithPath: root, isDirectory: true)
        )
        let configuration = try resources.loadConfiguration()
        let transformer = try MiniMaxH3ModelLoader.loadInferenceTransformer(
            resources: resources,
            configuration: configuration,
            cachedAdaLN: nil
        )
        XCTAssertTrue(transformer.supportsAffineQ8ExactKernels)
        transformer.usesLayerwiseEvaluation = true
        transformer.clearsCacheAfterLayerwiseEvaluation = false

        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1],
            videoLatentFrames: 1,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 1,
            keyframeAnchors: []
        )
        MLXRandom.seed(2_026_081_013)
        let video = MLXRandom.uniform(
            -0.05 ..< 0.05,
            [1, layout.targetVideoRows.count, transformer.configuration.videoPatchDimension]
        ).asType(.bfloat16)
        let audio = MLXRandom.uniform(
            -0.05 ..< 0.05,
            [1, layout.targetAudioRows.count, configuration.audioLatentChannels]
        ).asType(.bfloat16)
        let text = MLXRandom.uniform(
            -0.05 ..< 0.05,
            [1, layout.textRows.count, configuration.textDimension]
        ).asType(.bfloat16)

        func metrics(_ value: MLXArray, _ reference: MLXArray) -> (Float, Float) {
            let difference = value.asType(.float32) - reference.asType(.float32)
            let maximum = MLX.max(MLX.abs(difference)).item(Float.self)
            let numerator = MLX.sum(difference * difference)
            let denominator = MLX.maximum(
                MLX.sum(reference.asType(.float32) * reference.asType(.float32)),
                MLXArray(Float(1e-12))
            )
            return (maximum, MLX.sqrt(numerator / denominator).item(Float.self))
        }

        transformer.exactKernelMode = .disabled
        let baseline = transformer(
            videoRows: video,
            audioRows: audio,
            textStates: text,
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(baseline.videoVelocityRows, baseline.audioVelocityRows)

        if environment["MERERUN_H3_EXACT_STAGE_DIAGNOSTICS"] == "1" {
            for stage in MiniMaxH3ExactKernelStage.allCases {
                transformer.enabledExactKernelStages = [stage]
                transformer.exactKernelMode = .affineQ8
                let stageCandidate = transformer(
                    videoRows: video,
                    audioRows: audio,
                    textStates: text,
                    layout: layout,
                    videoTimestep: 0.2,
                    audioTimestep: 0.4
                )
                MLX.eval(
                    stageCandidate.videoVelocityRows,
                    stageCandidate.audioVelocityRows
                )
                let stageVideo = metrics(
                    stageCandidate.videoVelocityRows,
                    baseline.videoVelocityRows
                )
                let stageAudio = metrics(
                    stageCandidate.audioVelocityRows,
                    baseline.audioVelocityRows
                )
                print(String(
                    format: "[h3-transfer] stage=%@ video_rel_l2=%.6g audio_rel_l2=%.6g",
                    stage.rawValue,
                    stageVideo.1,
                    stageAudio.1
                ))
            }
        }
        let affineStages = Set(MiniMaxH3ExactKernelStage.allCases).subtracting([.qkvLayout])
        transformer.enabledExactKernelStages = affineStages

        var dispatchCounts = Dictionary(
            uniqueKeysWithValues: MiniMaxH3ExactKernelStage.allCases.map { ($0, 0) }
        )
        var fallbackCounts: [String: Int] = [:]
        transformer.exactKernelDispatchHandler = { stage in
            dispatchCounts[stage, default: 0] += 1
        }
        transformer.exactKernelFallbackHandler = { stage, reason in
            fallbackCounts["\(stage.rawValue):\(reason)", default: 0] += 1
        }
        transformer.exactKernelMode = .affineQ8
        let candidate = transformer(
            videoRows: video,
            audioRows: audio,
            textStates: text,
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(candidate.videoVelocityRows, candidate.audioVelocityRows)

        let videoMetrics = metrics(candidate.videoVelocityRows, baseline.videoVelocityRows)
        let audioMetrics = metrics(candidate.audioVelocityRows, baseline.audioVelocityRows)
        let dispatchReceipt = MiniMaxH3ExactKernelStage.allCases.map { stage in
            "\(stage.rawValue)=\(dispatchCounts[stage, default: 0])"
        }.joined(separator: " ")
        let fallbackReceipt = fallbackCounts.sorted { $0.key < $1.key }.map { entry in
            "\(entry.key)=\(entry.value)"
        }.joined(separator: " | ")
        print(String(
            format: "[h3-transfer] real-weight-full-forward blocks=%d rows=%d "
                + "video_max_abs=%.6g video_rel_l2=%.6g "
                + "audio_max_abs=%.6g audio_rel_l2=%.6g %@ fallbacks=%@",
            transformer.affineQ8ExactKernelBlockCount,
            layout.sequenceLength,
            videoMetrics.0,
            videoMetrics.1,
            audioMetrics.0,
            audioMetrics.1,
            dispatchReceipt,
            fallbackReceipt
        ))

        for stage in MiniMaxH3ExactKernelStage.allCases {
            let expected = affineStages.contains(stage) ? configuration.layerCount : 0
            XCTAssertEqual(dispatchCounts[stage], expected, stage.rawValue)
        }
        XCTAssertTrue(videoMetrics.0.isFinite && videoMetrics.1.isFinite)
        XCTAssertTrue(audioMetrics.0.isFinite && audioMetrics.1.isFinite)
        XCTAssertLessThan(videoMetrics.1, 0.05)
        XCTAssertLessThan(audioMetrics.1, 0.05)

        dispatchCounts = Dictionary(
            uniqueKeysWithValues: MiniMaxH3ExactKernelStage.allCases.map { ($0, 0) }
        )
        fallbackCounts = [:]
        let boundaryStages: Set<MiniMaxH3ExactKernelStage> = [.gateAdaLN, .qkvLayout]
        transformer.enabledExactKernelStages = boundaryStages
        transformer.exactKernelMode = .boundaryLayout
        let boundaryCandidate = transformer(
            videoRows: video,
            audioRows: audio,
            textStates: text,
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(
            boundaryCandidate.videoVelocityRows,
            boundaryCandidate.audioVelocityRows
        )
        let boundaryVideoMetrics = metrics(
            boundaryCandidate.videoVelocityRows,
            baseline.videoVelocityRows
        )
        let boundaryAudioMetrics = metrics(
            boundaryCandidate.audioVelocityRows,
            baseline.audioVelocityRows
        )
        let boundaryDispatchReceipt = MiniMaxH3ExactKernelStage.allCases.map { stage in
            "\(stage.rawValue)=\(dispatchCounts[stage, default: 0])"
        }.joined(separator: " ")
        let boundaryFallbackReceipt = fallbackCounts.sorted { $0.key < $1.key }.map { entry in
            "\(entry.key)=\(entry.value)"
        }.joined(separator: " | ")
        print(String(
            format: "[h3-transfer] real-weight-boundary-layout blocks=%d rows=%d "
                + "video_max_abs=%.6g video_rel_l2=%.6g "
                + "audio_max_abs=%.6g audio_rel_l2=%.6g %@ fallbacks=%@",
            transformer.affineQ8ExactKernelBlockCount,
            layout.sequenceLength,
            boundaryVideoMetrics.0,
            boundaryVideoMetrics.1,
            boundaryAudioMetrics.0,
            boundaryAudioMetrics.1,
            boundaryDispatchReceipt,
            boundaryFallbackReceipt
        ))
        for stage in MiniMaxH3ExactKernelStage.allCases {
            let expected = boundaryStages.contains(stage) ? configuration.layerCount : 0
            XCTAssertEqual(dispatchCounts[stage], expected, stage.rawValue)
        }
        XCTAssertTrue(boundaryFallbackReceipt.isEmpty)
        XCTAssertLessThan(boundaryVideoMetrics.1, 0.05)
        XCTAssertLessThan(boundaryAudioMetrics.1, 0.05)
    }

    func testInstalledRef2VAAdaLNCacheMatchesLiveBranchWhenEnabled() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_H3_ADALN_CACHE_PARITY"] == "1",
              let root = environment["MERERUN_H3_EXACT_KERNEL_MODEL_ROOT"],
              !root.isEmpty else {
            throw XCTSkip(
                "Set MERERUN_H3_ADALN_CACHE_PARITY=1 and "
                    + "MERERUN_H3_EXACT_KERNEL_MODEL_ROOT to run the Ref2VA cache gate."
            )
        }
        let resources = MiniMaxH3Resources(
            rootURL: URL(fileURLWithPath: root, isDirectory: true)
        )
        let configuration = try resources.loadConfiguration()
        let transformerConfiguration = MiniMaxH3TransformerConfiguration(configuration)
        let videoSchedule = try MiniMaxH3Schedule(
            pointCount: configuration.sampleSteps,
            shift: configuration.videoFlowShift
        )
        let audioSchedule = try MiniMaxH3Schedule(
            pointCount: configuration.sampleSteps,
            shift: configuration.audioFlowShift
        )
        let sourceIdentity = try resources.adaLNCacheSourceIdentity()
        let transformer = try MiniMaxH3ModelLoader.loadInferenceTransformer(
            resources: resources,
            configuration: configuration,
            cachedAdaLN: nil
        )
        let freshCache = transformer.precomputeAdaLN(
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule,
            sourceIdentity: sourceIdentity
        )
        if environment["MERERUN_H3_WRITE_ADALN_CACHE"] == "1" {
            try freshCache.save(to: resources.adaLNCacheURL, replacing: true)
            print(
                "[h3-transfer] rebuilt Ref2VA AdaLN cache at "
                    + resources.adaLNCacheURL.path
            )
        }
        let cache = try MiniMaxH3AdaLNCache.load(
            from: resources.adaLNCacheURL,
            configuration: transformerConfiguration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule,
            sourceIdentity: sourceIdentity
        )
        transformer.usesLayerwiseEvaluation = true
        transformer.clearsCacheAfterLayerwiseEvaluation = false

        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1],
            videoLatentFrames: 1,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 1,
            keyframeAnchors: []
        )
        MLXRandom.seed(2_026_081_015)
        let video = MLXRandom.uniform(
            -0.05 ..< 0.05,
            [1, layout.targetVideoRows.count, transformerConfiguration.videoPatchDimension]
        ).asType(.bfloat16)
        let audio = MLXRandom.uniform(
            -0.05 ..< 0.05,
            [1, layout.targetAudioRows.count, configuration.audioLatentChannels]
        ).asType(.bfloat16)
        let text = MLXRandom.uniform(
            -0.05 ..< 0.05,
            [1, layout.textRows.count, configuration.textDimension]
        ).asType(.bfloat16)
        let context = transformer.prepare(textStates: text, layout: layout)
        let stepIndex = min(10, cache.stepCount - 1)
        let timesteps = MLXArray([
            videoSchedule.timesteps[stepIndex],
            audioSchedule.timesteps[stepIndex],
            max(videoSchedule.timesteps[stepIndex], 0.999),
        ])
        let directStep = transformer.precomputeAdaLNStep(timesteps: timesteps)
        let live = transformer(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: nil
        )
        let cached = transformer(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: cache.step(at: stepIndex)
        )
        let fresh = transformer(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: freshCache.step(at: stepIndex)
        )
        let direct = transformer(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: directStep
        )
        MLX.eval(
            live.videoVelocityRows,
            live.audioVelocityRows,
            cached.videoVelocityRows,
            cached.audioVelocityRows,
            fresh.videoVelocityRows,
            fresh.audioVelocityRows,
            direct.videoVelocityRows,
            direct.audioVelocityRows
        )
        let videoError = MLX.max(MLX.abs(
            cached.videoVelocityRows.asType(DType.float32)
                - live.videoVelocityRows.asType(DType.float32)
        )).item(Float.self)
        let audioError = MLX.max(MLX.abs(
            cached.audioVelocityRows.asType(DType.float32)
                - live.audioVelocityRows.asType(DType.float32)
        )).item(Float.self)
        let freshVideoError = MLX.max(MLX.abs(
            fresh.videoVelocityRows.asType(DType.float32)
                - live.videoVelocityRows.asType(DType.float32)
        )).item(Float.self)
        let freshAudioError = MLX.max(MLX.abs(
            fresh.audioVelocityRows.asType(DType.float32)
                - live.audioVelocityRows.asType(DType.float32)
        )).item(Float.self)
        let directVideoError = MLX.max(MLX.abs(
            direct.videoVelocityRows.asType(DType.float32)
                - live.videoVelocityRows.asType(DType.float32)
        )).item(Float.self)
        let directAudioError = MLX.max(MLX.abs(
            direct.audioVelocityRows.asType(DType.float32)
                - live.audioVelocityRows.asType(DType.float32)
        )).item(Float.self)
        let cacheBlockError = MLX.max(MLX.abs(
            cache.blockModulations[0][stepIndex].asType(DType.float32)
                - directStep.blockModulations[0].asType(DType.float32)
        )).item(Float.self)
        let active45 = MiniMaxH3LayerThinningPolicy(activeBlockCount: 45)
            .activeBlockIndices(blockModulations: cache.blockModulations)
        let active40 = MiniMaxH3LayerThinningPolicy(activeBlockCount: 40)
            .activeBlockIndices(blockModulations: cache.blockModulations)
        print(String(
            format: "[h3-transfer] ref2va-adaln-cache steps=%d step=%d "
                + "file_video_max_abs=%.6g file_audio_max_abs=%.6g "
                + "fresh_video_max_abs=%.6g fresh_audio_max_abs=%.6g "
                + "direct_video_max_abs=%.6g direct_audio_max_abs=%.6g "
                + "block0_cache_direct_max_abs=%.6g layers45=%d layers40=%d",
            cache.stepCount,
            stepIndex,
            videoError,
            audioError,
            freshVideoError,
            freshAudioError,
            directVideoError,
            directAudioError,
            cacheBlockError,
            active45.count,
            active40.count
        ))

        XCTAssertLessThanOrEqual(freshVideoError, 1e-5)
        XCTAssertLessThanOrEqual(freshAudioError, 1e-5)
        XCTAssertLessThanOrEqual(directVideoError, 1e-5)
        XCTAssertLessThanOrEqual(directAudioError, 1e-5)
        XCTAssertLessThanOrEqual(videoError, 1e-5)
        XCTAssertLessThanOrEqual(audioError, 1e-5)
        XCTAssertEqual(active45.count, 45)
        XCTAssertEqual(active40.count, 40)
        XCTAssertTrue([0, 1, 49].allSatisfy(active45.contains))
        XCTAssertTrue([0, 1, 49].allSatisfy(active40.contains))
    }

    func testCompactTransformerPreservesAdaLNCacheSourceIdentity() throws {
        let rootURL = FileManager.default.temporaryDirectory
            .appending(path: "minimax-h3-compact-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: rootURL) }
        let transformerURL = rootURL.appending(path: "transformer.safetensors")
        try MLX.save(
            arrays: ["probe": MLXArray.zeros([1])],
            metadata: [
                "adaln_cache_source_identity": "35248980296:1785757316960297728",
                "cache_covered_weights_omitted": "true",
            ],
            url: transformerURL
        )
        let resources = MiniMaxH3Resources(rootURL: rootURL)
        XCTAssertEqual(
            try resources.adaLNCacheSourceIdentity(),
            "35248980296:1785757316960297728"
        )
        XCTAssertEqual(
            try resources.transformerMetadata()["cache_covered_weights_omitted"],
            "true"
        )
        XCTAssertTrue(try resources.requiresAdaLNCache())
    }

    func testPinnedArtifactWeightMaps() {
        XCTAssertEqual(
            MiniMaxH3ModelLoader.conditionerWeightKey("model.layers.0.self_attn.q_proj.weight"),
            "textEncoder.encoder.layers.0.self_attn.q_proj.weight"
        )
        XCTAssertEqual(
            MiniMaxH3ModelLoader.conditionerWeightKey("visual.merger.linear_fc1.weight"),
            "visionTower.patch_merger.mlp_0.weight"
        )
        let convolution = MLXArray(0..<720).reshaped(2, 3, 4, 5, 6)
        let mappedConvolution = MiniMaxH3VideoVAE.mapCheckpointWeight(
            key: "encoder.conv_in.weight",
            value: convolution
        )
        XCTAssertEqual(mappedConvolution.first?.0, "encoder.conv_in.weight")
        XCTAssertEqual(mappedConvolution.first?.1.shape, [2, 4, 5, 6, 3])

        let fused = MLXArray(0..<6_144)
        let mappedQKV = MiniMaxH3VideoVAE.mapCheckpointWeight(
            key: "decoder.transformer_blocks.0.attn.to_qkv.weight",
            value: fused
        )
        XCTAssertEqual(mappedQKV.map(\.0), [
            "decoder.transformer_blocks.0.attn.to_qkv.weight",
        ])
        XCTAssertEqual(mappedQKV[0].1.shape, [6_144])
        let packedQKV = mappedQKV[0].1.asArray(Int32.self)
        XCTAssertEqual(packedQKV[0], 0)
        XCTAssertEqual(packedQKV[64], 192)
        XCTAssertEqual(packedQKV[2_047], 6_015)
        XCTAssertEqual(packedQKV[2_048], 64)
        XCTAssertEqual(packedQKV[2_112], 256)
        XCTAssertEqual(packedQKV[4_095], 6_079)
        XCTAssertEqual(packedQKV[4_096], 128)
        XCTAssertEqual(packedQKV[4_160], 320)
        XCTAssertEqual(packedQKV[6_143], 6_143)

        let audioEncoder = MiniMaxH3AudioVAE.mapConvertedWeight(
            key: "encoder.block.0.weight",
            value: MLXArray.zeros([64, 1, 7])
        )
        XCTAssertEqual(audioEncoder.first?.1.shape, [64, 7, 1])
        let audioInputBias = MiniMaxH3AudioVAE.mapConvertedWeight(
            key: "dec_in_proj.bias",
            value: MLXArray.zeros([2_048])
        )
        XCTAssertEqual(audioInputBias.map(\.0), ["dec_in_proj.bias"])
        XCTAssertEqual(audioInputBias.first?.1.shape, [2_048])
        let audioResidual = MiniMaxH3AudioVAE.mapConvertedWeight(
            key: "encoder.block.1.block.0.block.1.weight",
            value: MLXArray.zeros([64, 64, 7])
        )
        XCTAssertEqual(audioResidual.first?.0, "encoder.block.1.block.0.block.1.weight")
        XCTAssertEqual(audioResidual.first?.1.shape, [64, 7, 64])
        let audioDecoder = MiniMaxH3AudioVAE.mapConvertedWeight(
            key: "decoder.conv_pre.weight",
            value: MLXArray.zeros([4, 2, 7])
        )
        XCTAssertEqual(audioDecoder.map(\.0), [
            "decoder.conv_pre.parametrizations.weight.original0",
            "decoder.conv_pre.parametrizations.weight.original1",
        ])

        let audioUpsample = MiniMaxH3AudioVAE.mapConvertedWeight(
            key: "decoder.ups.0.0.weight",
            value: MLXArray.zeros([1_024, 512, 9])
        )
        XCTAssertEqual(audioUpsample.map(\.0), [
            "decoder.ups.0.convolution.parametrizations.weight.original0",
            "decoder.ups.0.convolution.parametrizations.weight.original1",
        ])
        XCTAssertEqual(audioUpsample[0].1.shape, [1_024, 1, 1])
        XCTAssertEqual(audioUpsample[1].1.shape, [512, 9, 1_024])
    }

    func testReleasedTemporalGeometry() throws {
        XCTAssertEqual(try MiniMaxH3Geometry.alignFrameCount(120), 124)
        XCTAssertEqual(try MiniMaxH3Geometry.videoLatentFrameCount(for: 124), 37)
        XCTAssertEqual(MiniMaxH3Geometry.audioLatentFrameCount(for: 124), 207)
        XCTAssertThrowsError(try MiniMaxH3Geometry.videoLatentFrameCount(for: 120))
    }

    func testFL2VAPackedLayoutRangesAndTags() throws {
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1, 0, 1],
            videoLatentFrames: 7,
            latentHeight: 4,
            latentWidth: 6,
            audioLatentFrames: 5,
            keyframeAnchors: [.first, .last]
        )
        XCTAssertEqual(layout.textRows, 0..<3)
        XCTAssertEqual(layout.conditionVideoRowCount, 12)
        XCTAssertEqual(layout.targetAudioRows.count, 10)
        XCTAssertEqual(layout.targetVideoRows.count, 42)
        XCTAssertEqual(layout.sequenceLength, 67)
        XCTAssertEqual(layout.positions.shape, [67, 3])
        XCTAssertEqual(layout.tokenTags[0..<3], [1, 0, 1])
        XCTAssertTrue(layout.tokenTags[3..<15].allSatisfy { $0 == MiniMaxH3Modality.video.rawValue })
    }

    func testKeyframeAnchorPreservesStringAndCodableCompatibility() throws {
        let anchors: [MiniMaxH3KeyframeAnchor] = [
            .first,
            .last,
            .history(latentFrameCount: 6),
            .frame(11),
        ]
        XCTAssertEqual(anchors.map(\.rawValue), ["first", "last", "history:6", "frame:11"])
        XCTAssertEqual(anchors.compactMap { MiniMaxH3KeyframeAnchor(rawValue: $0.rawValue) }, anchors)

        let encoded = try JSONEncoder().encode(anchors)
        XCTAssertEqual(try JSONDecoder().decode([MiniMaxH3KeyframeAnchor].self, from: encoded), anchors)
        XCTAssertEqual(
            try JSONDecoder().decode(MiniMaxH3KeyframeAnchor.self, from: Data("\"first\"".utf8)),
            .first
        )
    }

    func testFL2VAHistoryFrameAndAudioConditionsShareShiftedTargetTimeline() throws {
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1, 1],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3,
            keyframeAnchors: [
                .history(latentFrameCount: 2),
                .first,
                .frame(6),
                .last,
            ],
            audioConditionAnchors: [
                .history(latentFrameCount: 3),
                .first(latentFrameCount: 2),
            ]
        )
        XCTAssertEqual(layout.textRows, 0..<2)
        XCTAssertEqual(layout.conditionRows, 2..<32)
        XCTAssertEqual(layout.conditionVideoRowCount, 20)
        XCTAssertEqual(layout.conditionAudioRowCount, 10)
        XCTAssertEqual(layout.targetAudioRows, 32..<38)
        XCTAssertEqual(layout.targetVideoRows, 38..<46)
        XCTAssertEqual(layout.conditionSegments.map(\.modality), [.video, .audio])
        XCTAssertEqual(layout.conditionSegments.map(\.sourceRows), [0..<20, 0..<10])

        let positions = layout.positions.asArray(Float.self)
        func time(at row: Int) -> Float { positions[row * 3] }
        XCTAssertEqual(time(at: 2), 2, accuracy: 1e-5)
        XCTAssertEqual(time(at: 6), 2 + 5.0 / 3.0, accuracy: 1e-5)
        XCTAssertEqual(time(at: 10), 2 + 25.0 / 3.0, accuracy: 1e-5)
        XCTAssertEqual(time(at: 14), 2 + 55.0 / 3.0, accuracy: 1e-5)
        XCTAssertEqual(time(at: 18), 17, accuracy: 1e-5)
        XCTAssertEqual(time(at: 22), 2, accuracy: 1e-5)
        XCTAssertEqual(time(at: 28), 2 + 25.0 / 3.0, accuracy: 1e-5)
        XCTAssertEqual(time(at: layout.targetAudioRows.lowerBound), 2 + 25.0 / 3.0, accuracy: 1e-5)
        XCTAssertEqual(time(at: layout.targetVideoRows.lowerBound), 2 + 25.0 / 3.0, accuracy: 1e-5)
    }

    func testVideoAndAudioPackingRoundTrip() {
        let video = MLXArray(0..<384).asType(.float32).reshaped(1, 3, 2, 8, 8)
        let rows = MiniMaxH3Geometry.patchifyVideo(video)
        let roundTrip = MiniMaxH3Geometry.unpatchifyVideo(
            rows,
            frames: 2,
            height: 8,
            width: 8,
            channels: 3
        )
        MLX.eval(roundTrip)
        XCTAssertEqual(roundTrip.shape, video.shape)
        XCTAssertEqual(roundTrip.asArray(Float.self), video.asArray(Float.self))

        let audio = MLXArray(0..<40).asType(.float32).reshaped(1, 4, 2, 5)
        let audioRows = MiniMaxH3Geometry.packAudio(audio)
        let audioRoundTrip = MiniMaxH3Geometry.unpackAudio(audioRows)
        MLX.eval(audioRoundTrip)
        XCTAssertEqual(audioRoundTrip.shape, audio.shape)
        XCTAssertEqual(audioRoundTrip.asArray(Float.self), audio.asArray(Float.self))
    }

    func testInstalledAudioVAEDecodeMatchesReference() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_H3_MODEL_ROOT"], !root.isEmpty else {
            throw XCTSkip(
                "Set MERERUN_H3_MODEL_ROOT to run the checkpoint-backed H3 audio parity fixture."
            )
        }

        let latentFrames = 8
        let latentValues = (0..<(32 * 2 * latentFrames)).map { index in
            Float((index * 37) % 257 - 128) / 64
        }
        let latent = MLXArray(latentValues).reshaped(1, 32, 2, latentFrames)
        let resources = MiniMaxH3Resources(rootURL: URL(fileURLWithPath: root, isDirectory: true))
        let waveform = try MiniMaxH3ModelLoader.loadAudioVAE(resources: resources).decode(latent)
        MLX.eval(waveform)

        XCTAssertEqual(waveform.shape, [1, latentFrames * MiniMaxH3AudioVAE.hopLength, 2])
        let values = waveform.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy(\.isFinite))
        // FP32 samples from ComfyUI's MiniMaxH3AudioVAE at
        // 16e3f3034f2bba1fff6c70cbd759339778555cd6 for the latent fixture above.
        let referenceSamples: [(Int, Float)] = [
            (0, 0.044_520_34), (1, 0.011_212_92), (2, -0.013_318_60),
            (63, -0.142_264_11), (255, 0.202_620_71), (511, -0.673_421_03),
            (1_023, 0.361_299_25), (2_047, -0.206_609_98),
            (4_095, -0.600_924_31), (6_143, 0.713_577_15),
            (8_191, 0.200_860_02), (10_239, 0.757_872_64),
            (12_287, -0.011_021_69), (12_799, 0.009_704_24),
        ]
        for (index, expected) in referenceSamples {
            XCTAssertEqual(values[index], expected, accuracy: 0.000_1, "reference sample \(index)")
        }
        let rootMeanSquare = sqrt(values.reduce(0) { $0 + $1 * $1 } / Float(values.count))
        XCTAssertEqual(rootMeanSquare, 0.556_741_18, accuracy: 0.000_1)
    }

    func testDecodedFramesConvertToMediaPixels() {
        let decoded = MLXArray([Float(0), 0.5, 1]).reshaped(1, 1, 1, 1, 3)
        let pixels = MiniMaxH3Generator.mediaFrames(from: decoded)
        MLX.eval(pixels)
        XCTAssertEqual(pixels.dtype, .uint8)
        XCTAssertEqual(pixels.shape, decoded.shape)
        XCTAssertEqual(pixels.asArray(UInt8.self), [0, 127, 255])
    }

    func testH3FrameScalerMatchesSolidColorAndRequestedShape() throws {
        let source = MLXArray(
            Array(repeating: [UInt8(10), 20, 30], count: 4).flatMap { $0 }
        ).reshaped(1, 1, 2, 2, 3)
        let scaled = try MiniMaxH3FrameScaler.scaled(source, width: 4, height: 4)
        MLX.eval(scaled)

        XCTAssertEqual(scaled.dtype, .uint8)
        XCTAssertEqual(scaled.shape, [1, 1, 4, 4, 3])
        XCTAssertEqual(
            scaled.asArray(UInt8.self),
            Array(repeating: [UInt8(10), 20, 30], count: 16).flatMap { $0 }
        )
    }

    func testVideoVAETilePlansPreserveCanvasAndMinimumOverlap() {
        XCTAssertEqual(MiniMaxH3VideoVAE.defaultSpatialTileSize, 256)
        for tileSize in [256, 304, 320] {
            for length in [480, 832, 1_344] {
                let plan = MiniMaxH3VideoVAE.tilePlan(length: length, tileSize: tileSize)
                XCTAssertEqual(plan.starts.first, 0)
                XCTAssertEqual(plan.starts.last! + plan.lengths.last!, length)
                XCTAssertTrue(plan.lengths.allSatisfy { $0 == tileSize || $0 == length })
                XCTAssertTrue(plan.overlaps.allSatisfy {
                    $0 >= MiniMaxH3VideoVAE.minimumSpatialTileOverlap
                })
            }
        }
    }

    func testRef2VAPackedLayoutPreservesOrderedReferenceBlocks() throws {
        let layout = try MiniMaxH3Geometry.buildRef2VA(
            textTokenTags: [1, 0],
            references: [
                .init(kind: .image, videoLatentFrames: 1, latentHeight: 4, latentWidth: 4),
                .init(
                    kind: .video,
                    videoLatentFrames: 2,
                    latentHeight: 4,
                    latentWidth: 6,
                    audioLatentFrames: 3
                ),
                .init(kind: .audio, audioLatentFrames: 2),
            ],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 4
        )
        XCTAssertEqual(layout.conditionVideoRowCount, 16)
        XCTAssertEqual(layout.conditionAudioRowCount, 10)
        XCTAssertEqual(layout.conditionRows, 2..<28)
        XCTAssertEqual(layout.targetAudioRows, 28..<36)
        XCTAssertEqual(layout.targetVideoRows, 36..<44)
        XCTAssertEqual(layout.conditionSegments.map(\.modality), [.video, .audio, .video, .audio])
        XCTAssertEqual(layout.conditionSegments.map(\.packedRows), [2..<6, 6..<12, 12..<24, 24..<28])
        XCTAssertEqual(layout.conditionSegments.map(\.sourceRows), [0..<4, 0..<6, 4..<16, 6..<10])
        XCTAssertTrue(layout.tokenTags[6..<12].allSatisfy { $0 == MiniMaxH3Modality.audio.rawValue })
        XCTAssertTrue(layout.tokenTags[12..<24].allSatisfy { $0 == MiniMaxH3Modality.video.rawValue })
    }

    func testRef2VAContinuationPrecedesReferencesAndShiftsTarget() throws {
        let layout = try MiniMaxH3Geometry.buildRef2VA(
            textTokenTags: [1, 0],
            references: [
                .init(kind: .image, videoLatentFrames: 1, latentHeight: 4, latentWidth: 4),
                .init(
                    kind: .video,
                    videoLatentFrames: 2,
                    latentHeight: 4,
                    latentWidth: 6,
                    audioLatentFrames: 3
                ),
                .init(kind: .audio, audioLatentFrames: 2),
            ],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 4,
            keyframeAnchors: [.history(latentFrameCount: 2), .first],
            audioConditionAnchors: [
                .history(latentFrameCount: 3),
                .first(latentFrameCount: 2),
            ]
        )

        XCTAssertEqual(layout.conditionVideoRowCount, 28)
        XCTAssertEqual(layout.conditionAudioRowCount, 20)
        XCTAssertEqual(layout.conditionRows, 2..<50)
        XCTAssertEqual(layout.targetAudioRows, 50..<58)
        XCTAssertEqual(layout.targetVideoRows, 58..<66)
        XCTAssertEqual(
            layout.conditionSegments.map(\.modality),
            [.video, .audio, .video, .audio, .video, .audio]
        )
        XCTAssertEqual(layout.conditionSegments.map(\.packedRows), [
            2..<14,
            14..<24,
            24..<28,
            28..<34,
            34..<46,
            46..<50,
        ])
        XCTAssertEqual(layout.conditionSegments.map(\.sourceRows), [
            0..<12,
            0..<10,
            12..<16,
            10..<16,
            16..<28,
            16..<20,
        ])

        let positions = layout.positions.asArray(Float.self)
        func time(at row: Int) -> Float { positions[row * 3] }
        let referenceEnd = Float(40.0 / 3.0)
        let targetOrigin = Float(65.0 / 3.0)
        XCTAssertEqual(time(at: 2), referenceEnd, accuracy: 1e-5)
        XCTAssertEqual(time(at: 10), targetOrigin, accuracy: 1e-5)
        XCTAssertEqual(time(at: 14), referenceEnd, accuracy: 1e-5)
        XCTAssertEqual(time(at: 20), targetOrigin, accuracy: 1e-5)
        XCTAssertEqual(time(at: layout.targetAudioRows.lowerBound), targetOrigin, accuracy: 1e-5)
        XCTAssertEqual(time(at: layout.targetVideoRows.lowerBound), targetOrigin, accuracy: 1e-5)
    }

    func testShiftedSchedulesTerminateAtCleanEndpoint() throws {
        let video = try MiniMaxH3Schedule(pointCount: 5, shift: 12)
        let audio = try MiniMaxH3Schedule(pointCount: 5, shift: 3)
        XCTAssertEqual(video.sigmas.first, 1)
        XCTAssertEqual(video.sigmas.last, 0)
        XCTAssertEqual(audio.sigmas.first, 1)
        XCTAssertEqual(audio.sigmas.last, 0)
        XCTAssertEqual(video.timesteps.count, 4)
        XCTAssertEqual(video.timesteps.first, 0)

        let sample = MLXArray([1, 2, 3, 4]).reshaped(1, 4)
        let velocity = MLXArray.ones([1, 4])
        let final = video.step(sample: sample, velocity: velocity, index: video.timesteps.count - 1)
        MLX.eval(final)
        XCTAssertTrue(final.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testBlockReusePolicyBoundsCacheStepsForPracticalSchedule() throws {
        XCTAssertNil(MiniMaxH3AccelerationMode.quality.blockReusePolicy)
        XCTAssertEqual(
            MiniMaxH3BlockReusePolicy(cacheDepth: 0.5).warmBlockCount(totalBlockCount: 50),
            25
        )
        let policy = try XCTUnwrap(MiniMaxH3AccelerationMode.maximum.blockReusePolicy)
        XCTAssertEqual(policy.warmBlockCount(totalBlockCount: 50), 9)
        XCTAssertEqual(policy.maximumConsecutiveCachedSteps, 4)

        let video = try MiniMaxH3Schedule(pointCount: 9, shift: 12)
        let audio = try MiniMaxH3Schedule(pointCount: 9, shift: 3)
        var hasResidual = false
        var consecutive = 0
        var decisions: [Bool] = []
        for index in video.timesteps.indices {
            let reuses = policy.shouldReuseTail(
                stepIndex: index,
                stepCount: video.timesteps.count,
                videoSigmas: video.sigmas,
                audioSigmas: audio.sigmas,
                hasCachedResidual: hasResidual,
                consecutiveCachedSteps: consecutive
            )
            decisions.append(reuses)
            if reuses {
                consecutive += 1
            } else {
                hasResidual = true
                consecutive = 0
            }
        }
        XCTAssertEqual(decisions, [false, false, true, true, true, true, false, false])

        let longVideo = try MiniMaxH3Schedule(pointCount: 16, shift: 12)
        let longAudio = try MiniMaxH3Schedule(pointCount: 16, shift: 3)
        hasResidual = false
        consecutive = 0
        var longCachedSteps = 0
        for index in longVideo.timesteps.indices {
            let reuses = policy.shouldReuseTail(
                stepIndex: index,
                stepCount: longVideo.timesteps.count,
                videoSigmas: longVideo.sigmas,
                audioSigmas: longAudio.sigmas,
                hasCachedResidual: hasResidual,
                consecutiveCachedSteps: consecutive
            )
            if reuses {
                longCachedSteps += 1
                consecutive += 1
            } else {
                hasResidual = true
                consecutive = 0
            }
        }
        XCTAssertEqual(longCachedSteps, 10)
    }

    func testAdaptiveFirstBlockCachePolicyUsesSafeModalityGuards() throws {
        XCTAssertNil(MiniMaxH3AccelerationMode.quality.adaptiveFirstBlockCachePolicy)
        let balanced = try XCTUnwrap(
            MiniMaxH3AccelerationMode.balanced.adaptiveFirstBlockCachePolicy
        )
        XCTAssertEqual(balanced.globalThreshold, 0.08)
        XCTAssertEqual(balanced.temporalThreshold, 0.12)
        XCTAssertEqual(balanced.maximumConsecutiveCachedSteps, 2)
        XCTAssertEqual(balanced.requiredFinalFullSteps, 2)
        let maximum = try XCTUnwrap(
            MiniMaxH3AccelerationMode.maximum.adaptiveFirstBlockCachePolicy
        )
        XCTAssertEqual(maximum.globalThreshold, 0.30)
        XCTAssertEqual(maximum.temporalThreshold, 0.40)
        XCTAssertEqual(maximum.maximumConsecutiveCachedSteps, 4)
        XCTAssertEqual(maximum.requiredFinalFullSteps, 1)

        XCTAssertFalse(balanced.canConsiderReuse(
            stepIndex: 1,
            stepCount: 8,
            fullStepCount: 1,
            consecutiveCachedSteps: 0,
            hasCachedState: true
        ))
        XCTAssertTrue(balanced.canConsiderReuse(
            stepIndex: 2,
            stepCount: 8,
            fullStepCount: 2,
            consecutiveCachedSteps: 0,
            hasCachedState: true
        ))
        XCTAssertFalse(balanced.canConsiderReuse(
            stepIndex: 6,
            stepCount: 8,
            fullStepCount: 2,
            consecutiveCachedSteps: 0,
            hasCachedState: true
        ))
        XCTAssertFalse(balanced.canConsiderReuse(
            stepIndex: 3,
            stepCount: 8,
            fullStepCount: 2,
            consecutiveCachedSteps: 2,
            hasCachedState: true
        ))

        XCTAssertTrue(balanced.shouldReuse(change: .init(
            videoGlobal: 0.05,
            audioGlobal: 0.03,
            videoTemporalMaximum: 0.10,
            audioTemporalMaximum: 0.08
        )))
        XCTAssertFalse(balanced.shouldReuse(change: .init(
            videoGlobal: 0.05,
            audioGlobal: 0.03,
            videoTemporalMaximum: 0.13,
            audioTemporalMaximum: 0.08
        )))
        XCTAssertFalse(balanced.shouldReuse(change: .init(
            videoGlobal: 0.05,
            audioGlobal: .nan,
            videoTemporalMaximum: 0.10,
            audioTemporalMaximum: 0.08
        )))

        var fullSteps = 0
        var consecutiveCachedSteps = 0
        var cachedSteps = 0
        for stepIndex in 0..<19 {
            let canReuse = maximum.canConsiderReuse(
                stepIndex: stepIndex,
                stepCount: 19,
                fullStepCount: fullSteps,
                consecutiveCachedSteps: consecutiveCachedSteps,
                hasCachedState: fullSteps > 0
            )
            if canReuse {
                cachedSteps += 1
                consecutiveCachedSteps += 1
            } else {
                fullSteps += 1
                consecutiveCachedSteps = 0
            }
        }
        XCTAssertEqual(cachedSteps, 13)
        XCTAssertEqual(fullSteps, 6)
        XCTAssertEqual(cachedSteps + fullSteps, 19)
        XCTAssertEqual(cachedSteps + fullSteps * 50, 313)
    }

    func testFirstBlockChangeSeparatesGlobalAndTemporalAudioVideoDrift() throws {
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [MiniMaxH3Modality.text.rawValue, MiniMaxH3Modality.text.rawValue],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3,
            keyframeAnchors: []
        )
        let targetRowCount = layout.targetAudioRows.count + layout.targetVideoRows.count
        let hiddenSize = 2
        let previousValues = [Float](repeating: 1, count: targetRowCount * hiddenSize)
        var currentValues = previousValues

        for row in 0..<layout.targetAudioRows.count {
            for hidden in 0..<hiddenSize {
                currentValues[row * hiddenSize + hidden] = 1.02
            }
        }
        for row in layout.targetAudioRows.count..<targetRowCount {
            for hidden in 0..<hiddenSize {
                currentValues[row * hiddenSize + hidden] = 1.05
            }
        }
        let uniform = MiniMaxH3FirstBlockChange.measure(
            current: MLXArray(currentValues, [1, targetRowCount, hiddenSize]),
            previous: MLXArray(previousValues, [1, targetRowCount, hiddenSize]),
            layout: layout
        )
        XCTAssertEqual(uniform.videoGlobal, 0.05, accuracy: 1e-5)
        XCTAssertEqual(uniform.audioGlobal, 0.02, accuracy: 1e-5)
        XCTAssertEqual(uniform.videoTemporalMaximum, 0.05, accuracy: 1e-5)
        XCTAssertEqual(uniform.audioTemporalMaximum, 0.02, accuracy: 1e-5)

        currentValues = previousValues
        let audioFrames = layout.audioLatentFrames
        for row in [1, audioFrames + 1] {
            for hidden in 0..<hiddenSize {
                currentValues[row * hiddenSize + hidden] = 1.3
            }
        }
        let videoStart = layout.targetAudioRows.count
        let videoRowsPerFrame = layout.targetVideoRows.count / layout.videoLatentFrames
        for row in videoStart..<(videoStart + videoRowsPerFrame) {
            for hidden in 0..<hiddenSize {
                currentValues[row * hiddenSize + hidden] = 1.2
            }
        }
        let localized = MiniMaxH3FirstBlockChange.measure(
            current: MLXArray(currentValues, [1, targetRowCount, hiddenSize]),
            previous: MLXArray(previousValues, [1, targetRowCount, hiddenSize]),
            layout: layout
        )
        XCTAssertEqual(localized.videoGlobal, 0.10, accuracy: 1e-5)
        XCTAssertEqual(localized.videoTemporalMaximum, 0.20, accuracy: 1e-5)
        XCTAssertEqual(localized.audioGlobal, 0.10, accuracy: 1e-5)
        XCTAssertEqual(localized.audioTemporalMaximum, 0.30, accuracy: 1e-5)
    }

    func testResidentBF16PolicyAccountsForGeometryAndMemory() throws {
        let denseBytes = 41 * MiniMaxH3ResidentBF16Policy.gibibyte
        XCTAssertTrue(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .automatic,
            physicalMemoryBytes: 64 * MiniMaxH3ResidentBF16Policy.gibibyte,
            estimatedResidentBytes: denseBytes,
            sequenceLength: 12_925,
            hasAdaLNCache: true,
            isPortableMac: false
        ))
        XCTAssertFalse(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .automatic,
            physicalMemoryBytes: 64 * MiniMaxH3ResidentBF16Policy.gibibyte,
            estimatedResidentBytes: denseBytes,
            sequenceLength: 37_966,
            hasAdaLNCache: true,
            isPortableMac: false
        ))
        XCTAssertTrue(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .automatic,
            physicalMemoryBytes: 128 * MiniMaxH3ResidentBF16Policy.gibibyte,
            estimatedResidentBytes: denseBytes,
            sequenceLength: 37_966,
            hasAdaLNCache: true,
            isPortableMac: false
        ))
        XCTAssertFalse(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .quantized,
            physicalMemoryBytes: 128 * MiniMaxH3ResidentBF16Policy.gibibyte,
            estimatedResidentBytes: denseBytes,
            sequenceLength: 12_925,
            hasAdaLNCache: true,
            isPortableMac: false
        ))
        XCTAssertFalse(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .automatic,
            physicalMemoryBytes: 128 * MiniMaxH3ResidentBF16Policy.gibibyte,
            estimatedResidentBytes: denseBytes,
            sequenceLength: 12_925,
            hasAdaLNCache: false,
            isPortableMac: false
        ))
        XCTAssertTrue(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .automatic,
            physicalMemoryBytes: 128 * MiniMaxH3ResidentBF16Policy.gibibyte,
            estimatedResidentBytes: denseBytes,
            sequenceLength: 12_925,
            hasAdaLNCache: true,
            isPortableMac: true
        ))
        XCTAssertFalse(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .automatic,
            physicalMemoryBytes: 64 * MiniMaxH3ResidentBF16Policy.gibibyte,
            estimatedResidentBytes: denseBytes,
            sequenceLength: 12_925,
            hasAdaLNCache: true,
            isPortableMac: true
        ))
        XCTAssertThrowsError(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .residentBF16,
            physicalMemoryBytes: 48 * MiniMaxH3ResidentBF16Policy.gibibyte,
            estimatedResidentBytes: denseBytes,
            sequenceLength: 12_925,
            hasAdaLNCache: true,
            isPortableMac: true
        ))
    }

    func testDenoiseExecutionPolicyKeepsProfilingOutsideCompiledTransforms() {
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.mode(
                usesResidentBF16: false,
                sequenceLength: 11_925,
                usesBlockProfiling: false
            ),
            .compiledStep
        )
        XCTAssertFalse(MiniMaxH3DenoiseExecutionMode.compiledStep.usesLayerwiseEvaluation)
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.mode(
                usesResidentBF16: false,
                sequenceLength: 11_925,
                usesBlockProfiling: true
            ),
            .eagerStep
        )
        XCTAssertTrue(MiniMaxH3DenoiseExecutionMode.eagerStep.usesLayerwiseEvaluation)
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.mode(
                usesResidentBF16: true,
                sequenceLength: 11_925,
                usesBlockProfiling: false,
                denoiseStepCount: 1
            ),
            .eagerStep
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.mode(
                usesResidentBF16: true,
                sequenceLength: 11_925,
                usesBlockProfiling: false,
                denoiseStepCount: 2
            ),
            .compiledStep
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.mode(
                usesResidentBF16: true,
                sequenceLength: 11_925,
                usesBlockProfiling: false,
                profilingOverride: "compiled"
            ),
            .compiledStep
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.mode(
                usesResidentBF16: false,
                sequenceLength: 11_925,
                usesBlockProfiling: true,
                profilingOverride: "compiled"
            ),
            .eagerStep
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.mode(
                usesResidentBF16: true,
                sequenceLength: MiniMaxH3DenoiseExecutionPolicy.blockwiseSequenceThreshold + 1,
                usesBlockProfiling: true
            ),
            .blockwiseCompiled
        )
        XCTAssertFalse(MiniMaxH3DenoiseExecutionMode.blockwiseCompiled.usesLayerwiseEvaluation)
    }

    func testDenoiseExecutionPolicyUsesMeasuredBlockwiseKernelSchedule() {
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 12_930)
                .maximumQueryTokens,
            640
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 12_930)
                .maximumKernelsPerEvaluation,
            1
        )
        XCTAssertNil(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 12_930)
                .maximumHeadsPerKernel
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 14_958)
                .maximumQueryTokens,
            1_024
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 14_958)
                .maximumKernelsPerEvaluation,
            1
        )
        XCTAssertNil(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 14_958)
                .maximumHeadsPerKernel
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 37_966)
                .maximumQueryTokens,
            768
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 37_966)
                .maximumKernelsPerEvaluation,
            1
        )
        XCTAssertNil(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 37_966)
                .maximumHeadsPerKernel
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 73_470)
                .maximumQueryTokens,
            640
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 73_470)
                .maximumHeadsPerKernel,
            8
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 73_470)
                .maximumKernelsPerEvaluation,
            1
        )
    }

    func testStepPolicyUsesPracticalExtendedAndMaximumEnvelopes() throws {
        XCTAssertEqual(
            try MiniMaxH3StepPolicy.recommendedPointCount(
                width: 512,
                height: 512,
                numFrames: 56
            ),
            9
        )
        XCTAssertEqual(
            try MiniMaxH3StepPolicy.recommendedPointCount(
                width: 768,
                height: 448,
                numFrames: 124
            ),
            9
        )
        XCTAssertEqual(
            try MiniMaxH3StepPolicy.recommendedPointCount(
                width: 768,
                height: 512,
                numFrames: 124
            ),
            16
        )
        XCTAssertEqual(
            try MiniMaxH3StepPolicy.recommendedPointCount(
                width: 1_344,
                height: 768,
                numFrames: 124
            ),
            21
        )
        XCTAssertEqual(
            try MiniMaxH3StepPolicy.recommendedPointCount(
                width: 1_344,
                height: 768,
                numFrames: 124,
                accelerationMode: .maximum
            ),
            12
        )
        XCTAssertEqual(
            try MiniMaxH3StepPolicy.recommendedPointCount(
                width: 768,
                height: 448,
                numFrames: 124,
                referenceKinds: [.video]
            ),
            16
        )

        let automatic = try MiniMaxH3GenerationOptions(
            prompt: "a practical local video",
            width: 768,
            height: 448,
            numFrames: 124
        )
        XCTAssertEqual(automatic.steps, 9)
        let explicit = try MiniMaxH3GenerationOptions(
            prompt: "a maximum quality local video",
            width: 768,
            height: 448,
            numFrames: 124,
            steps: 31
        )
        XCTAssertEqual(explicit.steps, 31)
        let longClip = try MiniMaxH3GenerationOptions(
            prompt: "a ten second local video",
            width: 832,
            height: 480,
            numFrames: 243
        )
        XCTAssertEqual(longClip.steps, 21)
        let acceleratedLongClip = try MiniMaxH3GenerationOptions(
            prompt: "a fast ten second local video",
            width: 832,
            height: 480,
            numFrames: 243,
            accelerationMode: .maximum
        )
        XCTAssertEqual(acceleratedLongClip.steps, 12)
    }

    func testExactScheduleCacheSurvivesDiscardingBF16AdaLNWeights() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 32,
            layerCount: 2,
            refinerLayerCount: 1,
            attentionHeadCount: 4,
            attentionHeadDimension: 8,
            feedForwardSize: 64,
            videoLatentChannels: 8,
            audioLatentChannels: 32,
            patchSize: [1, 2, 2],
            textDimension: 32,
            timeFrequencyDimension: 32,
            timeEmbeddingHiddenSize: 32,
            timeEmbeddingDimension: 32,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        let videoSchedule = try MiniMaxH3Schedule(pointCount: 4, shift: 12)
        let audioSchedule = try MiniMaxH3Schedule(pointCount: 4, shift: 3)
        let cache = model.precomputeAdaLN(
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule,
            sourceIdentity: "test"
        )
        XCTAssertEqual(cache.stepCount, 3)
        let scheduleBytesBefore = model.parameters().flattened()
            .filter { $0.0.contains("adaln_proj") || $0.0.hasPrefix("time_embedder.") }
            .reduce(0) { $0 + $1.1.nbytes }
        XCTAssertGreaterThan(scheduleBytesBefore, 1_000)

        model.discardAdaLNWeights()

        let scheduleParametersAfter = model.parameters().flattened()
            .filter { $0.0.contains("adaln_proj") || $0.0.hasPrefix("time_embedder.") }
        XCTAssertTrue(scheduleParametersAfter.allSatisfy { $0.1.size == 1 })
        XCTAssertLessThan(scheduleParametersAfter.reduce(0) { $0 + $1.1.nbytes }, scheduleBytesBefore)
        XCTAssertEqual(cache.step(at: 0).blockModulations.count, 2)
    }

    func testTinyTransformerResidentBF16MatchesQuantizedExecution() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 32,
            layerCount: 2,
            refinerLayerCount: 1,
            attentionHeadCount: 4,
            attentionHeadDimension: 8,
            feedForwardSize: 64,
            videoLatentChannels: 8,
            audioLatentChannels: 32,
            patchSize: [1, 2, 2],
            textDimension: 32,
            timeFrequencyDimension: 32,
            timeEmbeddingHiddenSize: 32,
            timeEmbeddingDimension: 32,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        quantize(
            model: model,
            groupSize: 32,
            bits: 4,
            filter: { _, _ in true },
            apply: { module, groupSize, bits, mode in
                guard let quantized = quantizeSingle(
                    layer: module,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                ) as? QuantizedLinear else { return nil }
                return PortableQuantizedLinear(
                    weight: quantized.weight,
                    bias: quantized.bias,
                    scales: quantized.scales,
                    biases: quantized.biases,
                    groupSize: quantized.groupSize,
                    bits: quantized.bits,
                    mode: quantized.mode,
                    globalScale: quantized.globalScale
                )
            }
        )
        let quantizedCount = model.leafModules().flattened()
            .count(where: { $0.1 is QuantizedLinear })
        let estimatedBytes = model.estimatedResidentBF16ByteCount
        XCTAssertGreaterThan(quantizedCount, 0)
        XCTAssertGreaterThan(estimatedBytes, 0)

        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1, 1],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3,
            keyframeAnchors: [.first]
        )
        let video = MLXArray.zeros([1, 12, 32], dtype: .bfloat16)
        let audio = MLXArray.zeros([1, 6, 32], dtype: .bfloat16)
        let text = MLXArray.zeros([1, 2, 32], dtype: .bfloat16)
        let quantizedOutput = model(
            videoRows: video,
            audioRows: audio,
            textStates: text,
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(quantizedOutput.videoVelocityRows, quantizedOutput.audioVelocityRows)

        let materialized = model.materializeResidentBF16()
        XCTAssertTrue(model.usesResidentBF16)
        XCTAssertEqual(materialized.linearCount, quantizedCount)
        XCTAssertEqual(materialized.byteCount, estimatedBytes)
        XCTAssertFalse(model.leafModules().flattened().contains { $0.1 is QuantizedLinear })
        let denseOutput = model(
            videoRows: video,
            audioRows: audio,
            textStates: text,
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(denseOutput.videoVelocityRows, denseOutput.audioVelocityRows)
        XCTAssertLessThanOrEqual(
            MLX.abs(
                quantizedOutput.videoVelocityRows.asType(.float32)
                    - denseOutput.videoVelocityRows.asType(.float32)
            ).max().item(Float.self),
            0.05
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                quantizedOutput.audioVelocityRows.asType(.float32)
                    - denseOutput.audioVelocityRows.asType(.float32)
            ).max().item(Float.self),
            0.05
        )
    }

    func testTinyDenseBF16TransformerIsActuallyMaterialized() {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 32,
            layerCount: 2,
            refinerLayerCount: 1,
            attentionHeadCount: 4,
            attentionHeadDimension: 8,
            feedForwardSize: 64,
            videoLatentChannels: 8,
            audioLatentChannels: 32,
            patchSize: [1, 2, 2],
            textDimension: 32,
            timeFrequencyDimension: 32,
            timeEmbeddingHiddenSize: 32,
            timeEmbeddingDimension: 32,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        let linearCount = model.leafModules().flattened().count { $0.1 is Linear }
        let estimatedBytes = model.estimatedResidentBF16ByteCount

        XCTAssertGreaterThan(linearCount, 0)
        XCTAssertGreaterThan(estimatedBytes, 0)
        XCTAssertFalse(model.usesResidentBF16)

        let materialized = model.materializeResidentBF16()

        XCTAssertTrue(model.usesResidentBF16)
        XCTAssertEqual(materialized.linearCount, linearCount)
        XCTAssertEqual(materialized.byteCount, estimatedBytes)
        XCTAssertFalse(model.leafModules().flattened().contains { $0.1 is QuantizedLinear })
    }

    func testTinyTransformerPreservesTargetShapes() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 12,
            layerCount: 2,
            refinerLayerCount: 1,
            attentionHeadCount: 2,
            attentionHeadDimension: 6,
            feedForwardSize: 16,
            videoLatentChannels: 3,
            audioLatentChannels: 4,
            patchSize: [1, 2, 2],
            textDimension: 10,
            timeFrequencyDimension: 4,
            timeEmbeddingHiddenSize: 12,
            timeEmbeddingDimension: 8,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1, 1],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3,
            keyframeAnchors: [.first]
        )
        let result = model(
            videoRows: MLXArray.zeros([1, 12, 12]),
            audioRows: MLXArray.zeros([1, 6, 4]),
            textStates: MLXArray.zeros([1, 2, 10]),
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(result.videoVelocityRows, result.audioVelocityRows)
        XCTAssertEqual(result.videoVelocityRows.shape, [1, 8, 12])
        XCTAssertEqual(result.audioVelocityRows.shape, [1, 6, 4])
    }

    func testTinyTransformerExecutesTokenReductionAcrossFullAndReducedBoundaries() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 12,
            layerCount: 4,
            refinerLayerCount: 1,
            attentionHeadCount: 2,
            attentionHeadDimension: 6,
            feedForwardSize: 16,
            videoLatentChannels: 3,
            audioLatentChannels: 4,
            patchSize: [1, 2, 2],
            textDimension: 10,
            timeFrequencyDimension: 4,
            timeEmbeddingHiddenSize: 12,
            timeEmbeddingDimension: 8,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1, 1],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 6,
            audioLatentFrames: 3,
            keyframeAnchors: []
        )
        let video = MLXArray.zeros([1, 12, 12])
        let audio = MLXArray.zeros([1, 6, 4])
        let text = MLXArray.zeros([1, 2, 10])
        let context = model.prepare(textStates: text, layout: layout)
        let reduction = model.prepareTokenReduction(context: context)
        let output = model.callWithTokenReduction(
            videoRows: video,
            audioRows: audio,
            context: context,
            reduction: reduction,
            timesteps: MLXArray([Float(0.2), 0.4, 0.999]),
            cachedAdaLN: nil,
            policy: MiniMaxH3TokenReductionPolicy(
                beginBlock: 1,
                endBlock: 2,
                earlyStepCount: 1,
                earlyEndBlock: 3
            ),
            stepIndex: 0
        )
        MLX.eval(output.videoVelocityRows, output.audioVelocityRows)

        XCTAssertEqual(reduction.reducedContext.layout.sequenceLength, 16)
        XCTAssertEqual(output.videoVelocityRows.shape, [1, 12, 12])
        XCTAssertEqual(output.audioVelocityRows.shape, [1, 6, 4])
        XCTAssertTrue(output.videoVelocityRows.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertTrue(output.audioVelocityRows.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testTinyTransformerAcceptsRef2VAConditionAudioAndVideo() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 12,
            layerCount: 1,
            refinerLayerCount: 1,
            attentionHeadCount: 2,
            attentionHeadDimension: 6,
            feedForwardSize: 16,
            videoLatentChannels: 3,
            audioLatentChannels: 4,
            patchSize: [1, 2, 2],
            textDimension: 10,
            timeFrequencyDimension: 4,
            timeEmbeddingHiddenSize: 12,
            timeEmbeddingDimension: 8,
            ropeFrequencyCount: 1
        )
        let layout = try MiniMaxH3Geometry.buildRef2VA(
            textTokenTags: [1, 1],
            references: [
                .init(
                    kind: .video,
                    videoLatentFrames: 1,
                    latentHeight: 4,
                    latentWidth: 4,
                    audioLatentFrames: 2
                ),
            ],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3
        )
        let result = MiniMaxH3Transformer(configuration: configuration)(
            videoRows: MLXArray.zeros([1, 12, 12]),
            audioRows: MLXArray.zeros([1, 10, 4]),
            textStates: MLXArray.zeros([1, 2, 10]),
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(result.videoVelocityRows, result.audioVelocityRows)
        XCTAssertEqual(result.videoVelocityRows.shape, [1, 8, 12])
        XCTAssertEqual(result.audioVelocityRows.shape, [1, 6, 4])
    }

    func testPreparedAndCompiledTinyTransformerMatchDirectExecution() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 32,
            layerCount: 2,
            refinerLayerCount: 1,
            attentionHeadCount: 4,
            attentionHeadDimension: 8,
            feedForwardSize: 64,
            videoLatentChannels: 3,
            audioLatentChannels: 4,
            patchSize: [1, 2, 2],
            textDimension: 32,
            timeFrequencyDimension: 8,
            timeEmbeddingHiddenSize: 32,
            timeEmbeddingDimension: 32,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        quantize(
            model: model,
            groupSize: 32,
            bits: 4,
            filter: { path, _ in path.hasPrefix("blocks.") },
            apply: { module, groupSize, bits, mode in
                guard let quantized = quantizeSingle(
                    layer: module,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                ) as? QuantizedLinear else { return nil }
                return PortableQuantizedLinear(
                    weight: quantized.weight,
                    bias: quantized.bias,
                    scales: quantized.scales,
                    biases: quantized.biases,
                    groupSize: quantized.groupSize,
                    bits: quantized.bits,
                    mode: quantized.mode,
                    globalScale: quantized.globalScale
                )
            }
        )
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1, 1],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3,
            keyframeAnchors: [.first]
        )
        let video = MLXArray.zeros([1, 12, 12])
        let audio = MLXArray.zeros([1, 6, 4])
        let text = MLXArray.zeros([1, 2, 32])
        let timesteps = MLXArray([Float(0.2), 0.4, 0.999])
        let direct = model(
            videoRows: video,
            audioRows: audio,
            textStates: text,
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        let context = model.prepare(textStates: text, layout: layout)
        let prepared = model(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: nil
        )
        let compiled = MLX.compile { (inputs: [MLXArray]) -> [MLXArray] in
            let output = model(
                videoRows: inputs[0],
                audioRows: inputs[1],
                context: context,
                timesteps: inputs[2],
                cachedAdaLN: nil
            )
            return [output.videoVelocityRows, output.audioVelocityRows]
        }
        let compiledOutputs = compiled([video, audio, timesteps])
        MLX.eval(compiledOutputs)
        model.maximumAttentionQueryTokensPerKernel = 2
        model.maximumAttentionHeadsPerKernel = 1
        model.usesBlockwiseCompilation = true
        let blockwiseCompiled = model(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: nil
        )
        MLX.eval(blockwiseCompiled.videoVelocityRows, blockwiseCompiled.audioVelocityRows)
        let refreshedReuse = model.callWithBlockResidualReuse(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: nil,
            warmBlockCount: 1,
            cachedTailResidual: nil
        )
        let reusedTail = model.callWithBlockResidualReuse(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: nil,
            warmBlockCount: 1,
            cachedTailResidual: try XCTUnwrap(refreshedReuse.refreshedTailResidual)
        )
        let adaptivePolicy = MiniMaxH3AdaptiveFirstBlockCachePolicy(
            globalThreshold: 0.1,
            temporalThreshold: 0.1
        )
        let adaptiveRefresh = model.callWithAdaptiveFirstBlockReuse(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: nil,
            policy: adaptivePolicy,
            canConsiderReuse: false,
            previousFirstResidual: nil,
            cachedTargetTailResidual: nil
        )
        let adaptiveReuse = model.callWithAdaptiveFirstBlockReuse(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: nil,
            policy: adaptivePolicy,
            canConsiderReuse: true,
            previousFirstResidual: try XCTUnwrap(adaptiveRefresh.refreshedFirstResidual),
            cachedTargetTailResidual: try XCTUnwrap(adaptiveRefresh.refreshedTargetTailResidual)
        )
        MLX.eval(
            refreshedReuse.output.videoVelocityRows,
            refreshedReuse.output.audioVelocityRows,
            reusedTail.output.videoVelocityRows,
            reusedTail.output.audioVelocityRows,
            adaptiveRefresh.output.videoVelocityRows,
            adaptiveRefresh.output.audioVelocityRows,
            adaptiveReuse.output.videoVelocityRows,
            adaptiveReuse.output.audioVelocityRows
        )
        model.usesBlockwiseCompilation = false
        model.usesLayerwiseEvaluation = true
        model.clearsCacheAfterLayerwiseEvaluation = false
        let layerwise = model(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: nil
        )
        MLX.eval(layerwise.videoVelocityRows, layerwise.audioVelocityRows)
        model.usesLayerwiseEvaluation = false
        MLX.eval(
            direct.videoVelocityRows,
            direct.audioVelocityRows,
            prepared.videoVelocityRows,
            prepared.audioVelocityRows,
            compiledOutputs[0],
            compiledOutputs[1],
            blockwiseCompiled.videoVelocityRows,
            blockwiseCompiled.audioVelocityRows,
            layerwise.videoVelocityRows,
            layerwise.audioVelocityRows
        )
        XCTAssertTrue(direct.videoVelocityRows.allClose(prepared.videoVelocityRows).item(Bool.self))
        XCTAssertTrue(direct.audioVelocityRows.allClose(prepared.audioVelocityRows).item(Bool.self))
        XCTAssertLessThanOrEqual(
            MLX.abs(prepared.videoVelocityRows - compiledOutputs[0]).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(prepared.audioVelocityRows - compiledOutputs[1]).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(prepared.videoVelocityRows - blockwiseCompiled.videoVelocityRows).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(prepared.audioVelocityRows - blockwiseCompiled.audioVelocityRows)
                .max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                blockwiseCompiled.videoVelocityRows
                    - refreshedReuse.output.videoVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                blockwiseCompiled.audioVelocityRows
                    - refreshedReuse.output.audioVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                refreshedReuse.output.videoVelocityRows
                    - reusedTail.output.videoVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                refreshedReuse.output.audioVelocityRows
                    - reusedTail.output.audioVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertFalse(adaptiveRefresh.reusedTail)
        XCTAssertNil(adaptiveRefresh.change)
        XCTAssertTrue(adaptiveReuse.reusedTail)
        let adaptiveChange = try XCTUnwrap(adaptiveReuse.change)
        XCTAssertEqual(adaptiveChange.videoGlobal, 0, accuracy: 1e-6)
        XCTAssertEqual(adaptiveChange.audioGlobal, 0, accuracy: 1e-6)
        XCTAssertLessThanOrEqual(
            MLX.abs(
                blockwiseCompiled.videoVelocityRows
                    - adaptiveRefresh.output.videoVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                blockwiseCompiled.audioVelocityRows
                    - adaptiveRefresh.output.audioVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                adaptiveRefresh.output.videoVelocityRows
                    - adaptiveReuse.output.videoVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                adaptiveRefresh.output.audioVelocityRows
                    - adaptiveReuse.output.audioVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(prepared.videoVelocityRows - layerwise.videoVelocityRows)
                .max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(prepared.audioVelocityRows - layerwise.audioVelocityRows)
                .max().item(Float.self),
            1e-5
        )

        let videoSchedule = try MiniMaxH3Schedule(pointCount: 3, shift: 12)
        let audioSchedule = try MiniMaxH3Schedule(pointCount: 3, shift: 3)
        let cache = model.precomputeAdaLN(
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule,
            sourceIdentity: "test-transformer"
        )
        let scheduleTimesteps = MLXArray([
            videoSchedule.timesteps[1],
            audioSchedule.timesteps[1],
            max(videoSchedule.timesteps[1], 0.999),
        ])
        let eagerScheduleOutput = model(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: scheduleTimesteps,
            cachedAdaLN: nil
        )
        let cachedScheduleOutput = model(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: scheduleTimesteps,
            cachedAdaLN: cache.step(at: 1)
        )
        model.usesBlockwiseCompilation = true
        let blockwiseCachedScheduleOutput = model(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: scheduleTimesteps,
            cachedAdaLN: cache.step(at: 1)
        )
        MLX.eval(
            blockwiseCachedScheduleOutput.videoVelocityRows,
            blockwiseCachedScheduleOutput.audioVelocityRows
        )
        model.usesBlockwiseCompilation = false
        MLX.eval(
            eagerScheduleOutput.videoVelocityRows,
            eagerScheduleOutput.audioVelocityRows,
            cachedScheduleOutput.videoVelocityRows,
            cachedScheduleOutput.audioVelocityRows,
            blockwiseCachedScheduleOutput.videoVelocityRows,
            blockwiseCachedScheduleOutput.audioVelocityRows
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                eagerScheduleOutput.videoVelocityRows
                    - cachedScheduleOutput.videoVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                eagerScheduleOutput.audioVelocityRows
                    - cachedScheduleOutput.audioVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                cachedScheduleOutput.videoVelocityRows
                    - blockwiseCachedScheduleOutput.videoVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                cachedScheduleOutput.audioVelocityRows
                    - blockwiseCachedScheduleOutput.audioVelocityRows
            ).max().item(Float.self),
            1e-5
        )

        let cacheURL = FileManager.default.temporaryDirectory
            .appending(path: "minimax-h3-adaln-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: cacheURL) }
        try cache.save(to: cacheURL, replacing: false)
        let loaded = try MiniMaxH3AdaLNCache.load(
            from: cacheURL,
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule,
            sourceIdentity: "test-transformer"
        )
        XCTAssertEqual(loaded.stepCount, 2)
        XCTAssertEqual(loaded.blockModulations.count, 2)
        XCTAssertTrue(loaded.isCompatible(
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule
        ))
        XCTAssertThrowsError(try MiniMaxH3AdaLNCache.load(
            from: cacheURL,
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule,
            sourceIdentity: "stale-transformer"
        ))

        let denseVideoSchedule = try MiniMaxH3Schedule(pointCount: 5, shift: 12)
        let denseAudioSchedule = try MiniMaxH3Schedule(pointCount: 5, shift: 3)
        let denseCache = model.precomputeAdaLN(
            videoSchedule: denseVideoSchedule,
            audioSchedule: denseAudioSchedule,
            sourceIdentity: "test-transformer"
        )
        let resampled = try denseCache.resampled(
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule
        )
        MLX.eval(
            [resampled.timeEmbeddings, resampled.finalModulations]
                + resampled.blockModulations
        )
        XCTAssertTrue(resampled.isCompatible(
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule
        ))
        XCTAssertEqual(resampled.timeEmbeddings.dtype, denseCache.timeEmbeddings.dtype)
        XCTAssertEqual(resampled.finalModulations.dtype, denseCache.finalModulations.dtype)
        XCTAssertEqual(resampled.blockModulations[0].dtype, denseCache.blockModulations[0].dtype)
        XCTAssertLessThanOrEqual(
            MLX.abs(resampled.timeEmbeddings - cache.timeEmbeddings).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(resampled.finalModulations - cache.finalModulations).max().item(Float.self),
            1e-5
        )
        for index in cache.blockModulations.indices {
            XCTAssertLessThanOrEqual(
                MLX.abs(resampled.blockModulations[index] - cache.blockModulations[index])
                    .max().item(Float.self),
                1e-5
            )
        }

        try denseCache.save(to: cacheURL, replacing: true)
        XCTAssertThrowsError(try MiniMaxH3AdaLNCache.load(
            from: cacheURL,
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule,
            sourceIdentity: "test-transformer"
        ))
        let loadedResampled = try MiniMaxH3AdaLNCache.load(
            from: cacheURL,
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule,
            sourceIdentity: "test-transformer",
            allowScheduleResampling: true
        )
        XCTAssertTrue(loadedResampled.isCompatible(
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule
        ))
    }

    func testTinyVideoDecoderShape() {
        let decoder = MiniMaxH3VideoDecoder(configuration: .init(
            latentChannels: 2,
            outputChannels: 3,
            patchSize: 2,
            temporalPatchSize: 2,
            layerCount: 1,
            headCount: 2,
            headDimension: 6,
            registerTokenCount: 1,
            feedForwardMultiplier: 2,
            rotaryDimensionRatio: 1
        ))
        let result = decoder(MLXArray.zeros([1, 2, 2, 2, 2]))
        MLX.eval(result)
        XCTAssertEqual(result.shape, [1, 3, 4, 4, 4])
    }
}
