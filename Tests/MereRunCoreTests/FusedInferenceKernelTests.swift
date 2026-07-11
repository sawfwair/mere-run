import Foundation
import MLX
import MLXFast
import MLXRandom
import XCTest
@testable import MereRunCore

final class FusedInferenceKernelTests: MereRunCoreTestCase {
    func testFusedFullAttentionMatchesMaterializedReference() {
        MLXRandom.seed(71)
        let queries = MLXRandom.uniform(-0.4 ..< 0.4, [1, 4, 17, 64])
        let keys = MLXRandom.uniform(-0.4 ..< 0.4, [1, 4, 17, 64])
        let values = MLXRandom.uniform(-0.4 ..< 0.4, [1, 4, 17, 64])
        let scale: Float = 1.0 / sqrt(64.0)

        let reference = materializedAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: nil
        )
        let fused = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: MLXFast.ScaledDotProductAttentionMaskMode.none
        )
        MLX.eval(reference, fused)

        XCTAssertEqual(fused.shape, reference.shape)
        XCTAssertLessThan(maxDifference(reference, fused), 2e-4)
    }

    func testFusedFullAttentionArrayMaskMatchesMaterializedReference() {
        MLXRandom.seed(72)
        let queries = MLXRandom.uniform(-0.4 ..< 0.4, [1, 2, 19, 64])
        let keys = MLXRandom.uniform(-0.4 ..< 0.4, [1, 2, 19, 64])
        let values = MLXRandom.uniform(-0.4 ..< 0.4, [1, 2, 19, 64])
        let scale: Float = 1.0 / sqrt(64.0)
        let mask = causalAdditiveMask(sequence: 19)

        let reference = materializedAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        let fused = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        MLX.eval(reference, fused)

        XCTAssertLessThan(maxDifference(reference, fused), 2e-4)
    }

    func testFusedCheckpointAttentionWidthsMatchBFloat16Reference() {
        MLXRandom.seed(721)
        for (sequence, heads, headDim) in [(17, 4, 64), (17, 2, 128), (1, 16, 96)] {
            let queries = MLXRandom.uniform(
                -0.4 ..< 0.4,
                [1, heads, sequence, headDim]
            ).asType(.bfloat16)
            let keys = MLXRandom.uniform(
                -0.4 ..< 0.4,
                [1, heads, max(sequence, 23), headDim]
            ).asType(.bfloat16)
            let values = MLXRandom.uniform(
                -0.4 ..< 0.4,
                [1, heads, max(sequence, 23), headDim]
            ).asType(.bfloat16)
            let scale = Float(1.0 / sqrt(Double(headDim)))
            let reference = materializedAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: nil
            )
            let fused = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: MLXFast.ScaledDotProductAttentionMaskMode.none
            )
            MLX.eval(reference, fused)

            XCTAssertLessThan(
                maxDifference(reference, fused),
                0.016,
                "Mismatch for qLen=\(sequence), headDim=\(headDim)"
            )
        }
    }

    func testSAMLayerNorm2dMatchesOpsReference() {
        MLXRandom.seed(73)
        let input = MLXRandom.uniform(-0.7 ..< 0.7, [2, 3, 4, 8])
        let norm = SAM31LayerNorm2d(numChannels: 8, eps: 1e-6)
        let output = norm(input)

        let mean = input.mean(axis: -1, keepDims: true)
        let variance = ((input - mean) ** 2).mean(axis: -1, keepDims: true)
        let reference = (input - mean) / MLX.sqrt(variance + MLXArray(Float(1e-6)))
        MLX.eval(reference, output)

        XCTAssertLessThan(maxDifference(reference, output), 2e-4)
    }

    func testWooshFusedNormsMatchOpsReferences() {
        MLXRandom.seed(74)
        let input = MLXRandom.uniform(-0.7 ..< 0.7, [2, 3, 32])
        let weight = MLXRandom.uniform(0.7 ..< 1.3, [32])
        let epsilon = Float(1e-6)

        let mean = input.mean(axis: -1, keepDims: true)
        let variance = ((input - mean) * (input - mean)).mean(axis: -1, keepDims: true)
        let layerReference = (input - mean) / MLX.sqrt(variance + MLXArray(epsilon))
        let rmsReference = input * MLX.rsqrt(
            (input * input).mean(axis: -1, keepDims: true) + MLXArray(epsilon)
        ) * weight
        let layerFused = WooshTensorOps.layerNormNoAffine(input, eps: epsilon)
        let rmsFused = WooshTensorOps.rmsNorm(input, weight: weight, eps: epsilon)
        MLX.eval(layerReference, rmsReference, layerFused, rmsFused)

        XCTAssertLessThan(maxDifference(layerReference, layerFused), 2e-4)
        XCTAssertLessThan(maxDifference(rmsReference, rmsFused), 2e-4)
    }

    func testIdeogramStructuralSavingsAccountForEveryDenoiserLayer() {
        let savings = Ideogram4FusedKernelSavings.estimate(
            batch: 1,
            sequence: 4_160,
            heads: 18,
            headDim: 256
        )

        XCTAssertEqual(savings.explicitDispatchesPerLayer, 5)
        XCTAssertEqual(savings.stridedQKElementsPerLayer, 38_338_560)
        XCTAssertEqual(savings.uniformMaskElementsPerLayer, 17_305_600)
        XCTAssertEqual(savings.maskedScoreElementsPerLayer, 311_500_800)
    }

    #if os(macOS)
    func testIdeogramQKVKernelMatchesSeparateNorms() throws {
        try requireGPU()
        MLXRandom.seed(75)
        let batch = 1
        let sequence = 7
        let heads = 3
        let headDim = 64
        let qkv = MLXRandom.uniform(
            -0.5 ..< 0.5,
            [batch, sequence, 3 * heads * headDim]
        ).asType(.bfloat16)
        let qWeight = MLXRandom.uniform(0.7 ..< 1.3, [headDim]).asType(.bfloat16)
        let kWeight = MLXRandom.uniform(0.7 ..< 1.3, [headDim]).asType(.bfloat16)
        let epsilon = Float(1e-5)
        let pieces = MLX.split(qkv, parts: 3, axis: -1)
        let qReference = MLXFast.rmsNorm(
            pieces[0].reshaped(batch, sequence, heads, headDim),
            weight: qWeight,
            eps: epsilon
        ).transposed(0, 2, 1, 3)
        let kReference = MLXFast.rmsNorm(
            pieces[1].reshaped(batch, sequence, heads, headDim),
            weight: kWeight,
            eps: epsilon
        ).transposed(0, 2, 1, 3)
        let vReference = pieces[2].reshaped(batch, sequence, heads, headDim)
            .transposed(0, 2, 1, 3)

        let fused = try XCTUnwrap(Ideogram4FusedKernels.qkvNorms(
            qkv: qkv,
            qWeight: qWeight,
            kWeight: kWeight,
            eps: epsilon,
            numHeads: heads,
            headDim: headDim
        ))
        MLX.eval(qReference, kReference, vReference, fused.queries, fused.keys, fused.values)

        XCTAssertLessThan(maxDifference(qReference, fused.queries), 0.016)
        XCTAssertLessThan(maxDifference(kReference, fused.keys), 0.016)
        XCTAssertEqual(maxDifference(vReference, fused.values), 0)
    }

    func testIdeogramAdaLNKernelsMatchSeparateOps() throws {
        try requireGPU()
        MLXRandom.seed(76)
        let input = MLXRandom.uniform(-0.5 ..< 0.5, [2, 5, 128]).asType(.bfloat16)
        let residual = MLXRandom.uniform(-0.5 ..< 0.5, [2, 5, 128]).asType(.bfloat16)
        let weight = MLXRandom.uniform(0.7 ..< 1.3, [128]).asType(.bfloat16)
        let modulation = MLXRandom.uniform(-0.7 ..< 0.7, [2, 1, 512]).asType(.bfloat16)
        let epsilon = Float(1e-5)
        let normalized = MLXFast.rmsNorm(input, weight: weight, eps: epsilon)
        let pieces = MLX.split(modulation, parts: 4, axis: -1)
        let scaledReference = normalized * (1 + pieces[0])
        let residualReference = residual + MLX.tanh(pieces[1]) * normalized

        XCTAssertEqual(input.ndim, 3)
        XCTAssertEqual(weight.ndim, 1)
        XCTAssertEqual(weight.dim(0), input.dim(2))
        XCTAssertEqual(weight.dtype, input.dtype)
        XCTAssertEqual(modulation.dtype, input.dtype)
        XCTAssertEqual(modulation.size, 4 * input.dim(0) * input.dim(2))

        let scaled = try XCTUnwrap(Ideogram4FusedKernels.scaledRMSNorm(
            input,
            weight: weight,
            modulation: modulation,
            modulationIndex: 0,
            eps: epsilon
        ))
        let gated = try XCTUnwrap(Ideogram4FusedKernels.gatedResidualRMSNorm(
            input,
            residual: residual,
            weight: weight,
            modulation: modulation,
            modulationIndex: 1,
            eps: epsilon
        ))
        MLX.eval(scaledReference, residualReference, scaled, gated)

        XCTAssertLessThan(maxDifference(scaledReference, scaled), 0.016)
        XCTAssertLessThan(maxDifference(residualReference, gated), 0.016)
    }

    func testTinyIdeogramFusedForwardMatchesPortableGraph() throws {
        try requireGPU()
        let configuration = Ideogram4TransformerConfiguration(
            adalnDim: 32,
            attentionHeadDim: 64,
            inChannels: 16,
            intermediateSize: 256,
            llmFeaturesDim: 32,
            mropeSection: [16, 8, 8],
            normEps: 1e-5,
            numAttentionHeads: 2,
            numLayers: 2,
            ropeTheta: 10_000
        )
        let portable = Ideogram4Transformer(
            configuration: configuration,
            fusedKernelsEnabled: false
        )
        let fused = Ideogram4Transformer(
            configuration: configuration,
            fusedKernelsEnabled: true
        )
        let parameters = portable.parameters().mapValues {
            $0.dtype == .float32 ? $0.asType(.bfloat16) : $0
        }
        portable.update(parameters: parameters)
        fused.update(parameters: parameters)

        MLXRandom.seed(77)
        let features = MLXRandom.uniform(-0.5 ..< 0.5, [1, 2, 32]).asType(.bfloat16)
        let latents = MLXRandom.uniform(-0.5 ..< 0.5, [1, 2, 16]).asType(.bfloat16)
        let sample = try Ideogram4SampleBuilder.pack(
            llmFeatures: features,
            imageLatents: latents,
            imageWidth: 32,
            imageHeight: 16,
            inChannels: 16
        )

        let portableOutput = portable(sample: sample, timestep: MLXArray([Float(0.5)]))
        let fusedOutput = fused(sample: sample, timestep: MLXArray([Float(0.5)]))
        let maskedOutput = portable(
            llmFeatures: sample.llmFeatures,
            x: sample.x,
            timestep: MLXArray([Float(0.5)]),
            positionIds: sample.positionIds,
            segmentIds: sample.segmentIds,
            indicator: sample.indicator
        )
        MLX.eval(portableOutput, fusedOutput, maskedOutput)

        XCTAssertLessThan(maxDifference(portableOutput, fusedOutput), 0.08)
        XCTAssertLessThan(maxDifference(portableOutput, maskedOutput), 0.001)
    }

    func testSAMAttentionReleaseBenchmark() throws {
        guard ProcessInfo.processInfo.environment["MERERUN_BENCHMARK_FUSED_VISION"] == "1" else {
            throw XCTSkip(
                "Set MERERUN_BENCHMARK_FUSED_VISION=1 and MERERUN_TEST_MLX_DEVICE=gpu "
                    + "to run the fused vision-attention A/B."
            )
        }
        try requireGPU()
        MLXRandom.seed(78)
        let sequence = Int(
            ProcessInfo.processInfo.environment["MERERUN_BENCHMARK_FUSED_VISION_SEQUENCE"] ?? ""
        ) ?? 576
        let queries = MLXRandom.normal([1, 16, sequence, 64]).asType(.bfloat16)
        let keys = MLXRandom.normal([1, 16, sequence, 64]).asType(.bfloat16)
        let values = MLXRandom.normal([1, 16, sequence, 64]).asType(.bfloat16)
        let scale: Float = 1.0 / sqrt(64.0)
        MLX.eval(queries, keys, values)

        func portable() -> MLXArray {
            materializedAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: nil
            )
        }
        func fused() -> MLXArray {
            MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: MLXFast.ScaledDotProductAttentionMaskMode.none
            )
        }

        let portableMilliseconds = benchmarkMilliseconds(iterations: 8, portable)
        let fusedMilliseconds = benchmarkMilliseconds(iterations: 8, fused)
        let portablePeak = peakIncrement(portable)
        let fusedPeak = peakIncrement(fused)
        print(
            String(
                format: "[fused-vision] sequence=%d portable=%.3fms fused=%.3fms "
                    + "speedup=%.2fx portablePeak=%.2fMiB fusedPeak=%.2fMiB",
                sequence,
                portableMilliseconds,
                fusedMilliseconds,
                portableMilliseconds / fusedMilliseconds,
                Double(portablePeak) / 1_048_576.0,
                Double(fusedPeak) / 1_048_576.0
            )
        )

        XCTAssertLessThan(fusedMilliseconds, portableMilliseconds)
        XCTAssertLessThan(fusedPeak, portablePeak)
    }

    func testIdeogramKernelReleaseBenchmark() throws {
        guard ProcessInfo.processInfo.environment["MERERUN_BENCHMARK_FUSED_IDEOGRAM"] == "1" else {
            throw XCTSkip(
                "Set MERERUN_BENCHMARK_FUSED_IDEOGRAM=1 and MERERUN_TEST_MLX_DEVICE=gpu "
                    + "to run the Ideogram custom-kernel A/B."
            )
        }
        try requireGPU()
        MLXRandom.seed(79)
        let sequence = Int(
            ProcessInfo.processInfo.environment["MERERUN_BENCHMARK_FUSED_IDEOGRAM_SEQUENCE"] ?? ""
        ) ?? 512
        let heads = 18
        let headDim = 256
        let hidden = heads * headDim
        let epsilon = Float(1e-5)
        let qkv = MLXRandom.normal([1, sequence, 3 * hidden]).asType(.bfloat16)
        let qWeight = MLXRandom.uniform(0.7 ..< 1.3, [headDim]).asType(.bfloat16)
        let kWeight = MLXRandom.uniform(0.7 ..< 1.3, [headDim]).asType(.bfloat16)
        let input = MLXRandom.normal([1, sequence, hidden]).asType(.bfloat16)
        let weight = MLXRandom.uniform(0.7 ..< 1.3, [hidden]).asType(.bfloat16)
        let modulation = MLXRandom.uniform(-0.7 ..< 0.7, [1, 1, 4 * hidden]).asType(.bfloat16)
        MLX.eval(qkv, qWeight, kWeight, input, weight, modulation)

        func portableQKV() -> [MLXArray] {
            let pieces = MLX.split(qkv, parts: 3, axis: -1)
            let queries = MLXFast.rmsNorm(
                pieces[0].reshaped(1, sequence, heads, headDim),
                weight: qWeight,
                eps: epsilon
            ).transposed(0, 2, 1, 3)
            let keys = MLXFast.rmsNorm(
                pieces[1].reshaped(1, sequence, heads, headDim),
                weight: kWeight,
                eps: epsilon
            ).transposed(0, 2, 1, 3)
            let values = pieces[2].reshaped(1, sequence, heads, headDim)
                .transposed(0, 2, 1, 3)
            return [queries, keys, values]
        }
        func fusedQKV() -> [MLXArray] {
            guard let fused = Ideogram4FusedKernels.qkvNorms(
                qkv: qkv,
                qWeight: qWeight,
                kWeight: kWeight,
                eps: epsilon,
                numHeads: heads,
                headDim: headDim
            ) else {
                XCTFail("Expected Ideogram QKV Metal kernel on GPU")
                return portableQKV()
            }
            return [fused.queries, fused.keys, fused.values]
        }
        func portableAdaLN() -> [MLXArray] {
            let scale = 1 + modulation[0..., 0..., 0..<hidden]
            return [MLXFast.rmsNorm(input, weight: weight, eps: epsilon) * scale]
        }
        func fusedAdaLN() -> [MLXArray] {
            guard let fused = Ideogram4FusedKernels.scaledRMSNorm(
                input,
                weight: weight,
                modulation: modulation,
                modulationIndex: 0,
                eps: epsilon
            ) else {
                XCTFail("Expected Ideogram AdaLN Metal kernel on GPU")
                return portableAdaLN()
            }
            return [fused]
        }

        let portableQKVMS = benchmarkMilliseconds(iterations: 8, portableQKV)
        let fusedQKVMS = benchmarkMilliseconds(iterations: 8, fusedQKV)
        let portableAdaLNMS = benchmarkMilliseconds(iterations: 8, portableAdaLN)
        let fusedAdaLNMS = benchmarkMilliseconds(iterations: 8, fusedAdaLN)
        let portableAdaLNPeak = peakIncrement(portableAdaLN)
        let fusedAdaLNPeak = peakIncrement(fusedAdaLN)
        print(
            String(
                format: "[fused-ideogram] sequence=%d qkv-portable=%.3fms qkv-fused=%.3fms "
                    + "qkv-speedup=%.2fx adaln-portable=%.3fms adaln-fused=%.3fms "
                    + "adaln-speedup=%.2fx adaln-portable-peak=%.2fMiB adaln-fused-peak=%.2fMiB",
                sequence,
                portableQKVMS,
                fusedQKVMS,
                portableQKVMS / fusedQKVMS,
                portableAdaLNMS,
                fusedAdaLNMS,
                portableAdaLNMS / fusedAdaLNMS,
                Double(portableAdaLNPeak) / 1_048_576.0,
                Double(fusedAdaLNPeak) / 1_048_576.0
            )
        )

        XCTAssertLessThan(fusedQKVMS, portableQKVMS)
        XCTAssertLessThan(fusedAdaLNMS, portableAdaLNMS)
        XCTAssertLessThan(fusedAdaLNPeak, portableAdaLNPeak)
    }

    func testIdeogramInstalledCheckpointBenchmark() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_BENCHMARK_IDEOGRAM_CHECKPOINT"] == "1" else {
            throw XCTSkip(
                "Set MERERUN_BENCHMARK_IDEOGRAM_CHECKPOINT=1, MERERUN_TEST_MLX_DEVICE=gpu, "
                    + "and the installed model/output paths to run the checkpoint A/B."
            )
        }
        try requireGPU()
        let modelRoot = try XCTUnwrap(environment["MERERUN_TEST_IDEOGRAM_MODEL_ROOT"])
        let outputPath = try XCTUnwrap(environment["MERERUN_TEST_IDEOGRAM_OUTPUT"])
        let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
        let warmupURL = outputURL.deletingLastPathComponent()
            .appendingPathComponent("warmup-\(outputURL.lastPathComponent)")
        let request = GenerationRequest(
            prompt: "a red apple on a wooden table, studio lighting",
            width: 256,
            height: 256,
            steps: 1,
            guidanceScale: 7,
            seed: 4_242,
            outputURL: warmupURL,
            model: modelRoot,
            maxSequenceLength: 512
        )

        Memory.clearCache()
        Memory.peakMemory = 0
        let coldStart = DispatchTime.now().uptimeNanoseconds
        let generator = Ideogram4Generator()
        _ = try await generator.generate(request)
        Stream.gpu.synchronize()
        let coldNanoseconds = DispatchTime.now().uptimeNanoseconds - coldStart
        let coldPeak = Memory.peakMemory

        Memory.clearCache()
        Memory.peakMemory = 0
        let warmBaseline = Memory.activeMemory
        var measuredRequest = request
        measuredRequest.outputURL = outputURL
        let warmStart = DispatchTime.now().uptimeNanoseconds
        _ = try await generator.generate(measuredRequest)
        Stream.gpu.synchronize()
        let warmNanoseconds = DispatchTime.now().uptimeNanoseconds - warmStart
        let warmPeakIncrement = max(0, Memory.peakMemory - warmBaseline)

        let outputBytes = try Data(contentsOf: outputURL).count
        print(
            String(
                format: "[fused-ideogram-checkpoint] kernels=%@ cold=%.3fs warm=%.3fs "
                    + "coldPeak=%.2fGiB warmPeakIncrement=%.2fMiB outputBytes=%d output=%@",
                Ideogram4FusedKernelPolicy.enabled ? "on" : "off",
                Double(coldNanoseconds) / 1_000_000_000.0,
                Double(warmNanoseconds) / 1_000_000_000.0,
                Double(coldPeak) / 1_073_741_824.0,
                Double(warmPeakIncrement) / 1_048_576.0,
                outputBytes,
                outputURL.path
            )
        )

        XCTAssertGreaterThan(outputBytes, 0)
    }
    #endif

    private func materializedAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXArray?
    ) -> MLXArray {
        var scores = MLX.matmul(queries, keys.transposed(0, 1, 3, 2)) * scale
        if let mask {
            scores = scores + mask.asType(scores.dtype)
        }
        scores = softmax(scores, axis: -1)
        return MLX.matmul(scores, values)
    }

    private func causalAdditiveMask(sequence: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: sequence * sequence)
        for row in 0..<sequence {
            for column in (row + 1)..<sequence {
                values[row * sequence + column] = -1e9
            }
        }
        return MLXArray(values, [1, 1, sequence, sequence])
    }

    private func maxDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        MLX.max(MLX.abs(lhs.asType(.float32) - rhs.asType(.float32))).item(Float.self)
    }

    #if os(macOS)
    private func requireGPU() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Custom Metal kernel parity requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }
    }

    private func benchmarkMilliseconds(iterations: Int, _ body: () -> MLXArray) -> Double {
        MLX.eval(body())
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<3 {
            let start = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<iterations {
                MLX.eval(body())
            }
            let elapsed = DispatchTime.now().uptimeNanoseconds - start
            best = min(best, Double(elapsed) / 1_000_000.0 / Double(iterations))
        }
        return best
    }

    private func benchmarkMilliseconds(iterations: Int, _ body: () -> [MLXArray]) -> Double {
        MLX.eval(body())
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<3 {
            let start = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<iterations {
                MLX.eval(body())
            }
            let elapsed = DispatchTime.now().uptimeNanoseconds - start
            best = min(best, Double(elapsed) / 1_000_000.0 / Double(iterations))
        }
        return best
    }

    private func peakIncrement(_ body: () -> MLXArray) -> Int {
        Memory.clearCache()
        Memory.peakMemory = 0
        let baseline = Memory.activeMemory
        autoreleasepool {
            MLX.eval(body())
        }
        return max(0, Memory.peakMemory - baseline)
    }

    private func peakIncrement(_ body: () -> [MLXArray]) -> Int {
        Memory.clearCache()
        Memory.peakMemory = 0
        let baseline = Memory.activeMemory
        autoreleasepool {
            MLX.eval(body())
        }
        return max(0, Memory.peakMemory - baseline)
    }
    #endif
}
