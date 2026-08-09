import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import MereRunCore

/// Always dequantizes in-graph and runs the dense GEMM (per-call transient).
private final class BenchTransientDenseLinear: QuantizedLinear {
    init(copying other: QuantizedLinear) {
        super.init(
            weight: other.weight, bias: other.bias, scales: other.scales,
            biases: other.biases, groupSize: other.groupSize, bits: other.bits,
            mode: other.mode
        )
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dense = MLX.dequantized(
            weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: mode, dtype: x.dtype
        )
        var output = MLX.matmul(x, dense.T)
        if let bias { output = output + bias }
        return output
    }
}

/// Dequantizes once and keeps the dense weight resident.
private final class BenchCachedDenseLinear: QuantizedLinear {
    private var cache: MLXArray?

    init(copying other: QuantizedLinear) {
        super.init(
            weight: other.weight, bias: other.bias, scales: other.scales,
            biases: other.biases, groupSize: other.groupSize, bits: other.bits,
            mode: other.mode
        )
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if cache == nil {
            let dense = MLX.dequantized(
                weight, scales: scales, biases: biases,
                groupSize: groupSize, bits: bits, mode: mode, dtype: x.dtype
            )
            MLX.eval(dense)
            cache = dense
        }
        var output = MLX.matmul(x, cache!.T)
        if let bias { output = output + bias }
        return output
    }
}

/// Env-gated microbenchmarks for the DiT MFU investigation (not part of CI).
/// Times the dominant op shapes of the klein-nano (Flux2, 4-bit) forward pass
/// in-process so ambient machine load affects both arms equally.
/// Run: MERERUN_DIT_BENCH=1 swift test -c release --filter DiTShapeBenchTests
final class DiTShapeBenchTests: XCTestCase {
    private func benchGate() throws {
        guard ProcessInfo.processInfo.environment["MERERUN_DIT_BENCH"] == "1" else {
            throw XCTSkip("Set MERERUN_DIT_BENCH=1 to run DiT shape benchmarks")
        }
    }

    /// Median-of-rounds timing; min is robust to contention spikes.
    private func time(_ label: String, rounds: Int = 3, iters: Int = 10, _ body: () -> MLXArray) -> Double {
        // Warmup
        for _ in 0..<3 { MLX.eval(body()) }
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<rounds {
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iters { MLX.eval(body()) }
            let perIter = (CFAbsoluteTimeGetCurrent() - start) / Double(iters)
            best = min(best, perIter)
        }
        print(String(format: "[dit-bench] %@ %.3f ms", label, best * 1000))
        return best
    }

    /// Measures the fused SDPA dispatch used by a full upstream-equivalent
    /// SCAIL-2 window. This is env-gated because the key/value tensors occupy
    /// about 900 MB and the larger query sizes intentionally stress Metal's
    /// command-buffer watchdog. Set
    /// `MERERUN_SCAIL2_BENCH_QUERY_TOKENS=1024,2048` to isolate sizes during
    /// watchdog qualification. `MERERUN_SCAIL2_BENCH_EVAL_BATCH=4` controls
    /// how many kernels are submitted before each synchronization.
    func testSCAIL2AttentionQueryChunkSizes() throws {
        try benchGate()
        let keyTokens = 42_510
        let heads = 40
        let headDimension = 128
        let keys = MLXRandom.normal([1, heads, keyTokens, headDimension]).asType(.bfloat16)
        let values = MLXRandom.normal([1, heads, keyTokens, headDimension]).asType(.bfloat16)
        MLX.eval(keys, values)
        let scale = Float(1.0 / Double(headDimension).squareRoot())
        let configuredQueryTokens = ProcessInfo.processInfo.environment[
            "MERERUN_SCAIL2_BENCH_QUERY_TOKENS"
        ]?
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            .filter { $0 > 0 } ?? []
        let queryTokenSizes = configuredQueryTokens.isEmpty
            ? [1_024, 2_048, 4_096, 8_192]
            : configuredQueryTokens
        let evaluationBatchSize = max(
            1,
            Int(ProcessInfo.processInfo.environment["MERERUN_SCAIL2_BENCH_EVAL_BATCH"] ?? "") ?? 1
        )
        let clearCacheAfterChunk = ProcessInfo.processInfo.environment[
            "MERERUN_SCAIL2_BENCH_CLEAR_CACHE"
        ] == "1"

        for queryTokens in queryTokenSizes {
            let queries = MLXRandom.normal([1, heads, queryTokens, headDimension]).asType(.bfloat16)
            MLX.eval(queries)
            let start = CFAbsoluteTimeGetCurrent()
            let output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .none
            )
            MLX.eval(output)
            print(
                String(
                    format: "[dit-bench] SCAIL-2 SDPA q=%d k=%d %.3f ms",
                    queryTokens,
                    keyTokens,
                    (CFAbsoluteTimeGetCurrent() - start) * 1_000
                )
            )
            Memory.clearCache()
        }

        let fullQueries = MLXRandom.normal([1, heads, keyTokens, headDimension]).asType(.bfloat16)
        MLX.eval(fullQueries)
        let fullStart = CFAbsoluteTimeGetCurrent()
        let chunkSize = configuredQueryTokens.first ?? 2_048
        let fullOutput: MLXArray
        if clearCacheAfterChunk {
            var chunks: [MLXArray] = []
            for start in stride(from: 0, to: keyTokens, by: chunkSize) {
                let end = min(start + chunkSize, keyTokens)
                let chunk = MLXFast.scaledDotProductAttention(
                    queries: fullQueries[0..., 0..., start..<end, 0...],
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: .none
                )
                MLX.eval(chunk)
                Memory.clearCache()
                chunks.append(chunk)
            }
            fullOutput = MLX.concatenated(chunks, axis: 2)
        } else {
            fullOutput = SCAIL2SelfAttention.scaledDotProductAttention(
                queries: fullQueries,
                keys: keys,
                values: values,
                scale: scale,
                maximumQueryTokens: chunkSize,
                maximumKernelsPerEvaluation: evaluationBatchSize
            )
        }
        MLX.eval(fullOutput)
        print(
            String(
                format: "[dit-bench] SCAIL-2 full q=%d k=%d chunk=%d eval-batch=%d clear=%d %.3f ms",
                keyTokens,
                keyTokens,
                chunkSize,
                evaluationBatchSize,
                clearCacheAfterChunk ? 1 : 0,
                (CFAbsoluteTimeGetCurrent() - fullStart) * 1_000
            )
        )
    }

    /// Compares dense and affine 4-bit projection throughput at the exact
    /// token and channel sizes of an 832x480 SCAIL-2 window.
    func testSCAIL2ProjectionShapes() throws {
        try benchGate()
        let tokens = 42_510
        let hidden = 5_120
        let feedForward = 13_824
        let input = MLXRandom.normal([1, tokens, hidden]).asType(.bfloat16)
        MLX.eval(input)

        func measure(_ label: String, _ body: () -> MLXArray) {
            MLX.eval(body())
            var best = Double.greatestFiniteMagnitude
            for _ in 0..<3 {
                let start = CFAbsoluteTimeGetCurrent()
                MLX.eval(body())
                best = min(best, CFAbsoluteTimeGetCurrent() - start)
            }
            print(String(format: "[dit-bench] SCAIL-2 %@ %.3f ms", label, best * 1_000))
            Memory.clearCache()
        }

        let denseSquare = Linear(hidden, hidden, bias: true)
        denseSquare.update(parameters: denseSquare.parameters().mapValues { $0.asType(.bfloat16) })
        MLX.eval(denseSquare.parameters())
        measure("bf16 \(hidden)->\(hidden)") { denseSquare(input) }

        let quantizedSquare = QuantizedLinear(
            hidden,
            hidden,
            bias: true,
            groupSize: 64,
            bits: 4
        )
        MLX.eval(quantizedSquare.parameters())
        measure("q4-g64 \(hidden)->\(hidden)") { quantizedSquare(input) }

        let denseUp = Linear(hidden, feedForward, bias: true)
        denseUp.update(parameters: denseUp.parameters().mapValues { $0.asType(.bfloat16) })
        MLX.eval(denseUp.parameters())
        measure("bf16 \(hidden)->\(feedForward)") { denseUp(input) }

        let quantizedUp = QuantizedLinear(
            hidden,
            feedForward,
            bias: true,
            groupSize: 64,
            bits: 4
        )
        MLX.eval(quantizedUp.parameters())
        measure("q4-g64 \(hidden)->\(feedForward)") { quantizedUp(input) }
    }

    func testKleinNanoShapes() throws {
        try benchGate()
        // klein-nano: hidden 3072 (24 heads x 128), mlp_ratio 3, 5 double +
        // 20 single blocks, 4-bit groupwise quantized weights. 1024x1024 image
        // = 4096 image tokens; text ~512.
        let tokens = 4608
        let dim = 3072
        let mlp = 9216

        let x = MLXRandom.normal([1, tokens, dim]).asType(.bfloat16)
        MLX.eval(x)

        // --- GEMM arms at the two dominant shapes ---
        var results: [String: Double] = [:]
        for (label, outDim) in [("proj \(dim)->\(dim)", dim), ("mlp \(dim)->\(mlp)", mlp)] {
            let fp = Linear(dim, outDim, bias: false)
            fp.update(parameters: fp.parameters().mapValues { $0.asType(.bfloat16) })
            MLX.eval(fp.parameters())
            results["fp16 \(label)"] = time("bf16-gemm  \(label)") { fp(x) }

            let q4 = QuantizedLinear(dim, outDim, bias: false, groupSize: 64, bits: 4)
            MLX.eval(q4.parameters())
            results["q4 \(label)"] = time("q4-matmul  \(label)") { q4(x) }
        }
        // Down projection (mlp -> dim)
        let xWide = MLXRandom.normal([1, tokens, mlp]).asType(.bfloat16)
        MLX.eval(xWide)
        let fpDown = Linear(mlp, dim, bias: false)
        fpDown.update(parameters: fpDown.parameters().mapValues { $0.asType(.bfloat16) })
        MLX.eval(fpDown.parameters())
        results["fp16 down"] = time("bf16-gemm  down \(mlp)->\(dim)") { fpDown(xWide) }
        let q4Down = QuantizedLinear(mlp, dim, bias: false, groupSize: 64, bits: 4)
        MLX.eval(q4Down.parameters())
        results["q4 down"] = time("q4-matmul  down \(mlp)->\(dim)") { q4Down(xWide) }

        // --- SDPA at klein-nano attention shape ---
        let q = MLXRandom.normal([1, 24, tokens, 128]).asType(.bfloat16)
        let k = MLXRandom.normal([1, 24, tokens, 128]).asType(.bfloat16)
        let v = MLXRandom.normal([1, 24, tokens, 128]).asType(.bfloat16)
        MLX.eval(q, k, v)
        let scale = Float(1.0 / Double(128).squareRoot())
        let sdpa = time("sdpa       [1,24,\(tokens),128]") {
            MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        }

        // --- Roofline synthesis for one full forward ---
        // Per double block (x5): each stream QKV(3x proj) + out proj + MLP
        //   (in 2x mlp as one fused '2*mlp' ~ 2x mlp cost, down mlp->dim);
        //   treat text stream ~ 1/8 of image cost (512 vs 4096 tokens).
        // Per single block (x20): fused qkv+mlp-in ~ (3x proj + mlp), out
        //   (dim+mlp)->dim ~ down + proj.
        // Attention: 1 sdpa per block (joint), 25 blocks.
        func pick(_ key: String) -> Double { results[key] ?? 0 }
        for (arm, projKey, mlpKey, downKey) in [
            ("q4", "q4 proj \(dim)->\(dim)", "q4 mlp \(dim)->\(mlp)", "q4 down"),
            ("fp16", "fp16 proj \(dim)->\(dim)", "fp16 mlp \(dim)->\(mlp)", "fp16 down"),
        ] {
            let proj = pick(projKey), mlpT = pick(mlpKey), down = pick(downKey)
            let doubleBlock = 1.125 * (4 * proj + 2 * mlpT + down)
            let singleBlock = 4 * proj + mlpT + down
            let total = 5 * doubleBlock + 20 * singleBlock + 25 * sdpa
            print(String(format: "[dit-bench] SYNTH %@: double=%.1fms single=%.1fms sdpa=%.1fms -> forward ~= %.0f ms",
                         arm, doubleBlock * 1000, singleBlock * 1000, sdpa * 1000, total * 1000))
        }
    }

    /// Full klein-nano forward with random 4-bit weights: eager vs MLX.compile,
    /// interleaved rounds so ambient load hits both arms equally.
    func testKleinNanoForwardCompileAB() throws {
        try benchGate()
        let config = Flux2TransformerConfiguration(quantized: true)
        let model = Flux2Transformer2DModel(config: config)
        // Match the shipped checkpoint: bf16 activations/norms, quantized ints untouched.
        model.update(parameters: model.parameters().mapValues {
            $0.dtype == .float32 ? $0.asType(.bfloat16) : $0
        })
        MLX.eval(model.parameters())

        let txtLen = 512
        let hidden = MLXRandom.normal([1, 4096, config.inChannels]).asType(.bfloat16)
        let encoder = MLXRandom.normal([1, txtLen, config.contextDim]).asType(.bfloat16)
        let timestep = MLXArray([Float(0.5)])
        let imgIds = Flux2PosEmbed.prepareMultiImageIds(imageCount: 1, height: 64, width: 64, tCoords: [0])
        let txtIds = Flux2PosEmbed.prepareTextIds(seqLen: txtLen, numAxes: 4)
        MLX.eval(hidden, encoder, timestep, imgIds, txtIds)

        let compiled = compile(inputs: [model], outputs: [model]) { inputs in
            [model(
                hiddenStates: inputs[0],
                encoderHiddenStates: inputs[1],
                timestep: inputs[2],
                imgIds: inputs[3],
                txtIds: inputs[4],
                guidance: nil
            )]
        }

        func eager() -> MLXArray {
            model(
                hiddenStates: hidden,
                encoderHiddenStates: encoder,
                timestep: timestep,
                imgIds: imgIds,
                txtIds: txtIds,
                guidance: nil
            )
        }
        func compiledForward() -> MLXArray {
            compiled([hidden, encoder, timestep, imgIds, txtIds])[0]
        }

        // Warmup both arms (compile warmup includes trace+kernel build).
        MLX.eval(eager())
        MLX.eval(compiledForward())

        var eagerBest = Double.greatestFiniteMagnitude
        var compiledBest = Double.greatestFiniteMagnitude
        for _ in 0..<3 {
            var start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<3 { MLX.eval(eager()) }
            eagerBest = min(eagerBest, (CFAbsoluteTimeGetCurrent() - start) / 3)
            start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<3 { MLX.eval(compiledForward()) }
            compiledBest = min(compiledBest, (CFAbsoluteTimeGetCurrent() - start) / 3)
        }
        print(String(format: "[dit-bench] FORWARD eager=%.0f ms compiled=%.0f ms (%.1f%% faster)",
                     eagerBest * 1000, compiledBest * 1000,
                     (eagerBest - compiledBest) / eagerBest * 100))
    }

    /// Isolate per-block-type cost (including dispatch share) by scaling the
    /// block counts: forward(5,20) - forward(0,20) gives 5 double blocks, etc.
    func testKleinNanoBlockScaling() throws {
        try benchGate()
        let txtLen = 512
        let hidden = MLXRandom.normal([1, 4096, 128]).asType(.bfloat16)
        let encoder = MLXRandom.normal([1, txtLen, 7680]).asType(.bfloat16)
        let timestep = MLXArray([Float(0.5)])
        let imgIds = Flux2PosEmbed.prepareMultiImageIds(imageCount: 1, height: 64, width: 64, tCoords: [0])
        let txtIds = Flux2PosEmbed.prepareTextIds(seqLen: txtLen, numAxes: 4)
        MLX.eval(hidden, encoder, timestep, imgIds, txtIds)

        var measured: [String: Double] = [:]
        for (doubles, singles) in [(5, 20), (5, 0), (0, 20), (0, 0)] {
            let config = Flux2TransformerConfiguration(
                numLayers: doubles, numSingleLayers: singles, quantized: true
            )
            let model = Flux2Transformer2DModel(config: config)
            model.update(parameters: model.parameters().mapValues {
                $0.dtype == .float32 ? $0.asType(.bfloat16) : $0
            })
            MLX.eval(model.parameters())
            func forward() -> MLXArray {
                model(
                    hiddenStates: hidden,
                    encoderHiddenStates: encoder,
                    timestep: timestep,
                    imgIds: imgIds,
                    txtIds: txtIds,
                    guidance: nil
                )
            }
            MLX.eval(forward())
            var best = Double.greatestFiniteMagnitude
            for _ in 0..<3 {
                let start = CFAbsoluteTimeGetCurrent()
                for _ in 0..<2 { MLX.eval(forward()) }
                best = min(best, (CFAbsoluteTimeGetCurrent() - start) / 2)
            }
            measured["\(doubles),\(singles)"] = best
            print(String(format: "[dit-bench] blocks(%d,%d) forward=%.0f ms", doubles, singles, best * 1000))
        }
        if let full = measured["5,20"], let noD = measured["0,20"],
           let noS = measured["5,0"], let none = measured["0,0"] {
            print(String(format: "[dit-bench] SPLIT double=%.1f ms/block single=%.1f ms/block fixed=%.0f ms (cross-check full=%.0f vs %.0f)",
                         (full - noD) / 5 * 1000,
                         (full - noS) / 20 * 1000,
                         none * 1000,
                         full * 1000,
                         ((full - noD) + (full - noS) + none) * 1000))
        }
    }

    /// Cumulative prefix timing through one single-block attention: each
    /// stage's delta is its true cost including fusion effects.
    func testSingleBlockSegments() throws {
        try benchGate()
        let tokens = 4608
        let dim = 3072
        let heads = 24
        let headDim = 128
        let attn = Flux2ParallelSelfAttention(
            hiddenSize: dim, numHeads: heads, headDim: headDim, mlpRatio: 3.0, eps: 1e-6
        )
        attn.update(parameters: attn.parameters().mapValues { $0.asType(.bfloat16) })
        MLX.eval(attn.parameters())
        let x = MLXRandom.normal([1, tokens, dim]).asType(.bfloat16)
        let posEmbed = Flux2PosEmbed(theta: 2000, axesDim: [32, 32, 32, 32])
        let ids = Flux2PosEmbed.prepareMultiImageIds(imageCount: 1, height: 64, width: 72, tCoords: [0])
        let rotary = posEmbed(ids)
        MLX.eval(x, rotary.0, rotary.1)

        func stage(_ upTo: Int) -> [MLXArray] {
            let projected = attn.toQkvMlpProj(x)
            if upTo == 0 { return [projected] }
            let qkvDim = 3 * dim
            let qkv = projected[0..., 0..., 0..<qkvDim]
            let mlpIn = projected[0..., 0..., qkvDim...]
            let qkvParts = MLX.split(qkv, parts: 3, axis: -1)
            var q = qkvParts[0].reshaped([1, -1, heads, headDim]).transposed(0, 2, 1, 3)
            var k = qkvParts[1].reshaped([1, -1, heads, headDim]).transposed(0, 2, 1, 3)
            let v = qkvParts[2].reshaped([1, -1, heads, headDim]).transposed(0, 2, 1, 3)
            if upTo == 1 { return [q, k, v, mlpIn] }
            q = attn.normQ(q)
            k = attn.normK(k)
            if upTo == 2 { return [q, k, v, mlpIn] }
            q = Flux2PosEmbed.applyRotaryEmb(q, freqs: rotary)
            k = Flux2PosEmbed.applyRotaryEmb(k, freqs: rotary)
            if upTo == 3 { return [q, k, v, mlpIn] }
            let attnOut = Flux2Attention.attention(q, k, v)
            if upTo == 4 { return [attnOut, mlpIn] }
            let attnReshaped = attnOut.transposed(0, 2, 1, 3).reshaped([1, -1, dim])
            let mlpParts = MLX.split(mlpIn, parts: 2, axis: -1)
            let mlpOut = silu(mlpParts[0]) * mlpParts[1]
            return [attn.toOut(MLX.concatenated([attnReshaped, mlpOut], axis: -1))]
        }

        let labels = ["proj", "+split/transpose", "+qknorm", "+rope", "+sdpa", "+mlp/out"]
        var previous = 0.0
        for upTo in 0...5 {
            for _ in 0..<2 { MLX.eval(stage(upTo)) }
            var best = Double.greatestFiniteMagnitude
            for _ in 0..<3 {
                let start = CFAbsoluteTimeGetCurrent()
                for _ in 0..<4 { MLX.eval(stage(upTo)) }
                best = min(best, (CFAbsoluteTimeGetCurrent() - start) / 4)
            }
            print(String(format: "[dit-bench] seg %-17@ cum=%6.1f ms delta=%6.1f ms",
                         labels[upTo], best * 1000, (best - previous) * 1000))
            previous = best
        }

        // RoPE apply A/B: shipped fp32-upcast versus bf16-native rotation.
        let q4d = MLXRandom.normal([1, heads, tokens, headDim]).asType(.bfloat16)
        MLX.eval(q4d)
        let (fcos, fsin) = rotary
        let cosB = fcos.reshaped([1, 1, fcos.shape[0], fcos.shape[1]])
        let sinB = fsin.reshaped([1, 1, fsin.shape[0], fsin.shape[1]])
        let cosB16 = cosB.asType(.bfloat16)
        let sinB16 = sinB.asType(.bfloat16)
        MLX.eval(cosB16, sinB16)
        _ = time("rope shipped (fp32 upcast)") {
            Flux2PosEmbed.applyRotaryEmb(q4d, freqs: rotary)
        }
        _ = time("rope bf16 native") {
            let x2 = q4d.reshaped([1, heads, tokens, headDim / 2, 2])
            let real = x2[0..., 0..., 0..., 0..., 0]
            let imag = x2[0..., 0..., 0..., 0..., 1]
            let outReal = real * cosB16 + (-imag) * sinB16
            let outImag = imag * cosB16 + real * sinB16
            return MLX.stacked([outReal, outImag], axis: -1).reshaped(q4d.shape)
        }
    }

    /// The single block's exact fused GEMM shapes, bf16 vs 4-bit quantized.
    func testSingleBlockFusedShapesQuantized() throws {
        try benchGate()
        let tokens = 4608
        for (inDim, outDim) in [(3072, 18432), (12288, 3072), (3072, 9216), (3072, 3072)] {
            let x = MLXRandom.normal([1, tokens, inDim]).asType(.bfloat16)
            MLX.eval(x)
            let fp = Linear(inDim, outDim, bias: false)
            fp.update(parameters: fp.parameters().mapValues { $0.asType(.bfloat16) })
            MLX.eval(fp.parameters())
            let fpMs = time(String(format: "bf16 %5d->%5d", inDim, outDim)) { fp(x) }
            let q4 = QuantizedLinear(inDim, outDim, bias: false, groupSize: 64, bits: 4)
            MLX.eval(q4.parameters())
            let q4Ms = time(String(format: "q4   %5d->%5d", inDim, outDim)) { q4(x) }
            // Dequantize inside the graph, then plain GEMM: bf16 kernel speed
            // at 4-bit storage cost (transient dequant buffer only).
            let dqMs = time(String(format: "dq+mm%5d->%5d", inDim, outDim)) {
                let w = dequantized(
                    q4.weight, scales: q4.scales, biases: q4.biases,
                    groupSize: q4.groupSize, bits: q4.bits
                ).asType(.bfloat16)
                return MLX.matmul(x, w.transposed())
            }
            print(String(format: "[dit-bench] RATIO %5d->%5d q4/bf16=%.2fx dq/bf16=%.2fx", inDim, outDim, q4Ms / fpMs, dqMs / fpMs))
        }
    }

    /// MiniMax-H3's dominant projections at its practical 512-square and
    /// 768x448 packed row counts. The cached arm models loading a compact
    /// checkpoint, dequantizing each projection once, and keeping bf16 weights
    /// resident for the denoising loop.
    func testMiniMaxH3QmmVsResidentBF16() throws {
        try benchGate()

        Stream.withNewDefaultStream {

            func pairedTime(
                _ label: String,
                qmm: QuantizedLinear,
                dense: BenchCachedDenseLinear,
                input: MLXArray
            ) {
                MLX.eval(qmm(input), dense(input))
                var bestQMM = Double.greatestFiniteMagnitude
                var bestDense = Double.greatestFiniteMagnitude
                for _ in 0..<2 {
                    var start = CFAbsoluteTimeGetCurrent()
                    MLX.eval(qmm(input))
                    bestQMM = min(bestQMM, CFAbsoluteTimeGetCurrent() - start)
                    start = CFAbsoluteTimeGetCurrent()
                    MLX.eval(dense(input))
                    bestDense = min(bestDense, CFAbsoluteTimeGetCurrent() - start)
                }
                print(String(
                    format: "[dit-bench] H3 %@ qmm=%.0fms resident-bf16=%.0fms qmm/bf16=%.2fx",
                    label,
                    bestQMM * 1_000,
                    bestDense * 1_000,
                    bestQMM / bestDense
                ))
            }

            let configuredRows = ProcessInfo.processInfo.environment["MERERUN_H3_BENCH_ROWS"]?
                .split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
                .filter { $0 > 0 } ?? []
            let rowCounts = configuredRows.isEmpty ? [4_608, 12_925] : configuredRows
            for rows in rowCounts {
                for (name, inputDimension, outputDimension) in [
                    ("qkv", 5_376, 21_504),
                    ("ff2", 14_336, 5_376),
                ] {
                    let input = MLXRandom.normal([1, rows, inputDimension]).asType(.bfloat16)
                    let qmm = QuantizedLinear(
                        inputDimension,
                        outputDimension,
                        bias: false,
                        groupSize: 64,
                        bits: 4
                    )
                    MLX.eval(input, qmm.parameters())
                    pairedTime(
                        "rows=\(rows) \(name) \(inputDimension)->\(outputDimension)",
                        qmm: qmm,
                        dense: BenchCachedDenseLinear(copying: qmm),
                        input: input
                    )
                    MLX.Memory.clearCache()
                }
            }
        }
    }

    /// Compares H3's current per-token AdaLN gather with Maestro's exact-math
    /// contiguous-run formulation. The latter avoids materializing index
    /// gathers for layouts whose modality/timestep pairs form a handful of
    /// long runs. Both arms produce the same packed tensor and retain a single
    /// projection-sized input; this is a speed experiment, not an approximation.
    func testMiniMaxH3AdaLNRunModulation() throws {
        try benchGate()
        let rows = max(
            1,
            Int(ProcessInfo.processInfo.environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 29_018
        )
        let hiddenSize = 5_376
        let textRows = min(256, rows)
        let conditionRows = min(1_600, max(0, rows - textRows))
        let audioRows = min(840, max(0, rows - textRows - conditionRows))
        let boundaries = [
            0,
            textRows,
            textRows + conditionRows,
            textRows + conditionRows + audioRows,
            rows,
        ]
        let runIndices: [Int32] = [1, 6, 5, 0]
        let runs = zip(boundaries, boundaries.dropFirst()).enumerated().compactMap { offset, pair in
            pair.0 < pair.1 ? (range: pair.0..<pair.1, index: Int(runIndices[offset])) : nil
        }
        let indices = MLXArray(runs.flatMap { run in
            repeatElement(Int32(run.index), count: run.range.count)
        })
        let value = MLXRandom.normal([1, rows, hiddenSize]).asType(.bfloat16)
        let modulation = MLXRandom.normal([9, 3, hiddenSize]).asType(.bfloat16)
        MLX.eval(value, modulation, indices)

        func gathered() -> MLXArray {
            let shift = MLX.take(modulation[0..., 0, 0...], indices, axis: 0)
                .expandedDimensions(axis: 0)
            let scale = MLX.take(modulation[0..., 1, 0...], indices, axis: 0)
                .expandedDimensions(axis: 0)
            let gate = MLX.take(modulation[0..., 2, 0...], indices, axis: 0)
                .expandedDimensions(axis: 0)
            return gate * (value * (1 + scale) + shift)
        }

        func byRuns() -> MLXArray {
            MLX.concatenated(runs.map { run in
                let parameters = modulation[run.index, 0..., 0...]
                let shift = parameters[0, 0...]
                let scale = parameters[1, 0...]
                let gate = parameters[2, 0...]
                let slice = value[0..., run.range, 0...]
                return gate * (slice * (1 + scale) + shift)
            }, axis: 1)
        }

        let gatheredOutput = gathered()
        let runOutput = byRuns()
        MLX.eval(gatheredOutput, runOutput)
        let maximumAbsoluteError = MLX.max(MLX.abs(
            gatheredOutput.asType(.float32) - runOutput.asType(.float32)
        )).item(Float.self)
        XCTAssertEqual(maximumAbsoluteError, 0)

        func measure(_ body: () -> MLXArray) -> Double {
            var best = Double.greatestFiniteMagnitude
            for _ in 0..<3 {
                let started = CFAbsoluteTimeGetCurrent()
                MLX.eval(body())
                best = min(best, CFAbsoluteTimeGetCurrent() - started)
            }
            return best
        }
        let gatheredSeconds = measure(gathered)
        let runSeconds = measure(byRuns)
        print(String(
            format: "[h3-lab] modulation rows=%d hidden=%d runs=%d gather_ms=%.1f run_ms=%.1f speedup=%.3fx max_abs=%.6g",
            rows,
            hiddenSize,
            runs.count,
            gatheredSeconds * 1_000,
            runSeconds * 1_000,
            gatheredSeconds / runSeconds,
            maximumAbsoluteError
        ))
    }

    /// Exact dense MiniMax-H3 attention shape for the 768x448, 124-frame
    /// practical tier. Reports effective QK+AV throughput so attention can be
    /// separated from the already-qualified dense GEMM ceiling.
    func testMiniMaxH3PracticalTierSDPA() throws {
        try benchGate()
        Stream.withNewDefaultStream {
            let rows = max(
                1,
                Int(ProcessInfo.processInfo.environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 12_930
            )
            let heads = 56
            let headDimension = 128
            let query = MLXRandom.normal([1, heads, rows, headDimension]).asType(.bfloat16)
            let key = MLXRandom.normal([1, heads, rows, headDimension]).asType(.bfloat16)
            let value = MLXRandom.normal([1, heads, rows, headDimension]).asType(.bfloat16)
            MLX.eval(query, key, value)

            func attention() -> MLXArray {
                MLXFast.scaledDotProductAttention(
                    queries: query,
                    keys: key,
                    values: value,
                    scale: Float(1 / sqrt(Double(headDimension))),
                    mask: .none
                )
            }

            MLX.eval(attention())
            var best = Double.greatestFiniteMagnitude
            for _ in 0..<3 {
                let start = CFAbsoluteTimeGetCurrent()
                MLX.eval(attention())
                best = min(best, CFAbsoluteTimeGetCurrent() - start)
            }
            let operations = 4 * Double(heads) * Double(rows) * Double(rows) * Double(headDimension)
            print(String(
                format: "[dit-bench] H3 SDPA rows=%d heads=%d dim=%d %.0fms %.2fTFLOP/s",
                rows,
                heads,
                headDimension,
                best * 1_000,
                operations / best / 1e12
            ))
        }
    }

    /// Compares H3's production chunked dense attention with the native Metal
    /// dynamic sparse path, including summary construction, routing, prefix
    /// sink, and centroid correction. Configure with
    /// `MERERUN_H3_SPARSE_BENCH_ROWS`, `MERERUN_H3_SPARSE_BENCH_PREFIX`,
    /// `MERERUN_H3_SPARSE_BENCH_TAU`, `MERERUN_H3_SPARSE_BENCH_DTYPE`, and
    /// `MERERUN_H3_SPARSE_BENCH_ROUNDS`.
    func testMiniMaxH3DynamicSparseAttention() throws {
        try benchGate()
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("MiniMax-H3 dynamic sparse benchmark requires a Metal GPU.")
        }
        let environment = ProcessInfo.processInfo.environment
        let rows = max(
            256,
            Int(environment["MERERUN_H3_SPARSE_BENCH_ROWS"] ?? "") ?? 12_930
        )
        let heads = max(
            1,
            Int(environment["MERERUN_H3_SPARSE_BENCH_HEADS"] ?? "") ?? 56
        )
        let prefix = min(
            rows - 64,
            max(64, Int(environment["MERERUN_H3_SPARSE_BENCH_PREFIX"] ?? "") ?? 951)
        )
        let tau = max(
            0,
            Float(environment["MERERUN_H3_SPARSE_BENCH_TAU"] ?? "") ?? 1
        )
        let queryChunk = max(
            1,
            Int(environment["MERERUN_H3_SPARSE_BENCH_QUERY_TOKENS"] ?? "") ?? 640
        )
        let rounds = max(
            1,
            Int(environment["MERERUN_H3_SPARSE_BENCH_ROUNDS"] ?? "") ?? 2
        )
        let dtype: DType = environment["MERERUN_H3_SPARSE_BENCH_DTYPE"] == "fp32"
            ? .float32
            : .bfloat16
        let dimension = MiniMaxH3DynamicSparseAttention.headDimension
        let scale = 1 / sqrt(Float(dimension))
        MLXRandom.seed(2_026)
        let shape = [1, heads, rows, dimension]
        let queries = MLXRandom.normal(shape).asType(dtype)
        let keys = MLXRandom.normal(shape).asType(dtype)
        let values = MLXRandom.normal(shape).asType(dtype)
        MLX.eval(queries, keys, values)

        let request = MiniMaxH3DynamicSparseAttentionRequest(
            prefixTokenCount: prefix,
            thresholdStandardDeviations: tau
        )
        func sparse() -> MLXArray {
            MiniMaxH3DynamicSparseAttention.call(
                queries: queries,
                keys: keys,
                values: values,
                request: request,
                scale: scale,
                maximumQueryTokens: queryChunk,
                maximumKernelsPerEvaluation: 1
            )!
        }
        func dense() -> MLXArray {
            var outputs: [MLXArray] = []
            for start in stride(from: 0, to: rows, by: queryChunk) {
                let end = min(start + queryChunk, rows)
                let output = MLXFast.scaledDotProductAttention(
                    queries: queries[0..., 0..., start..<end, 0...],
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: .none
                )
                MLX.eval(output)
                outputs.append(output)
            }
            return MLX.concatenated(outputs, axis: 2)
        }

        let gate = try XCTUnwrap(MiniMaxH3DynamicSparseAttention.denseRouteGate(
            queries: queries,
            keys: keys,
            values: values,
            queryStart: prefix,
            scale: scale
        ))
        XCTAssertTrue(gate.passed)
        let plan = MiniMaxH3DynamicSparseAttention.makeRoutePlan(
            queries: queries,
            keys: keys,
            values: values,
            queryStart: prefix,
            thresholdStandardDeviations: tau,
            scale: scale
        )
        let routeDensity = plan.routes.asType(.float32).mean().item(Float.self)

        MLX.eval(dense(), sparse())
        var denseTotal = 0.0
        var sparseTotal = 0.0
        for round in 0..<rounds {
            if round.isMultiple(of: 2) {
                var started = CFAbsoluteTimeGetCurrent()
                MLX.eval(dense())
                denseTotal += CFAbsoluteTimeGetCurrent() - started
                started = CFAbsoluteTimeGetCurrent()
                MLX.eval(sparse())
                sparseTotal += CFAbsoluteTimeGetCurrent() - started
            } else {
                var started = CFAbsoluteTimeGetCurrent()
                MLX.eval(sparse())
                sparseTotal += CFAbsoluteTimeGetCurrent() - started
                started = CFAbsoluteTimeGetCurrent()
                MLX.eval(dense())
                denseTotal += CFAbsoluteTimeGetCurrent() - started
            }
        }
        let denseSeconds = denseTotal / Double(rounds)
        let sparseSeconds = sparseTotal / Double(rounds)
        let reference = dense().asType(.float32)
        let candidate = sparse().asType(.float32)
        let delta = candidate - reference
        let maximumAbsoluteError = MLX.max(MLX.abs(delta)).item(Float.self)
        let relativeL2Error = MLX.sqrt(
            MLX.sum(delta * delta)
                / MLX.maximum(MLX.sum(reference * reference), MLXArray(Float(1e-12)))
        ).item(Float.self)
        print(String(
            format: "[h3-lab] dynamic-sparse rows=%d prefix=%d heads=%d dtype=%@ tau=%.2f "
                + "route_density=%.4f dense_ms=%.0f sparse_ms=%.0f speedup=%.3fx "
                + "max_abs=%.6g rel_l2=%.6g gate_rel_l2=%.6g",
            rows,
            prefix,
            heads,
            String(describing: dtype),
            tau,
            routeDensity,
            denseSeconds * 1_000,
            sparseSeconds * 1_000,
            denseSeconds / sparseSeconds,
            maximumAbsoluteError,
            relativeL2Error,
            gate.relativeL2Error
        ))
        XCTAssertTrue(relativeL2Error.isFinite)
    }

    /// Searches exact H3 attention schedules across query chunks, head chunks,
    /// and Metal evaluation batches. Every candidate is compared numerically
    /// with the production 2,048-query/56-head/4-kernel schedule; this is a
    /// kernel-scheduling search, not an attention approximation.
    ///
    /// Configure the frontier with:
    ///
    /// - `MERERUN_H3_BENCH_ROWS=14958,37966`
    /// - `MERERUN_H3_BENCH_CHUNKS=1024,1536,2048,2560,3072,4096`
    /// - `MERERUN_H3_BENCH_HEAD_CHUNKS=14,28,56`
    /// - `MERERUN_H3_BENCH_EVAL_BATCHES=1,2,4,8`
    /// - `MERERUN_H3_BENCH_ROUNDS=2`
    /// - `MERERUN_H3_BENCH_SEARCH=coordinate` for a fast three-axis search,
    ///   or `grid` for the complete Cartesian frontier
    /// - `MERERUN_H3_BENCH_FAST=1` to share one paired baseline and defer
    ///   validation while searching; this keeps the full key/value length but
    ///   samples 8,192 query rows by default, then extrapolates the full pass
    /// - `MERERUN_H3_BENCH_SAMPLE_QUERY_ROWS=8192` to tune that fast sample
    ///   size; rerun the winner without fast mode for full-pass parity
    /// - `MERERUN_H3_BENCH_VALIDATE=0` to skip the exactness calculation
    /// - `MERERUN_H3_BENCH_DTYPE=fp16` to compare Metal's FP16 attention path
    ///   with the production-default BF16 path
    func testMiniMaxH3AttentionChunkSizes() throws {
        try benchGate()
        let environment = ProcessInfo.processInfo.environment
        let benchmarkDType: DType = environment["MERERUN_H3_BENCH_DTYPE"] == "fp16"
            ? .float16
            : .bfloat16
        let benchmarkDTypeLabel = benchmarkDType == .float16 ? "fp16" : "bf16"

        func configuredIntegers(_ key: String) -> [Int] {
            environment[key]?
                .split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
                .filter { $0 > 0 } ?? []
        }

        let configuredRows = configuredIntegers("MERERUN_H3_BENCH_ROWS")
        let rowCounts = configuredRows.isEmpty ? [14_958] : configuredRows
        let configuredChunks = configuredIntegers("MERERUN_H3_BENCH_CHUNKS")
        let chunkSizes = configuredChunks.isEmpty
            ? [1_024, 1_536, 2_048, 2_560, 3_072, 4_096, 8_192]
            : configuredChunks
        let configuredHeadChunks = configuredIntegers("MERERUN_H3_BENCH_HEAD_CHUNKS")
        let headChunkSizes = configuredHeadChunks.isEmpty ? [14, 28, 56] : configuredHeadChunks
        let configuredEvaluationBatches = configuredIntegers("MERERUN_H3_BENCH_EVAL_BATCHES")
        let evaluationBatchSizes = configuredEvaluationBatches.isEmpty
            ? [1, 2, 4, 8]
            : configuredEvaluationBatches
        let rounds = max(1, Int(environment["MERERUN_H3_BENCH_ROUNDS"] ?? "") ?? 2)
        let searchMode = environment["MERERUN_H3_BENCH_SEARCH"] ?? "coordinate"
        let usesFastSearch = environment["MERERUN_H3_BENCH_FAST"] == "1"
        let validatesExactness = environment["MERERUN_H3_BENCH_VALIDATE"] != "0"

        Stream.withNewDefaultStream {
            let heads = 56
            let headDimension = 128
            let scale = Float(1 / sqrt(Double(headDimension)))

            func attention(
                query: MLXArray,
                key: MLXArray,
                value: MLXArray,
                queryChunkSize: Int,
                headChunkSize: Int,
                evaluationBatchSize: Int
            ) -> MLXArray {
                var headOutputs: [MLXArray] = []
                var pending: [MLXArray] = []
                for headStart in stride(from: 0, to: query.dim(1), by: headChunkSize) {
                    let headEnd = min(headStart + headChunkSize, query.dim(1))
                    var queryOutputs: [MLXArray] = []
                    for queryStart in stride(from: 0, to: query.dim(2), by: queryChunkSize) {
                        let queryEnd = min(queryStart + queryChunkSize, query.dim(2))
                        let chunk = MLXFast.scaledDotProductAttention(
                            queries: query[0..., headStart..<headEnd, queryStart..<queryEnd, 0...],
                            keys: key[0..., headStart..<headEnd, 0..., 0...],
                            values: value[0..., headStart..<headEnd, 0..., 0...],
                            scale: scale,
                            mask: .none
                        )
                        queryOutputs.append(chunk)
                        pending.append(chunk)
                        if pending.count == evaluationBatchSize {
                            MLX.eval(pending)
                            pending.removeAll(keepingCapacity: true)
                        }
                    }
                    headOutputs.append(MLX.concatenated(queryOutputs, axis: 2))
                }
                if !pending.isEmpty {
                    MLX.eval(pending)
                }
                return MLX.concatenated(headOutputs, axis: 1)
            }

            for rows in rowCounts {
                let configuredSampleRows = max(
                    1,
                    Int(environment["MERERUN_H3_BENCH_SAMPLE_QUERY_ROWS"] ?? "") ?? 8_192
                )
                let queryRows = usesFastSearch ? min(rows, configuredSampleRows) : rows
                let query = MLXRandom.normal([1, heads, queryRows, headDimension])
                    .asType(benchmarkDType)
                let key = MLXRandom.normal([1, heads, rows, headDimension])
                    .asType(benchmarkDType)
                let value = MLXRandom.normal([1, heads, rows, headDimension])
                    .asType(benchmarkDType)
                MLX.eval(query, key, value)

                let baseline = {
                    attention(
                        query: query,
                        key: key,
                        value: value,
                        queryChunkSize: min(2_048, rows),
                        headChunkSize: heads,
                        evaluationBatchSize: 4
                    )
                }
                let baselineOutput = baseline()
                MLX.eval(baselineOutput)
                let baseline32 = validatesExactness ? baselineOutput.asType(.float32) : nil
                if let baseline32 { MLX.eval(baseline32) }

                struct AttentionSchedule: Hashable {
                    let queryChunkSize: Int
                    let headChunkSize: Int
                    let evaluationBatchSize: Int
                }

                struct Measurement {
                    let candidateSeconds: Double
                    let baselineSeconds: Double

                    var speedup: Double { baselineSeconds / candidateSeconds }
                }

                var measured: [AttentionSchedule: Measurement] = [:]
                var bestSeconds = Double.greatestFiniteMagnitude
                var bestSpeedup = 0.0
                var bestDescription = ""

                func elapsed(_ body: () -> MLXArray) -> Double {
                    let started = CFAbsoluteTimeGetCurrent()
                    MLX.eval(body())
                    return CFAbsoluteTimeGetCurrent() - started
                }
                let sharedBaselineSeconds: Double? = usesFastSearch ? elapsed(baseline) : nil

                func measure(_ schedule: AttentionSchedule) -> Measurement {
                    if let measurement = measured[schedule] { return measurement }
                    Memory.peakMemory = 0
                    let candidate = {
                        attention(
                            query: query,
                            key: key,
                            value: value,
                            queryChunkSize: schedule.queryChunkSize,
                            headChunkSize: schedule.headChunkSize,
                            evaluationBatchSize: schedule.evaluationBatchSize
                        )
                    }
                    let measurement: Measurement
                    let output: MLXArray?
                    if let sharedBaselineSeconds {
                        MLX.eval(candidate())
                        measurement = Measurement(
                            candidateSeconds: elapsed(candidate),
                            baselineSeconds: sharedBaselineSeconds
                        )
                        output = nil
                    } else {
                        MLX.eval(baseline(), candidate())
                        var baselineSeconds = 0.0
                        var candidateSeconds = 0.0
                        for round in 0..<rounds {
                            if round.isMultiple(of: 2) {
                                baselineSeconds += elapsed(baseline)
                                candidateSeconds += elapsed(candidate)
                            } else {
                                candidateSeconds += elapsed(candidate)
                                baselineSeconds += elapsed(baseline)
                            }
                        }
                        measurement = Measurement(
                            candidateSeconds: candidateSeconds / Double(rounds),
                            baselineSeconds: baselineSeconds / Double(rounds)
                        )
                        let candidateOutput = candidate()
                        MLX.eval(candidateOutput)
                        output = candidateOutput
                    }
                    let peakGiB = Double(Memory.peakMemory) / 1_073_741_824
                    var maximumAbsoluteError = 0.0
                    var relativeL2Error = 0.0
                    if let baseline32, let output {
                        let delta = output.asType(.float32) - baseline32
                        let errorSquared = MLX.sum(delta * delta).item(Float.self)
                        let referenceSquared = MLX.sum(baseline32 * baseline32).item(Float.self)
                        maximumAbsoluteError = Double(MLX.max(MLX.abs(delta)).item(Float.self))
                        relativeL2Error = sqrt(
                            Double(errorSquared) / max(Double(referenceSquared), .leastNonzeroMagnitude)
                        )
                        XCTAssertLessThan(relativeL2Error, 1e-3)
                    }
                    let operations = 4 * Double(heads) * Double(queryRows)
                        * Double(rows) * Double(headDimension)
                    let fullPassScale = Double(rows) / Double(queryRows)
                    print(String(
                        format: "[h3-lab] attention dtype=%@ rows=%d sampled_queries=%d query=%d "
                            + "heads=%d batch=%d sample_kernels=%d full_kernels=%d "
                            + "sample_ms=%.0f estimated_full_ms=%.0f paired_base_ms=%.0f "
                            + "speedup=%.3fx tflops=%.2f max_abs=%.6g rel_l2=%.6g peak_gib=%.2f",
                        benchmarkDTypeLabel,
                        rows,
                        queryRows,
                        schedule.queryChunkSize,
                        schedule.headChunkSize,
                        schedule.evaluationBatchSize,
                        ((queryRows + schedule.queryChunkSize - 1) / schedule.queryChunkSize)
                            * ((heads + schedule.headChunkSize - 1) / schedule.headChunkSize),
                        ((rows + schedule.queryChunkSize - 1) / schedule.queryChunkSize)
                            * ((heads + schedule.headChunkSize - 1) / schedule.headChunkSize),
                        measurement.candidateSeconds * 1_000,
                        measurement.candidateSeconds * fullPassScale * 1_000,
                        measurement.baselineSeconds * 1_000,
                        measurement.speedup,
                        operations / measurement.candidateSeconds / 1e12,
                        maximumAbsoluteError,
                        relativeL2Error,
                        peakGiB
                    ))
                    measured[schedule] = measurement
                    if measurement.speedup > bestSpeedup {
                        bestSeconds = measurement.candidateSeconds
                        bestSpeedup = measurement.speedup
                        bestDescription = "query=\(schedule.queryChunkSize) "
                            + "heads=\(schedule.headChunkSize) batch=\(schedule.evaluationBatchSize)"
                    }
                    Memory.clearCache()
                    return measurement
                }

                func fastest(_ schedules: [AttentionSchedule]) -> AttentionSchedule {
                    precondition(!schedules.isEmpty)
                    var winner = schedules[0]
                    var winnerSpeedup = measure(winner).speedup
                    for schedule in schedules.dropFirst() {
                        let speedup = measure(schedule).speedup
                        if speedup > winnerSpeedup {
                            winner = schedule
                            winnerSpeedup = speedup
                        }
                    }
                    return winner
                }

                if searchMode == "grid" {
                    for requestedChunkSize in chunkSizes {
                        for requestedHeadChunkSize in headChunkSizes {
                            for evaluationBatchSize in evaluationBatchSizes {
                                _ = measure(AttentionSchedule(
                                    queryChunkSize: min(requestedChunkSize, rows),
                                    headChunkSize: min(requestedHeadChunkSize, heads),
                                    evaluationBatchSize: evaluationBatchSize
                                ))
                            }
                        }
                    }
                } else {
                    let queryWinner = fastest(chunkSizes.map {
                        AttentionSchedule(
                            queryChunkSize: min($0, rows),
                            headChunkSize: heads,
                            evaluationBatchSize: 4
                        )
                    })
                    let headWinner = fastest(headChunkSizes.map {
                        AttentionSchedule(
                            queryChunkSize: queryWinner.queryChunkSize,
                            headChunkSize: min($0, heads),
                            evaluationBatchSize: 4
                        )
                    })
                    _ = fastest(evaluationBatchSizes.map {
                        AttentionSchedule(
                            queryChunkSize: headWinner.queryChunkSize,
                            headChunkSize: headWinner.headChunkSize,
                            evaluationBatchSize: $0
                        )
                    })
                }
                print(String(
                    format: "[h3-lab] winner dtype=%@ rows=%d sampled_queries=%d search=%@ "
                        + "%@ estimated_full_ms=%.0f speedup=%.3fx candidates=%d",
                    benchmarkDTypeLabel,
                    rows,
                    queryRows,
                    searchMode,
                    bestDescription,
                    bestSeconds * Double(rows) / Double(queryRows) * 1_000,
                    bestSpeedup,
                    measured.count
                ))
                Memory.clearCache()
            }
        }
    }

    /// Laboratory sweep for the non-NAX MLX Steel GEMM tile and threadgroup
    /// swizzle at H3's true-768 QKV projection shape.
    func testMiniMaxH3MetalGEMMTiles() throws {
        try benchGate()
        let environment = ProcessInfo.processInfo.environment
        let rows = max(1, Int(environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 37_966)

        struct Tile: Hashable {
            let blockRows: Int
            let blockColumns: Int
            let blockDepth: Int
            let rowWarps: Int
            let columnWarps: Int
        }
        struct Arm: Hashable {
            let tile: Tile
            let swizzle: Int
        }

        let defaultTile = Tile(
            blockRows: 64,
            blockColumns: 64,
            blockDepth: 16,
            rowWarps: 1,
            columnWarps: 2
        )
        let tiles = [
            defaultTile,
            Tile(blockRows: 64, blockColumns: 64, blockDepth: 16, rowWarps: 2, columnWarps: 2),
            Tile(blockRows: 64, blockColumns: 32, blockDepth: 32, rowWarps: 2, columnWarps: 2),
            Tile(blockRows: 32, blockColumns: 64, blockDepth: 16, rowWarps: 1, columnWarps: 2),
            Tile(blockRows: 32, blockColumns: 32, blockDepth: 16, rowWarps: 2, columnWarps: 2),
            Tile(blockRows: 64, blockColumns: 32, blockDepth: 8, rowWarps: 4, columnWarps: 1),
        ]
        let arms = tiles.flatMap { tile in
            (0...3).map { Arm(tile: tile, swizzle: $0) }
        }

        func select(_ arm: Arm) {
            setenv("MLX_GEMM_H3_TUNED", "0", 1)
            setenv("MLX_GEMM_BM", String(arm.tile.blockRows), 1)
            setenv("MLX_GEMM_BN", String(arm.tile.blockColumns), 1)
            setenv("MLX_GEMM_BK", String(arm.tile.blockDepth), 1)
            setenv("MLX_GEMM_WM", String(arm.tile.rowWarps), 1)
            setenv("MLX_GEMM_WN", String(arm.tile.columnWarps), 1)
            setenv("MLX_GEMM_SWIZZLE_LOG", String(arm.swizzle), 1)
        }

        let defaultArm = Arm(tile: defaultTile, swizzle: 0)
        defer { select(defaultArm) }

        Stream.withNewDefaultStream {
            let inputDimension = 5_376
            let outputDimension = 21_504
            let input = MLXRandom.normal([1, rows, inputDimension]).asType(.bfloat16)
            let weight = MLXRandom.normal([outputDimension, inputDimension]).asType(.bfloat16)
            MLX.eval(input, weight)

            func projection() -> MLXArray {
                MLX.matmul(input, weight.T)
            }

            select(defaultArm)
            let reference = projection()
            MLX.eval(reference)
            let referenceSample = reference[0..., 0..<128, 0...].asType(.float32)
            MLX.eval(referenceSample)

            for tile in tiles {
                let arm = Arm(tile: tile, swizzle: 0)
                select(arm)
                let output = projection()
                MLX.eval(output)
                let delta = output[0..., 0..<128, 0...].asType(.float32) - referenceSample
                let errorSquared = MLX.sum(delta * delta).item(Float.self)
                let referenceSquared = MLX.sum(referenceSample * referenceSample).item(Float.self)
                let maximumAbsoluteError = MLX.max(MLX.abs(delta)).item(Float.self)
                let relativeL2Error = sqrt(
                    Double(errorSquared) / max(Double(referenceSquared), .leastNonzeroMagnitude)
                )
                XCTAssertLessThan(relativeL2Error, 1e-3)
                print(String(
                    format: "[h3-lab] gemm-parity bm=%d bn=%d bk=%d wm=%d wn=%d "
                        + "max_abs=%.6g rel_l2=%.6g",
                    tile.blockRows,
                    tile.blockColumns,
                    tile.blockDepth,
                    tile.rowWarps,
                    tile.columnWarps,
                    maximumAbsoluteError,
                    relativeL2Error
                ))
            }

            var totals = Dictionary(uniqueKeysWithValues: arms.map { ($0, 0.0) })
            for orderedArms in [arms, Array(arms.reversed())] {
                for arm in orderedArms {
                    select(arm)
                    let started = CFAbsoluteTimeGetCurrent()
                    MLX.eval(projection())
                    totals[arm, default: 0] += CFAbsoluteTimeGetCurrent() - started
                }
            }

            let operations = 2 * Double(rows) * Double(inputDimension) * Double(outputDimension)
            for arm in arms {
                let seconds = totals[arm, default: 0] / 2
                print(String(
                    format: "[h3-lab] gemm-tile bm=%d bn=%d bk=%d wm=%d wn=%d "
                        + "swizzle=%d mean_ms=%.1f tflops=%.2f",
                    arm.tile.blockRows,
                    arm.tile.blockColumns,
                    arm.tile.blockDepth,
                    arm.tile.rowWarps,
                    arm.tile.columnWarps,
                    arm.swizzle,
                    seconds * 1_000,
                    operations / seconds / 1e12
                ))
            }
        }
    }

    /// Confirms the QKV winner against every dense projection in a production
    /// H3 block before any MLX heuristic changes are considered. Set
    /// `MERERUN_H3_BENCH_ROWS` to retune a longer packed sequence.
    func testMiniMaxH3MetalGEMMProjectionShapes() throws {
        try benchGate()
        let environment = ProcessInfo.processInfo.environment
        let rows = max(1, Int(environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 37_966)

        struct Arm {
            let name: String
            let blockRows: Int
            let blockColumns: Int
            let blockDepth: Int
            let rowWarps: Int
            let columnWarps: Int
            let swizzle: Int
        }

        let arms = [
            Arm(name: "default", blockRows: 64, blockColumns: 64, blockDepth: 16,
                rowWarps: 1, columnWarps: 2, swizzle: 0),
            Arm(name: "default-sw1", blockRows: 64, blockColumns: 64, blockDepth: 16,
                rowWarps: 1, columnWarps: 2, swizzle: 1),
            Arm(name: "four-warp-sw1", blockRows: 64, blockColumns: 64, blockDepth: 16,
                rowWarps: 2, columnWarps: 2, swizzle: 1),
            Arm(name: "four-warp-sw2", blockRows: 64, blockColumns: 64, blockDepth: 16,
                rowWarps: 2, columnWarps: 2, swizzle: 2),
            Arm(name: "four-warp-sw3", blockRows: 64, blockColumns: 64, blockDepth: 16,
                rowWarps: 2, columnWarps: 2, swizzle: 3),
            Arm(name: "narrow-m-sw1", blockRows: 32, blockColumns: 64, blockDepth: 16,
                rowWarps: 1, columnWarps: 2, swizzle: 1),
            Arm(name: "narrow-m-sw2", blockRows: 32, blockColumns: 64, blockDepth: 16,
                rowWarps: 1, columnWarps: 2, swizzle: 2),
            Arm(name: "narrow-m-sw3", blockRows: 32, blockColumns: 64, blockDepth: 16,
                rowWarps: 1, columnWarps: 2, swizzle: 3),
        ]
        let shapes = [
            (name: "qkv", input: 5_376, output: 21_504),
            (name: "attention-out", input: 7_168, output: 5_376),
            (name: "ff-in", input: 5_376, output: 28_672),
            (name: "ff-out", input: 14_336, output: 5_376),
        ]

        func select(_ arm: Arm) {
            setenv("MLX_GEMM_H3_TUNED", "0", 1)
            setenv("MLX_GEMM_BM", String(arm.blockRows), 1)
            setenv("MLX_GEMM_BN", String(arm.blockColumns), 1)
            setenv("MLX_GEMM_BK", String(arm.blockDepth), 1)
            setenv("MLX_GEMM_WM", String(arm.rowWarps), 1)
            setenv("MLX_GEMM_WN", String(arm.columnWarps), 1)
            setenv("MLX_GEMM_SWIZZLE_LOG", String(arm.swizzle), 1)
        }

        defer { select(arms[0]) }

        Stream.withNewDefaultStream {
            for shape in shapes {
                let input = MLXRandom.normal([1, rows, shape.input]).asType(.bfloat16)
                let weight = MLXRandom.normal([shape.output, shape.input]).asType(.bfloat16)
                MLX.eval(input, weight)

                func projection() -> MLXArray {
                    MLX.matmul(input, weight.T)
                }

                select(arms[0])
                let reference = projection()
                MLX.eval(reference)
                let referenceSample = reference[0..., 0..<128, 0...].asType(.float32)
                MLX.eval(referenceSample)

                var totals = Dictionary(uniqueKeysWithValues: arms.map { ($0.name, 0.0) })
                for orderedArms in [arms, Array(arms.reversed())] {
                    for arm in orderedArms {
                        select(arm)
                        let started = CFAbsoluteTimeGetCurrent()
                        let output = projection()
                        MLX.eval(output)
                        totals[arm.name, default: 0] += CFAbsoluteTimeGetCurrent() - started

                        let delta = output[0..., 0..<128, 0...].asType(.float32) - referenceSample
                        let errorSquared = MLX.sum(delta * delta).item(Float.self)
                        let referenceSquared = MLX.sum(referenceSample * referenceSample).item(Float.self)
                        let relativeL2Error = sqrt(
                            Double(errorSquared) / max(Double(referenceSquared), .leastNonzeroMagnitude)
                        )
                        XCTAssertLessThan(relativeL2Error, 1e-3)
                    }
                }

                let operations = 2 * Double(rows) * Double(shape.input) * Double(shape.output)
                for arm in arms {
                    let seconds = totals[arm.name, default: 0] / 2
                    print(String(
                        format: "[h3-lab] gemm-shape %@ %@ %d->%d mean_ms=%.1f tflops=%.2f",
                        shape.name,
                        arm.name,
                        shape.input,
                        shape.output,
                        seconds * 1_000,
                        operations / seconds / 1e12
                    ))
                }
            }
        }
    }

    /// Paired whole-block confirmation for the default and candidate GEMM
    /// schedules so thermal drift hits both arms symmetrically.
    func testMiniMaxH3BlockGEMMSchedules() throws {
        try benchGate()
        let rows = max(
            1,
            Int(ProcessInfo.processInfo.environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 37_966
        )
        let benchmark = MiniMaxH3BlockScheduleBenchmark(
            rowCount: rows,
            maximumQueryTokens: 1_024,
            maximumKernelsPerEvaluation: 4
        )

        func selectDefault() {
            setenv("MLX_GEMM_H3_TUNED", "0", 1)
            setenv("MLX_GEMM_BM", "64", 1)
            setenv("MLX_GEMM_BN", "64", 1)
            setenv("MLX_GEMM_BK", "16", 1)
            setenv("MLX_GEMM_WM", "1", 1)
            setenv("MLX_GEMM_WN", "2", 1)
            setenv("MLX_GEMM_SWIZZLE_LOG", "0", 1)
        }

        func selectCandidate() {
            setenv("MLX_GEMM_H3_TUNED", "0", 1)
            setenv("MLX_GEMM_BM", "64", 1)
            setenv("MLX_GEMM_BN", "64", 1)
            setenv("MLX_GEMM_BK", "16", 1)
            setenv("MLX_GEMM_WM", "2", 1)
            setenv("MLX_GEMM_WN", "2", 1)
            setenv("MLX_GEMM_SWIZZLE_LOG", "2", 1)
        }

        func selectHybrid() {
            selectDefault()
            unsetenv("MLX_GEMM_H3_TUNED")
        }

        defer { selectDefault() }

        selectDefault()
        let reference = benchmark(schedule: .splitPostAttention)
        MLX.eval(reference)
        selectCandidate()
        let candidate = benchmark(schedule: .splitPostAttention)
        MLX.eval(candidate)
        selectHybrid()
        let hybrid = benchmark(schedule: .splitPostAttention)
        MLX.eval(hybrid)
        let reference32 = reference.asType(.float32)
        MLX.eval(reference32)
        func relativeL2(_ output: MLXArray) -> Double {
            let delta = output.asType(.float32) - reference32
            let errorSquared = MLX.sum(delta * delta).item(Float.self)
            let referenceSquared = MLX.sum(reference32 * reference32).item(Float.self)
            return sqrt(Double(errorSquared) / max(Double(referenceSquared), .leastNonzeroMagnitude))
        }
        let candidateRelativeL2Error = relativeL2(candidate)
        let hybridRelativeL2Error = relativeL2(hybrid)
        XCTAssertLessThan(candidateRelativeL2Error, 1e-3)
        XCTAssertLessThan(hybridRelativeL2Error, 1e-3)

        var totals = [Double](repeating: 0, count: 3)
        let rounds = 3
        for round in 0..<rounds {
            let order: [(select: () -> Void, arm: Int)]
            switch round {
            case 0:
                order = [(selectDefault, 0), (selectCandidate, 1), (selectHybrid, 2)]
            case 1:
                order = [(selectCandidate, 1), (selectHybrid, 2), (selectDefault, 0)]
            default:
                order = [(selectHybrid, 2), (selectDefault, 0), (selectCandidate, 1)]
            }
            for arm in order {
                arm.select()
                let started = CFAbsoluteTimeGetCurrent()
                MLX.eval(benchmark(schedule: .splitPostAttention))
                totals[arm.arm] += CFAbsoluteTimeGetCurrent() - started
            }
        }
        totals = totals.map { $0 / Double(rounds) }
        print(String(
            format: "[h3-lab] block-gemm rows=%d default_ms=%.0f generic_ms=%.0f "
                + "hybrid_ms=%.0f generic_speedup=%.3fx hybrid_speedup=%.3fx "
                + "generic_rel_l2=%.6g hybrid_rel_l2=%.6g",
            rows,
            totals[0] * 1_000,
            totals[1] * 1_000,
            totals[2] * 1_000,
            totals[0] / totals[1],
            totals[0] / totals[2],
            candidateRelativeL2Error,
            hybridRelativeL2Error
        ))
    }

    /// Measures the exact compiled schedules used by one production H3 block.
    /// This catches graph-boundary and synchronization costs that isolated GEMM
    /// and SDPA probes cannot see. Use 37,966 rows for the true-768 target.
    func testMiniMaxH3BlockExecutionSchedules() throws {
        try benchGate()
        let environment = ProcessInfo.processInfo.environment
        let rows = max(1, Int(environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 14_958)
        let queryTokens = max(
            1,
            Int(environment["MERERUN_H3_BENCH_QUERY_TOKENS"] ?? "") ?? 1_024
        )
        let evaluationBatch = max(
            1,
            Int(environment["MERERUN_H3_BENCH_EVAL_BATCH"] ?? "") ?? 4
        )
        let headsPerKernel = max(
            1,
            Int(environment["MERERUN_H3_BENCH_HEADS"] ?? "") ?? 56
        )
        let rounds = max(1, Int(environment["MERERUN_H3_BENCH_ROUNDS"] ?? "") ?? 2)
        let benchmark = MiniMaxH3BlockScheduleBenchmark(
            rowCount: rows,
            maximumQueryTokens: queryTokens,
            maximumKernelsPerEvaluation: evaluationBatch,
            maximumHeadsPerKernel: headsPerKernel
        )

        func elapsed(_ schedule: MiniMaxH3BlockScheduleBenchmark.Schedule) -> Double {
            let started = CFAbsoluteTimeGetCurrent()
            MLX.eval(benchmark(schedule: schedule))
            return CFAbsoluteTimeGetCurrent() - started
        }

        MLX.eval(benchmark(schedule: .splitPostAttention))
        MLX.eval(benchmark(schedule: .fusedFeedForward))
        MLX.eval(benchmark(schedule: .fusedPostAttention))

        let splitOutput = benchmark(schedule: .splitPostAttention).asType(.float32)
        let feedForwardOutput = benchmark(schedule: .fusedFeedForward).asType(.float32)
        let fusedOutput = benchmark(schedule: .fusedPostAttention).asType(.float32)
        MLX.eval(splitOutput, feedForwardOutput, fusedOutput)
        let feedForwardDelta = feedForwardOutput - splitOutput
        let fusedDelta = fusedOutput - splitOutput
        let referenceSquared = MLX.sum(splitOutput * splitOutput).item(Float.self)
        let feedForwardRelativeL2Error = sqrt(
            Double(MLX.sum(feedForwardDelta * feedForwardDelta).item(Float.self))
                / max(Double(referenceSquared), .leastNonzeroMagnitude)
        )
        let fusedRelativeL2Error = sqrt(
            Double(MLX.sum(fusedDelta * fusedDelta).item(Float.self))
                / max(Double(referenceSquared), .leastNonzeroMagnitude)
        )
        XCTAssertLessThan(feedForwardRelativeL2Error, 1e-3)
        XCTAssertLessThan(fusedRelativeL2Error, 1e-3)

        var splitBest = Double.greatestFiniteMagnitude
        var feedForwardBest = Double.greatestFiniteMagnitude
        var fusedBest = Double.greatestFiniteMagnitude
        Memory.peakMemory = 0
        for round in 0..<rounds {
            switch round % 3 {
            case 0:
                splitBest = min(splitBest, elapsed(.splitPostAttention))
                feedForwardBest = min(feedForwardBest, elapsed(.fusedFeedForward))
                fusedBest = min(fusedBest, elapsed(.fusedPostAttention))
            case 1:
                feedForwardBest = min(feedForwardBest, elapsed(.fusedFeedForward))
                fusedBest = min(fusedBest, elapsed(.fusedPostAttention))
                splitBest = min(splitBest, elapsed(.splitPostAttention))
            default:
                fusedBest = min(fusedBest, elapsed(.fusedPostAttention))
                splitBest = min(splitBest, elapsed(.splitPostAttention))
                feedForwardBest = min(feedForwardBest, elapsed(.fusedFeedForward))
            }
        }

        let projectedAttention = benchmark.projectAttention()
        MLX.eval(projectedAttention)
        let attended = benchmark.attend(projectedAttention)
        MLX.eval(attended)
        func phaseElapsed(_ body: () -> Void) -> Double {
            var best = Double.greatestFiniteMagnitude
            for _ in 0..<rounds {
                let started = CFAbsoluteTimeGetCurrent()
                body()
                best = min(best, CFAbsoluteTimeGetCurrent() - started)
            }
            return best
        }
        let attentionProjectionSeconds = phaseElapsed {
            MLX.eval(benchmark.projectAttention())
        }
        let attentionSeconds = phaseElapsed {
            MLX.eval(benchmark.attend(projectedAttention))
        }
        let splitPostAttentionSeconds = phaseElapsed {
            MLX.eval(benchmark.postAttention(
                schedule: .splitPostAttention,
                attended: attended,
                gate: projectedAttention[3]
            ))
        }
        let fusedFeedForwardSeconds = phaseElapsed {
            MLX.eval(benchmark.postAttention(
                schedule: .fusedFeedForward,
                attended: attended,
                gate: projectedAttention[3]
            ))
        }
        let fusedPostAttentionSeconds = phaseElapsed {
            MLX.eval(benchmark.postAttention(
                schedule: .fusedPostAttention,
                attended: attended,
                gate: projectedAttention[3]
            ))
        }

        let configuration = MiniMaxH3TransformerConfiguration()
        let projectionOperations = 2 * Double(rows) * Double(configuration.hiddenSize)
            * Double(
                4 * configuration.attentionHeadCount * configuration.attentionHeadDimension
                    + 3 * configuration.feedForwardSize
            )
        let attentionOperations = 4 * Double(configuration.attentionHeadCount) * Double(rows)
            * Double(rows) * Double(configuration.attentionHeadDimension)
        let operations = projectionOperations + attentionOperations
        let peakGiB = Double(Memory.peakMemory) / 1_073_741_824
        print(String(
            format: "[h3-lab] block rows=%d query=%d heads=%d batch=%d split_ms=%.0f "
                + "ff_fused_ms=%.0f fused_ms=%.0f ff_fused_speedup=%.3fx "
                + "fused_speedup=%.3fx split_tflops=%.2f ff_fused_tflops=%.2f "
                + "fused_tflops=%.2f ff_fused_rel_l2=%.6g fused_rel_l2=%.6g peak_gib=%.2f",
            rows,
            queryTokens,
            headsPerKernel,
            evaluationBatch,
            splitBest * 1_000,
            feedForwardBest * 1_000,
            fusedBest * 1_000,
            splitBest / feedForwardBest,
            splitBest / fusedBest,
            operations / splitBest / 1e12,
            operations / feedForwardBest / 1e12,
            operations / fusedBest / 1e12,
            feedForwardRelativeL2Error,
            fusedRelativeL2Error,
            peakGiB
        ))
        print(String(
            format: "[h3-lab] block-phases rows=%d attention_projection_ms=%.0f "
                + "attention_ms=%.0f split_post_ms=%.0f ff_fused_post_ms=%.0f "
                + "fused_post_ms=%.0f",
            rows,
            attentionProjectionSeconds * 1_000,
            attentionSeconds * 1_000,
            splitPostAttentionSeconds * 1_000,
            fusedFeedForwardSeconds * 1_000,
            fusedPostAttentionSeconds * 1_000
        ))
        Memory.clearCache()
    }

    /// Compares the existing split post-attention graph with the exact fully
    /// fused residual/MLP tail on top of one selected attention schedule.
    func testMiniMaxH3BlockPostAttentionSchedules() throws {
        try benchGate()
        let environment = ProcessInfo.processInfo.environment
        let rows = max(1, Int(environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 73_470)
        let queryTokens = max(
            1,
            Int(environment["MERERUN_H3_BENCH_QUERY_TOKENS"] ?? "") ?? 640
        )
        let headsPerKernel = max(
            1,
            Int(environment["MERERUN_H3_BENCH_HEADS"] ?? "") ?? 8
        )
        let evaluationBatch = max(
            1,
            Int(environment["MERERUN_H3_BENCH_EVAL_BATCH"] ?? "") ?? 1
        )
        let rounds = max(2, Int(environment["MERERUN_H3_BENCH_ROUNDS"] ?? "") ?? 2)
        let benchmark = MiniMaxH3BlockScheduleBenchmark(
            rowCount: rows,
            maximumQueryTokens: queryTokens,
            maximumKernelsPerEvaluation: evaluationBatch,
            maximumHeadsPerKernel: headsPerKernel
        )

        func output(_ schedule: MiniMaxH3BlockScheduleBenchmark.Schedule) -> MLXArray {
            benchmark(schedule: schedule)
        }
        func elapsed(_ schedule: MiniMaxH3BlockScheduleBenchmark.Schedule) -> Double {
            let started = CFAbsoluteTimeGetCurrent()
            MLX.eval(output(schedule))
            return CFAbsoluteTimeGetCurrent() - started
        }

        let reference = output(.splitPostAttention).asType(.float32)
        let candidate = output(.fusedPostAttention).asType(.float32)
        MLX.eval(reference, candidate)
        let delta = candidate - reference
        let referenceSquared = MLX.sum(reference * reference).item(Float.self)
        let maximumAbsoluteError = Double(MLX.max(MLX.abs(delta)).item(Float.self))
        let relativeL2Error = sqrt(
            Double(MLX.sum(delta * delta).item(Float.self))
                / max(Double(referenceSquared), .leastNonzeroMagnitude)
        )
        XCTAssertLessThan(relativeL2Error, 1e-3)

        var splitTotal = 0.0
        var fusedTotal = 0.0
        Memory.peakMemory = 0
        for round in 0..<rounds {
            if round.isMultiple(of: 2) {
                splitTotal += elapsed(.splitPostAttention)
                fusedTotal += elapsed(.fusedPostAttention)
            } else {
                fusedTotal += elapsed(.fusedPostAttention)
                splitTotal += elapsed(.splitPostAttention)
            }
        }
        let splitSeconds = splitTotal / Double(rounds)
        let fusedSeconds = fusedTotal / Double(rounds)
        let peakGiB = Double(Memory.peakMemory) / 1_073_741_824
        print(String(
            format: "[h3-lab] block-post rows=%d query=%d heads=%d batch=%d "
                + "split_ms=%.0f fused_ms=%.0f speedup=%.3fx "
                + "max_abs=%.6g rel_l2=%.6g peak_gib=%.2f",
            rows,
            queryTokens,
            headsPerKernel,
            evaluationBatch,
            splitSeconds * 1_000,
            fusedSeconds * 1_000,
            splitSeconds / fusedSeconds,
            maximumAbsoluteError,
            relativeL2Error,
            peakGiB
        ))
        Memory.clearCache()
    }

    /// Screens the complete weighted H3 block in fp16 against the production
    /// bf16 path. Both arms reset the same PRNG seed so their weights and
    /// inputs differ only by the requested storage and execution dtype.
    func testMiniMaxH3BlockDTypes() throws {
        try benchGate()
        let environment = ProcessInfo.processInfo.environment
        let rows = max(1, Int(environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 37_966)
        let queryTokens = max(
            1,
            Int(environment["MERERUN_H3_BENCH_QUERY_TOKENS"] ?? "") ?? 768
        )
        let evaluationBatch = max(
            1,
            Int(environment["MERERUN_H3_BENCH_EVAL_BATCH"] ?? "") ?? 1
        )
        let rounds = max(2, Int(environment["MERERUN_H3_BENCH_ROUNDS"] ?? "") ?? 3)
        let seed: UInt64 = 20_260_805

        MLXRandom.seed(seed)
        let bf16 = MiniMaxH3BlockScheduleBenchmark(
            rowCount: rows,
            maximumQueryTokens: queryTokens,
            maximumKernelsPerEvaluation: evaluationBatch,
            dtype: .bfloat16
        )
        MLXRandom.seed(seed)
        let fp16 = MiniMaxH3BlockScheduleBenchmark(
            rowCount: rows,
            maximumQueryTokens: queryTokens,
            maximumKernelsPerEvaluation: evaluationBatch,
            dtype: .float16
        )

        func output(_ benchmark: MiniMaxH3BlockScheduleBenchmark) -> MLXArray {
            benchmark(schedule: .fusedPostAttention).asType(.float32)
        }
        func elapsed(_ benchmark: MiniMaxH3BlockScheduleBenchmark) -> Double {
            let started = CFAbsoluteTimeGetCurrent()
            MLX.eval(output(benchmark))
            return CFAbsoluteTimeGetCurrent() - started
        }

        MLX.eval(output(bf16))
        MLX.eval(output(fp16))
        let bf16Output = output(bf16)
        let fp16Output = output(fp16)
        MLX.eval(bf16Output, fp16Output)
        let delta = fp16Output - bf16Output
        let referenceSquared = MLX.sum(bf16Output * bf16Output).item(Float.self)
        let deltaSquared = MLX.sum(delta * delta).item(Float.self)
        let relativeL2Error = sqrt(
            Double(deltaSquared) / max(Double(referenceSquared), .leastNonzeroMagnitude)
        )
        let maximumAbsoluteError = MLX.max(MLX.abs(delta)).item(Float.self)
        XCTAssertTrue(relativeL2Error.isFinite)
        XCTAssertTrue(Double(maximumAbsoluteError).isFinite)

        var bf16Best = Double.greatestFiniteMagnitude
        var fp16Best = Double.greatestFiniteMagnitude
        Memory.peakMemory = 0
        for round in 0..<rounds {
            if round.isMultiple(of: 2) {
                bf16Best = min(bf16Best, elapsed(bf16))
                fp16Best = min(fp16Best, elapsed(fp16))
            } else {
                fp16Best = min(fp16Best, elapsed(fp16))
                bf16Best = min(bf16Best, elapsed(bf16))
            }
        }

        let qualityGate = 0.01
        let configuration = MiniMaxH3TransformerConfiguration()
        let projectionOperations = 2 * Double(rows) * Double(configuration.hiddenSize)
            * Double(
                4 * configuration.attentionHeadCount * configuration.attentionHeadDimension
                    + 3 * configuration.feedForwardSize
            )
        let attentionOperations = 4 * Double(configuration.attentionHeadCount) * Double(rows)
            * Double(rows) * Double(configuration.attentionHeadDimension)
        let operations = projectionOperations + attentionOperations
        let peakGiB = Double(Memory.peakMemory) / 1_073_741_824
        print(String(
            format: "[h3-lab] block-dtype rows=%d query=%d batch=%d "
                + "bf16_ms=%.0f fp16_ms=%.0f fp16_speedup=%.3fx "
                + "bf16_tflops=%.2f fp16_tflops=%.2f max_abs=%.6g rel_l2=%.6g "
                + "quality_gate=%.3g quality_safe=%@ peak_gib=%.2f",
            rows,
            queryTokens,
            evaluationBatch,
            bf16Best * 1_000,
            fp16Best * 1_000,
            bf16Best / fp16Best,
            operations / bf16Best / 1e12,
            operations / fp16Best / 1e12,
            maximumAbsoluteError,
            relativeL2Error,
            qualityGate,
            relativeL2Error <= qualityGate ? "yes" : "no",
            peakGiB
        ))
        Memory.clearCache()
    }

    /// Tests whether keeping one block's feed-forward tail and the next
    /// block's attention projection in one compiled graph can eliminate two
    /// large hidden-state materialization boundaries without changing H3 math.
    func testMiniMaxH3TwoBlockBoundaryFusion() throws {
        try benchGate()
        let environment = ProcessInfo.processInfo.environment
        let rows = max(1, Int(environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 14_958)
        let queryTokens = max(
            1,
            Int(environment["MERERUN_H3_BENCH_QUERY_TOKENS"] ?? "") ?? 1_024
        )
        let evaluationBatch = max(
            1,
            Int(environment["MERERUN_H3_BENCH_EVAL_BATCH"] ?? "") ?? 4
        )
        let rounds = max(2, Int(environment["MERERUN_H3_BENCH_ROUNDS"] ?? "") ?? 2)
        let inputSeed: UInt64 = 20_260_805
        let first = MiniMaxH3BlockScheduleBenchmark(
            rowCount: rows,
            maximumQueryTokens: queryTokens,
            maximumKernelsPerEvaluation: evaluationBatch,
            weightSeed: 1,
            inputSeed: inputSeed
        )
        let second = MiniMaxH3BlockScheduleBenchmark(
            rowCount: rows,
            maximumQueryTokens: queryTokens,
            maximumKernelsPerEvaluation: evaluationBatch,
            weightSeed: 2,
            inputSeed: inputSeed
        )
        let fusedBoundary = first.makeFusedBoundary(to: second)

        let firstProjection = first.projectAttention()
        MLX.eval(firstProjection)
        let firstAttended = first.attend(firstProjection)
        let boundaryInput = first.attentionOutput(
            input: first.defaultInput,
            attended: firstAttended,
            gate: firstProjection[3]
        )

        func baselineBoundary() -> [MLXArray] {
            let nextHidden = first.splitFeedForward(input: boundaryInput)
            let projected = second.projectAttention(input: nextHidden)
            MLX.eval(projected)
            return [nextHidden] + projected
        }

        func candidateBoundary() -> [MLXArray] {
            let outputs = fusedBoundary(first.fusedBoundaryInputs(
                to: second,
                input: boundaryInput
            ))
            MLX.eval(outputs)
            return outputs
        }

        func finish(_ boundary: [MLXArray]) -> MLXArray {
            let nextHidden = boundary[0]
            let projected = Array(boundary[1...4])
            let attended = second.attend(projected)
            return second.postAttention(
                schedule: .splitPostAttention,
                attended: attended,
                gate: projected[3],
                input: nextHidden
            )
        }

        func relativeL2(_ candidate: MLXArray, reference: MLXArray) -> Double {
            let candidate32 = candidate.asType(.float32)
            let reference32 = reference.asType(.float32)
            MLX.eval(candidate32, reference32)
            let delta = candidate32 - reference32
            let errorSquared = MLX.sum(delta * delta).item(Float.self)
            let referenceSquared = MLX.sum(reference32 * reference32).item(Float.self)
            return sqrt(
                Double(errorSquared) / max(Double(referenceSquared), .leastNonzeroMagnitude)
            )
        }

        MLX.eval(finish(baselineBoundary()))
        MLX.eval(finish(candidateBoundary()))
        let reference = finish(baselineBoundary())
        let candidate = finish(candidateBoundary())
        MLX.eval(reference, candidate)
        let candidateRelativeL2 = relativeL2(candidate, reference: reference)
        XCTAssertLessThan(candidateRelativeL2, 1e-3)

        func elapsed(_ body: () -> MLXArray) -> Double {
            let started = CFAbsoluteTimeGetCurrent()
            MLX.eval(body())
            return CFAbsoluteTimeGetCurrent() - started
        }
        func boundaryElapsed(_ body: () -> [MLXArray]) -> Double {
            let started = CFAbsoluteTimeGetCurrent()
            MLX.eval(body())
            return CFAbsoluteTimeGetCurrent() - started
        }

        var baselineTotal = 0.0
        var candidateTotal = 0.0
        var baselineBoundaryTotal = 0.0
        var candidateBoundaryTotal = 0.0
        Memory.peakMemory = 0
        for round in 0..<rounds {
            if round.isMultiple(of: 2) {
                baselineTotal += elapsed { finish(baselineBoundary()) }
                candidateTotal += elapsed { finish(candidateBoundary()) }
                baselineBoundaryTotal += boundaryElapsed { baselineBoundary() }
                candidateBoundaryTotal += boundaryElapsed { candidateBoundary() }
            } else {
                candidateBoundaryTotal += boundaryElapsed { candidateBoundary() }
                baselineBoundaryTotal += boundaryElapsed { baselineBoundary() }
                candidateTotal += elapsed { finish(candidateBoundary()) }
                baselineTotal += elapsed { finish(baselineBoundary()) }
            }
        }

        let baselineSeconds = baselineTotal / Double(rounds)
        let candidateSeconds = candidateTotal / Double(rounds)
        let baselineBoundarySeconds = baselineBoundaryTotal / Double(rounds)
        let candidateBoundarySeconds = candidateBoundaryTotal / Double(rounds)
        let peakGiB = Double(Memory.peakMemory) / 1_073_741_824
        print(String(
            format: "[h3-lab] two-block-boundary rows=%d query=%d batch=%d "
                + "baseline_ms=%.0f fused_ms=%.0f speedup=%.3fx "
                + "baseline_boundary_ms=%.0f fused_boundary_ms=%.0f "
                + "boundary_speedup=%.3fx fused_rel_l2=%.6g peak_gib=%.2f",
            rows,
            queryTokens,
            evaluationBatch,
            baselineSeconds * 1_000,
            candidateSeconds * 1_000,
            baselineSeconds / candidateSeconds,
            baselineBoundarySeconds * 1_000,
            candidateBoundarySeconds * 1_000,
            baselineBoundarySeconds / candidateBoundarySeconds,
            candidateRelativeL2,
            peakGiB
        ))
        Memory.clearCache()
    }

    /// Separates the cost of rebinding the shared compiled block runner from
    /// the cost of turning over independent H3 weight sets. Production reuses
    /// one compiled graph across 50 blocks, so a hot single-block loop can
    /// otherwise overstate sustained checkpoint throughput.
    func testMiniMaxH3BlockWeightTurnover() throws {
        try benchGate()
        let environment = ProcessInfo.processInfo.environment
        let rows = max(1, Int(environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 14_958)
        let queryTokens = max(
            1,
            Int(environment["MERERUN_H3_BENCH_QUERY_TOKENS"] ?? "") ?? 1_024
        )
        let evaluationBatch = max(
            1,
            Int(environment["MERERUN_H3_BENCH_EVAL_BATCH"] ?? "") ?? 4
        )
        let rounds = max(2, Int(environment["MERERUN_H3_BENCH_ROUNDS"] ?? "") ?? 2)
        let inputSeed: UInt64 = 20_260_805
        let shared = MiniMaxH3BlockScheduleBenchmark(
            rowCount: rows,
            maximumQueryTokens: queryTokens,
            maximumKernelsPerEvaluation: evaluationBatch,
            weightSeed: 1,
            inputSeed: inputSeed
        )
        let sourceA = MiniMaxH3BlockScheduleBenchmark(
            rowCount: rows,
            maximumQueryTokens: queryTokens,
            maximumKernelsPerEvaluation: evaluationBatch,
            weightSeed: 1,
            inputSeed: inputSeed
        )
        let sourceB = MiniMaxH3BlockScheduleBenchmark(
            rowCount: rows,
            maximumQueryTokens: queryTokens,
            maximumKernelsPerEvaluation: evaluationBatch,
            weightSeed: 2,
            inputSeed: inputSeed
        )

        func output(_ benchmark: MiniMaxH3BlockScheduleBenchmark) -> MLXArray {
            benchmark(schedule: .splitPostAttention)
        }
        func relativeL2(_ candidate: MLXArray, reference: MLXArray) -> Double {
            let candidate32 = candidate.asType(.float32)
            let reference32 = reference.asType(.float32)
            MLX.eval(candidate32, reference32)
            let delta = candidate32 - reference32
            let errorSquared = MLX.sum(delta * delta).item(Float.self)
            let referenceSquared = MLX.sum(reference32 * reference32).item(Float.self)
            return sqrt(
                Double(errorSquared) / max(Double(referenceSquared), .leastNonzeroMagnitude)
            )
        }

        let dedicatedA = output(sourceA)
        MLX.eval(dedicatedA)
        shared.useOriginalWeights(from: sourceA)
        let reboundA = output(shared)
        MLX.eval(reboundA)
        let reboundARelativeL2 = relativeL2(reboundA, reference: dedicatedA)
        XCTAssertLessThan(reboundARelativeL2, 1e-3)

        let dedicatedB = output(sourceB)
        MLX.eval(dedicatedB)
        shared.useOriginalWeights(from: sourceB)
        let reboundB = output(shared)
        MLX.eval(reboundB)
        let reboundBRelativeL2 = relativeL2(reboundB, reference: dedicatedB)
        XCTAssertLessThan(reboundBRelativeL2, 1e-3)

        func elapsed(_ body: () -> MLXArray) -> Double {
            let started = CFAbsoluteTimeGetCurrent()
            MLX.eval(body())
            return CFAbsoluteTimeGetCurrent() - started
        }

        var hotTotal = 0.0
        var reboundSameTotal = 0.0
        var reboundAlternatingTotal = 0.0
        var dedicatedAlternatingTotal = 0.0
        for round in 0..<rounds {
            let selectedSource = round.isMultiple(of: 2) ? sourceA : sourceB
            let selectedDedicated = round.isMultiple(of: 2) ? sourceA : sourceB

            shared.useOriginalWeights(from: selectedSource)
            hotTotal += elapsed { output(shared) }

            reboundSameTotal += elapsed {
                shared.useOriginalWeights(from: selectedSource)
                return output(shared)
            }

            let alternateSource = round.isMultiple(of: 2) ? sourceB : sourceA
            reboundAlternatingTotal += elapsed {
                shared.useOriginalWeights(from: alternateSource)
                return output(shared)
            }

            dedicatedAlternatingTotal += elapsed { output(selectedDedicated) }
        }

        let hotSeconds = hotTotal / Double(rounds)
        let reboundSameSeconds = reboundSameTotal / Double(rounds)
        let reboundAlternatingSeconds = reboundAlternatingTotal / Double(rounds)
        let dedicatedAlternatingSeconds = dedicatedAlternatingTotal / Double(rounds)
        print(String(
            format: "[h3-lab] block-turnover rows=%d query=%d batch=%d "
                + "hot_ms=%.0f rebound_same_ms=%.0f rebound_alternating_ms=%.0f "
                + "dedicated_alternating_ms=%.0f dedicated_speedup=%.3fx "
                + "rebound_a_rel_l2=%.6g rebound_b_rel_l2=%.6g",
            rows,
            queryTokens,
            evaluationBatch,
            hotSeconds * 1_000,
            reboundSameSeconds * 1_000,
            reboundAlternatingSeconds * 1_000,
            dedicatedAlternatingSeconds * 1_000,
            reboundAlternatingSeconds / dedicatedAlternatingSeconds,
            reboundARelativeL2,
            reboundBRelativeL2
        ))
        Memory.clearCache()
    }

    /// Compares attention chunk schedules inside the same compiled, weighted
    /// production block. This is the acceptance gate for changing the true-768
    /// H3 schedule; isolated SDPA timings are only useful for finding candidates.
    func testMiniMaxH3BlockAttentionSchedules() throws {
        try benchGate()
        let environment = ProcessInfo.processInfo.environment
        let rows = max(1, Int(environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 37_966)
        let rounds = max(2, Int(environment["MERERUN_H3_BENCH_ROUNDS"] ?? "") ?? 4)
        let referenceQueryTokens = max(
            1,
            Int(environment["MERERUN_H3_BENCH_REFERENCE_QUERY_TOKENS"] ?? "") ?? 1_024
        )
        let referenceEvaluationBatch = max(
            1,
            Int(environment["MERERUN_H3_BENCH_REFERENCE_EVAL_BATCH"] ?? "") ?? 4
        )
        let referenceHeads = max(
            1,
            Int(environment["MERERUN_H3_BENCH_REFERENCE_HEADS"] ?? "") ?? 56
        )
        let candidateQueryTokens = max(
            1,
            Int(environment["MERERUN_H3_BENCH_CANDIDATE_QUERY_TOKENS"] ?? "") ?? 768
        )
        let candidateEvaluationBatch = max(
            1,
            Int(environment["MERERUN_H3_BENCH_CANDIDATE_EVAL_BATCH"] ?? "") ?? 1
        )
        let candidateHeads = max(
            1,
            Int(environment["MERERUN_H3_BENCH_CANDIDATE_HEADS"] ?? "") ?? 56
        )
        let benchmark = MiniMaxH3BlockScheduleBenchmark(
            rowCount: rows,
            maximumQueryTokens: referenceQueryTokens,
            maximumKernelsPerEvaluation: referenceEvaluationBatch
        )

        func output(queryTokens: Int, heads: Int, evaluationBatch: Int) -> MLXArray {
            benchmark(
                schedule: .splitPostAttention,
                maximumQueryTokens: queryTokens,
                maximumHeadsPerKernel: heads,
                maximumKernelsPerEvaluation: evaluationBatch
            )
        }
        func elapsed(queryTokens: Int, heads: Int, evaluationBatch: Int) -> Double {
            let started = CFAbsoluteTimeGetCurrent()
            MLX.eval(output(
                queryTokens: queryTokens,
                heads: heads,
                evaluationBatch: evaluationBatch
            ))
            return CFAbsoluteTimeGetCurrent() - started
        }

        MLX.eval(output(
            queryTokens: referenceQueryTokens,
            heads: referenceHeads,
            evaluationBatch: referenceEvaluationBatch
        ))
        MLX.eval(output(
            queryTokens: candidateQueryTokens,
            heads: candidateHeads,
            evaluationBatch: candidateEvaluationBatch
        ))
        let reference = output(
            queryTokens: referenceQueryTokens,
            heads: referenceHeads,
            evaluationBatch: referenceEvaluationBatch
        ).asType(.float32)
        let candidate = output(
            queryTokens: candidateQueryTokens,
            heads: candidateHeads,
            evaluationBatch: candidateEvaluationBatch
        ).asType(.float32)
        MLX.eval(reference, candidate)
        let delta = candidate - reference
        let referenceSquared = MLX.sum(reference * reference).item(Float.self)
        let relativeL2Error = sqrt(
            Double(MLX.sum(delta * delta).item(Float.self))
                / max(Double(referenceSquared), .leastNonzeroMagnitude)
        )
        XCTAssertLessThan(relativeL2Error, 1e-3)

        var referenceTotal = 0.0
        var candidateTotal = 0.0
        for round in 0..<rounds {
            if round.isMultiple(of: 2) {
                referenceTotal += elapsed(
                    queryTokens: referenceQueryTokens,
                    heads: referenceHeads,
                    evaluationBatch: referenceEvaluationBatch
                )
                candidateTotal += elapsed(
                    queryTokens: candidateQueryTokens,
                    heads: candidateHeads,
                    evaluationBatch: candidateEvaluationBatch
                )
            } else {
                candidateTotal += elapsed(
                    queryTokens: candidateQueryTokens,
                    heads: candidateHeads,
                    evaluationBatch: candidateEvaluationBatch
                )
                referenceTotal += elapsed(
                    queryTokens: referenceQueryTokens,
                    heads: referenceHeads,
                    evaluationBatch: referenceEvaluationBatch
                )
            }
        }
        let referenceSeconds = referenceTotal / Double(rounds)
        let candidateSeconds = candidateTotal / Double(rounds)
        print(String(
            format: "[h3-lab] block-attention rows=%d reference=%dx%dx%d reference_ms=%.0f "
                + "candidate=%dx%dx%d candidate_ms=%.0f speedup=%.3fx relative_l2=%.6g",
            rows,
            referenceQueryTokens,
            referenceHeads,
            referenceEvaluationBatch,
            referenceSeconds * 1_000,
            candidateQueryTokens,
            candidateHeads,
            candidateEvaluationBatch,
            candidateSeconds * 1_000,
            referenceSeconds / candidateSeconds,
            relativeL2Error
        ))
        Memory.clearCache()
    }

    /// Loads the released H3 video VAE and times one complete decode with a
    /// configurable spatial tile and geometry. Run each size in a fresh process.
    func testMiniMaxH3VideoVAETileSize() throws {
        try benchGate()
        let environment = ProcessInfo.processInfo.environment
        guard let root = ProcessInfo.processInfo.environment["MERERUN_H3_MODEL_ROOT"],
              !root.isEmpty else {
            throw XCTSkip("Set MERERUN_H3_MODEL_ROOT to an installed H3 model root")
        }
        let tileSize = Int(environment["MERERUN_H3_VAE_TILE_SIZE"] ?? "")
            ?? MiniMaxH3VideoVAE.defaultSpatialTileSize
        let width = Int(environment["MERERUN_H3_VAE_WIDTH"] ?? "") ?? 832
        let height = Int(environment["MERERUN_H3_VAE_HEIGHT"] ?? "") ?? 480
        let frameCount = Int(environment["MERERUN_H3_VAE_FRAMES"] ?? "") ?? 124
        XCTAssertTrue(width.isMultiple(of: 16))
        XCTAssertTrue(height.isMultiple(of: 16))
        guard frameCount >= 22 else {
            XCTFail("H3 video VAE decode requires at least 22 output frames")
            return
        }
        let latentFrames = try MiniMaxH3Geometry.videoLatentFrameCount(for: frameCount)
        let usesCompiledDecoder = environment["MERERUN_H3_VAE_COMPILED"] != "0"
        let materializesWeights = environment["MERERUN_H3_VAE_MATERIALIZE"] == "1"
        let evaluatesChunksIndividually = environment["MERERUN_H3_VAE_CHUNK_EVAL"] != "0"
        let totalStarted = CFAbsoluteTimeGetCurrent()
        let resources = MiniMaxH3Resources(rootURL: URL(fileURLWithPath: root, isDirectory: true))
        let model = try MiniMaxH3ModelLoader.loadVideoVAE(resources: resources)
        model.spatialTileSize = tileSize
        model.usesCompiledTileDecoder = usesCompiledDecoder
        model.evaluatesTemporalChunksIndividually = evaluatesChunksIndividually
        if materializesWeights {
            MLX.eval(model.parameters())
        }
        let loadSeconds = CFAbsoluteTimeGetCurrent() - totalStarted
        let latents = MLXRandom.normal([
            1,
            24,
            latentFrames,
            height / 16,
            width / 16,
        ]).asType(.float32)
        MLX.eval(latents)

        let started = CFAbsoluteTimeGetCurrent()
        let decoded = model.decode(latents)
        MLX.eval(decoded)
        let decodeSeconds = CFAbsoluteTimeGetCurrent() - started
        print(String(
            format: "[dit-bench] H3 video VAE tile=%d compiled=%@ materialized=%@ chunk_eval=%@ "
                + "size=%dx%d frames=%d load=%.3fs decode=%.3fs total=%.3fs peak=%.2fGiB",
            tileSize,
            usesCompiledDecoder ? "true" : "false",
            materializesWeights ? "true" : "false",
            evaluatesChunksIndividually ? "true" : "false",
            width,
            height,
            decoded.dim(1),
            loadSeconds,
            decodeSeconds,
            loadSeconds + decodeSeconds,
            Double(Memory.snapshot().peakMemory) / 1_073_741_824
        ))
        XCTAssertEqual(decoded.shape, [1, frameCount, height, width, 3])
    }

    /// Full-model paired A/B: the same quantized klein-nano forward with the
    /// native qmm kernels versus the large-M dense path (transient dequant)
    /// versus a resident cached-dequant arm. Interleaved in one process.
    func testKleinNanoQmmVsDensePaired() throws {
        try benchGate()
        let config = Flux2TransformerConfiguration(quantized: true)
        let model = Flux2Transformer2DModel(config: config)
        model.update(parameters: model.parameters().mapValues {
            $0.dtype == .float32 ? $0.asType(.bfloat16) : $0
        })
        MLX.eval(model.parameters())

        let txtLen = 512
        let hidden = MLXRandom.normal([1, 4096, config.inChannels]).asType(.bfloat16)
        let encoder = MLXRandom.normal([1, txtLen, config.contextDim]).asType(.bfloat16)
        let timestep = MLXArray([Float(0.5)])
        let imgIds = Flux2PosEmbed.prepareMultiImageIds(imageCount: 1, height: 64, width: 64, tCoords: [0])
        let txtIds = Flux2PosEmbed.prepareTextIds(seqLen: txtLen, numAxes: 4)
        MLX.eval(hidden, encoder, timestep, imgIds, txtIds)

        func forward() -> MLXArray {
            model(
                hiddenStates: hidden, encoderHiddenStates: encoder,
                timestep: timestep, imgIds: imgIds, txtIds: txtIds, guidance: nil
            )
        }

        // Swap every QuantizedLinear leaf for the requested arm.
        enum Arm: String { case qmm, transient, cached }
        var originalByPath: [String: QuantizedLinear] = [:]
        func install(_ arm: Arm) {
            var updates: [(String, Module)] = []
            for (path, module) in model.leafModules().flattened() {
                guard let q = module as? QuantizedLinear else { continue }
                let base: QuantizedLinear
                if let original = originalByPath[path] {
                    base = original
                } else {
                    originalByPath[path] = q
                    base = q
                }
                switch arm {
                case .qmm: updates.append((path, base))
                case .transient: updates.append((path, BenchTransientDenseLinear(copying: base)))
                case .cached: updates.append((path, BenchCachedDenseLinear(copying: base)))
                }
            }
            model.update(modules: ModuleChildren.unflattened(updates))
        }

        var results: [String: Double] = [:]
        for round in 0..<3 {
            for arm in [Arm.qmm, .transient, .cached] {
                install(arm)
                MLX.eval(forward())
                let start = CFAbsoluteTimeGetCurrent()
                for _ in 0..<2 { MLX.eval(forward()) }
                let per = (CFAbsoluteTimeGetCurrent() - start) / 2
                results[arm.rawValue] = min(results[arm.rawValue] ?? .greatestFiniteMagnitude, per)
                if round == 2 {
                    print(String(format: "[dit-bench] PAIRED %@ forward=%.0f ms", arm.rawValue, (results[arm.rawValue] ?? 0) * 1000))
                }
            }
        }

        // Numerics: qmm vs dense on one layer's exact weights.
        install(.qmm)
        let reference = forward()
        install(.transient)
        let dense = forward()
        MLX.eval(reference, dense)
        let diff = MLX.abs(reference.asType(.float32) - dense.asType(.float32)).max().item(Float.self)
        let scale = MLX.abs(reference.asType(.float32)).max().item(Float.self)
        print(String(format: "[dit-bench] PARITY maxAbsDiff=%.5f (ref maxAbs=%.3f)", diff, scale))
    }

    /// Krea2 attention probes at its real geometry (48 q-heads, 12 kv-heads,
    /// head dim 128, ~4608 joint tokens): the cost of the dense all-zeros
    /// additive mask, and manual KV repeat versus native GQA.
    func testKrea2AttentionProbes() throws {
        try benchGate()
        let tokens = 4608
        let q = MLXRandom.normal([1, 48, tokens, 128]).asType(.bfloat16)
        let kFull = MLXRandom.normal([1, 48, tokens, 128]).asType(.bfloat16)
        let vFull = MLXRandom.normal([1, 48, tokens, 128]).asType(.bfloat16)
        let kGQA = MLXRandom.normal([1, 12, tokens, 128]).asType(.bfloat16)
        let vGQA = MLXRandom.normal([1, 12, tokens, 128]).asType(.bfloat16)
        let zeroMask = MLX.zeros([1, 1, tokens, tokens], dtype: .bfloat16)
        MLX.eval(q, kFull, vFull, kGQA, vGQA, zeroMask)
        let scale = Float(1.0 / Double(128).squareRoot())

        _ = time("sdpa mask=none      ") {
            MLXFast.scaledDotProductAttention(queries: q, keys: kFull, values: vFull, scale: scale, mask: .none)
        }
        _ = time("sdpa mask=zeros LxL ") {
            MLXFast.scaledDotProductAttention(queries: q, keys: kFull, values: vFull, scale: scale, mask: .array(zeroMask))
        }
        _ = time("sdpa gqa native 12kv") {
            MLXFast.scaledDotProductAttention(queries: q, keys: kGQA, values: vGQA, scale: scale, mask: .none)
        }
        _ = time("sdpa gqa manual rep ") {
            let kRep = MLX.repeated(kGQA, count: 4, axis: 1)
            let vRep = MLX.repeated(vGQA, count: 4, axis: 1)
            return MLXFast.scaledDotProductAttention(queries: q, keys: kRep, values: vRep, scale: scale, mask: .none)
        }
    }

    /// Crossover scan: at which row count does dequant+GEMM beat the native
    /// quantized matmul kernel?
    func testDequantGemmCrossover() throws {
        try benchGate()
        let dim = 3072
        let q4 = QuantizedLinear(dim, dim, bias: false, groupSize: 64, bits: 4)
        MLX.eval(q4.parameters())
        for rows in [64, 128, 256, 512, 1024, 2048, 4608] {
            let x = MLXRandom.normal([1, rows, dim]).asType(.bfloat16)
            MLX.eval(x)
            let qmm = time(String(format: "qmm   M=%4d", rows), iters: 20) { q4(x) }
            let dq = time(String(format: "dq+mm M=%4d", rows), iters: 20) {
                let w = dequantized(
                    q4.weight, scales: q4.scales, biases: q4.biases,
                    groupSize: q4.groupSize, bits: q4.bits
                ).asType(.bfloat16)
                return MLX.matmul(x, w.transposed())
            }
            print(String(format: "[dit-bench] XOVER M=%4d qmm=%.3fms dq+mm=%.3fms winner=%@",
                         rows, qmm * 1000, dq * 1000, dq < qmm ? "dq+mm" : "qmm"))
        }
    }
}
