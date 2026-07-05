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
