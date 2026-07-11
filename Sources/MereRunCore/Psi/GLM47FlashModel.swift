import Foundation
import MLX
import MLXFast
import MLXNN

private func swiglu(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
    MLXNN.silu(gate) * up
}

private func repeatAlongHeads(_ x: MLXArray, heads: Int) -> MLXArray {
    MLX.repeated(x, count: heads, axis: 1)
}

private func createMask(h: MLXArray, cache: KVCache?) -> MLXFast.ScaledDotProductAttentionMaskMode {
    createAttentionMask(h: h, cache: cache)
}

final class GLM47SwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray?
    @ModuleInfo(key: "biases") var biases: MLXArray?
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let groupSize: Int
    let bits: Int

    init(inputDims: Int, outputDims: Int, numExperts: Int, groupSize: Int, bits: Int, bias: Bool) {
        self.groupSize = groupSize
        self.bits = bits
        // GLM-Flash expert checkpoints are pre-quantized. Initializing a
        // dense random [experts,out,in] tensor creates a huge throwaway graph
        // before the packed safetensor replaces it.
        let packedInputDims = (inputDims * bits + 31) / 32
        self._weight.wrappedValue = MLXArray.zeros(
            [numExperts, outputDims, packedInputDims],
            dtype: .uint32
        )
        let groups = max(1, (inputDims + groupSize - 1) / groupSize)
        self._scales.wrappedValue = MLXArray.zeros([numExperts, outputDims, groups])
        self._biases.wrappedValue = MLXArray.zeros([numExperts, outputDims, groups])
        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batchTokens = x.dim(0) * x.dim(1)
        let topK = indices.dim(2)
        let inputDim = x.dim(x.ndim - 1)

        // SwitchGLU calls gate/up/down in sequence; downProj sees [B,L,topK,H].
        // Detect already-expanded inputs to avoid double expansion.
        let flatX: MLXArray
        if x.ndim == 4 && x.dim(2) == topK {
            flatX = x.reshaped([batchTokens * topK, 1, inputDim])
        } else {
            let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
            return applyGather(expanded, indices: indices, sortedIndices: false)
                .squeezed(axis: -2)
        }

        let flatIdx = indices.reshaped([batchTokens * topK])
        let output = applyFlat(flatX, indices: flatIdx, sortedIndices: false)
        let outDim = output.dim(2)
        var reshaped = output.reshaped([batchTokens, topK, outDim])
        reshaped = reshaped.reshaped([x.dim(0), x.dim(1), topK, outDim])

        return reshaped
    }

    func applyFlat(_ x: MLXArray, indices: MLXArray, sortedIndices: Bool) -> MLXArray {
        applyGather(x, indices: indices, sortedIndices: sortedIndices)
    }

    private func applyGather(_ x: MLXArray, indices: MLXArray, sortedIndices: Bool) -> MLXArray {
        let inputDim = x.dim(x.ndim - 1)
        let output: MLXArray
        if let scales {
            let resolved = resolvedQuantization(inputDim: inputDim, scales: scales)
            output = gatherQuantizedMM(
                x,
                weight,
                scales: scales,
                biases: biases,
                rhsIndices: indices,
                transpose: true,
                groupSize: resolved.groupSize,
                bits: resolved.bits,
                mode: .affine,
                sortedIndices: sortedIndices
            )
        } else {
            output = gatherMM(
                x,
                weight.swappedAxes(-1, -2),
                rhsIndices: indices,
                sortedIndices: sortedIndices
            )
        }

        if let bias {
            return output + bias.take(indices, axis: 0).expandedDimensions(axis: -2)
        }
        return output
    }

    private func resolvedQuantization(inputDim: Int, scales: MLXArray) -> (groupSize: Int, bits: Int) {
        var resolvedBits = bits
        let packedInputDim = weight.dim(weight.ndim - 1)
        let numerator = packedInputDim * 32
        if inputDim > 0, numerator % inputDim == 0 {
            let inferredBits = numerator / inputDim
            if (2...8).contains(inferredBits) {
                resolvedBits = inferredBits
            }
        }

        var resolvedGroupSize = groupSize
        let scaleGroups = scales.dim(scales.ndim - 1)
        if inputDim > 0, scaleGroups > 0, inputDim % scaleGroups == 0 {
            resolvedGroupSize = inputDim / scaleGroups
        }
        return (resolvedGroupSize, resolvedBits)
    }
}

final class GLM47SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: GLM47SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: GLM47SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: GLM47SwitchLinear

    private struct FusedGateUp {
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray?
        let intermediate: Int
        let sourceIDs: [ObjectIdentifier]
    }

    private var fusedGateUp: FusedGateUp?
    private var fusedGateUpAttempted = false
    private let fusedGateUpEnabled: Bool

    init(inputDims: Int, hiddenDims: Int, numExperts: Int, groupSize: Int, bits: Int) {
        self.fusedGateUpEnabled = GLM47FusedMoEPolicy.isEnabled()
        self._gateProj.wrappedValue = GLM47SwitchLinear(
            inputDims: inputDims,
            outputDims: hiddenDims,
            numExperts: numExperts,
            groupSize: groupSize,
            bits: bits,
            bias: false
        )
        self._upProj.wrappedValue = GLM47SwitchLinear(
            inputDims: inputDims,
            outputDims: hiddenDims,
            numExperts: numExperts,
            groupSize: groupSize,
            bits: bits,
            bias: false
        )
        self._downProj.wrappedValue = GLM47SwitchLinear(
            inputDims: hiddenDims,
            outputDims: inputDims,
            numExperts: numExperts,
            groupSize: groupSize,
            bits: bits,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batchTokens = x.dim(0) * x.dim(1)
        let topK = indices.dim(2)
        let routeCount = batchTokens * topK
        guard routeCount >= 64 else {
            return unsorted(x, indices: indices)
        }

        let inputDim = x.dim(x.ndim - 1)
        let flatIndices = indices.reshaped([routeCount])
        let order = argSort(flatIndices, axis: 0)
        let inverseOrder = argSort(order, axis: 0)
        let sortedIndices = flatIndices.take(order, axis: 0)
        let tokenOrder = order.floorDivide(topK)
        let flatInput = x.reshaped([batchTokens, inputDim])
            .take(tokenOrder, axis: 0)
            .reshaped([routeCount, 1, inputDim])

        let activated: MLXArray
        if let fused = resolvedFusedGateUp() {
            activated = fusedGateUpGLU(
                flatInput,
                fused: fused,
                indices: sortedIndices,
                sortedIndices: true
            )
        } else {
            let up = upProj.applyFlat(flatInput, indices: sortedIndices, sortedIndices: true)
            let gate = gateProj.applyFlat(flatInput, indices: sortedIndices, sortedIndices: true)
            activated = swiglu(gate, up)
        }
        let output = downProj.applyFlat(activated, indices: sortedIndices, sortedIndices: true)
            .take(inverseOrder, axis: 0)
        return output.reshaped([x.dim(0), x.dim(1), topK, output.dim(2)])
    }

    func unsorted(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let activated: MLXArray
        if let fused = resolvedFusedGateUp() {
            let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
            activated = fusedGateUpGLU(
                expanded,
                fused: fused,
                indices: indices,
                sortedIndices: false
            ).squeezed(axis: -2)
        } else {
            let up = upProj(x, indices: indices)
            let gate = gateProj(x, indices: indices)
            activated = swiglu(gate, up)
        }
        return downProj(activated, indices: indices)
    }

    func unfusedForTesting(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let up = upProj(x, indices: indices)
        let gate = gateProj(x, indices: indices)
        let activated = swiglu(gate, up)
        return downProj(activated, indices: indices)
    }

    func fusedForTesting(_ x: MLXArray, indices: MLXArray) -> MLXArray? {
        guard let fused = resolvedFusedGateUp(force: true) else { return nil }
        let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
        let activated = fusedGateUpGLU(
            expanded,
            fused: fused,
            indices: indices,
            sortedIndices: false
        ).squeezed(axis: -2)
        return downProj(activated, indices: indices)
    }

    private func resolvedFusedGateUp(force: Bool = false) -> FusedGateUp? {
        guard force || fusedGateUpEnabled else { return nil }
        guard let gateScales = gateProj.scales, let upScales = upProj.scales else { return nil }
        var sourceIDs = [
            ObjectIdentifier(gateProj.weight),
            ObjectIdentifier(upProj.weight),
            ObjectIdentifier(gateScales),
            ObjectIdentifier(upScales),
        ]
        if let gateBiases = gateProj.biases {
            sourceIDs.append(ObjectIdentifier(gateBiases))
        }
        if let upBiases = upProj.biases {
            sourceIDs.append(ObjectIdentifier(upBiases))
        }

        if let fusedGateUp, fusedGateUp.sourceIDs == sourceIDs {
            return fusedGateUp
        }
        if fusedGateUp != nil {
            self.fusedGateUp = nil
            fusedGateUpAttempted = false
        }
        if !fusedGateUpAttempted {
            fusedGateUpAttempted = true
            fusedGateUp = buildFusedGateUp(
                gateScales: gateScales,
                upScales: upScales,
                sourceIDs: sourceIDs
            )
        }
        return fusedGateUp
    }

    private func buildFusedGateUp(
        gateScales: MLXArray,
        upScales: MLXArray,
        sourceIDs: [ObjectIdentifier]
    ) -> FusedGateUp? {
        guard gateProj.bias == nil,
              upProj.bias == nil,
              gateProj.weight.shape == upProj.weight.shape,
              gateProj.weight.dtype == upProj.weight.dtype,
              gateScales.shape == upScales.shape,
              (gateProj.biases == nil) == (upProj.biases == nil) else {
            return nil
        }
        let weight = concatenated([gateProj.weight, upProj.weight], axis: 1)
        let scales = concatenated([gateScales, upScales], axis: 1)
        let biases = gateProj.biases.flatMap { gateBiases in
            upProj.biases.map { concatenated([gateBiases, $0], axis: 1) }
        }
        var arrays = [weight, scales]
        if let biases { arrays.append(biases) }
        MLX.eval(arrays)
        return FusedGateUp(
            weight: weight,
            scales: scales,
            biases: biases,
            intermediate: gateProj.weight.dim(1),
            sourceIDs: sourceIDs
        )
    }

    private func fusedGateUpGLU(
        _ x: MLXArray,
        fused: FusedGateUp,
        indices: MLXArray,
        sortedIndices: Bool
    ) -> MLXArray {
        let inputDim = x.dim(x.ndim - 1)
        let packedInputDim = fused.weight.dim(fused.weight.ndim - 1)
        let inferredBits = inputDim > 0 && (packedInputDim * 32) % inputDim == 0
            ? (packedInputDim * 32) / inputDim
            : gateProj.bits
        let scaleGroups = fused.scales.dim(fused.scales.ndim - 1)
        let inferredGroup = inputDim > 0 && scaleGroups > 0 && inputDim % scaleGroups == 0
            ? inputDim / scaleGroups
            : gateProj.groupSize
        let output = gatherQuantizedMM(
            x,
            fused.weight,
            scales: fused.scales,
            biases: fused.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: inferredGroup,
            bits: inferredBits,
            mode: .affine,
            sortedIndices: sortedIndices
        )
        let parts = split(output, indices: [fused.intermediate], axis: -1)
        return swiglu(parts[0], parts[1])
    }
}

private final class GLM47MoEGate: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    init(config: GLM47FlashConfig) {
        self.topK = config.numExpertsPerTok
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup
        self.routedScalingFactor = config.routedScalingFactor
        self.normTopkProb = config.normTopkProb
        let nExperts = config.nRoutedExperts ?? 0
        self._weight.wrappedValue = MLXArray.zeros([nExperts, config.hiddenSize])
        self._correctionBias.wrappedValue = MLXArray.zeros([nExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        let gates = MLX.matmul(x, weight.transposed())
        var scores = MLX.sigmoid(gates.asType(.float32))
        let origScores = scores

        scores = scores + correctionBias

        if nGroup > 1 {
            // n_group > 1 is unused for GLM-4.7 Flash; skip grouping.
        }

        let kth = topK - 1
        let inds = argPartition(-scores, kth: kth, axis: -1)[.ellipsis, 0..<topK]
        var topScores = takeAlong(origScores, inds, axis: -1)
        if topK > 1 && normTopkProb {
            let denom = topScores.sum(axis: -1, keepDims: true)
            topScores = topScores / denom
        }
        topScores = topScores * routedScalingFactor
        return (inds, topScores)
    }
}

private final class GLM47FlashAttention: Module {
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear

    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProj: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear

    @ModuleInfo(key: "o_proj") var oProj: Linear

    private let numHeads: Int
    private let qkRopeHeadDim: Int
    private let qkNopeHeadDim: Int
    private let vHeadDim: Int
    private let qHeadDim: Int
    private let scale: Float
    private let rope: RoPE
    private let kvLoraRank: Int

    private struct CachedAbsorbedMLA {
        let weights: GLM47AbsorbedMLAWeights
        let source: ObjectIdentifier
    }

    private var cachedAbsorbedMLA: CachedAbsorbedMLA?

    init(config: GLM47FlashConfig) {
        self.numHeads = config.numAttentionHeads
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.vHeadDim = config.vHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        self.scale = 1.0 / sqrt(Float(self.qHeadDim))
        self.kvLoraRank = config.kvLoraRank

        self._qAProj.wrappedValue = Linear(config.hiddenSize, config.qLoraRank, bias: config.attentionBias)
        self._qALayerNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        self._qBProj.wrappedValue = Linear(config.qLoraRank, config.numAttentionHeads * self.qHeadDim, bias: false)

        self._kvAProj.wrappedValue = Linear(
            config.hiddenSize,
            config.kvLoraRank + config.qkRopeHeadDim,
            bias: config.attentionBias
        )
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: config.kvLoraRank, eps: config.rmsNormEps)
        let kvOutDim = config.numAttentionHeads * (self.qHeadDim - config.qkRopeHeadDim + config.vHeadDim)
        self._kvBProj.wrappedValue = Linear(config.kvLoraRank, kvOutDim, bias: false)

        self._oProj.wrappedValue = Linear(config.numAttentionHeads * config.vHeadDim, config.hiddenSize, bias: config.attentionBias)
        self.rope = RoPE(dimensions: config.qkRopeHeadDim, traditional: config.ropeTraditional ?? true, base: config.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let q = qBProj(qALayerNorm(qAProj(x)))
        let qStates = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let qNope = qStates[.ellipsis, 0..<qkNopeHeadDim]
        var qPe = qStates[.ellipsis, qkNopeHeadDim...]

        let compressedKV = kvAProj(x)
        var kvNope = compressedKV[.ellipsis, 0..<kvLoraRank]
        let kPeRaw = compressedKV[.ellipsis, kvLoraRank...]
        kvNope = kvALayerNorm(kvNope)
        var kPe = kPeRaw.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)
        let offset = cache?.offset ?? 0
        qPe = rope(qPe, offset: offset)
        kPe = rope(kPe, offset: offset)

        if cache is GLM47CompressedMLACache,
           let absorbed = resolvedAbsorbedMLA() {
            var latentKeys = kvNope.expandedDimensions(axis: 1)
            var ropeKeys = kPe
            var latentValues = latentKeys
            if let cache {
                let compressedKeys = concatenated([latentKeys, ropeKeys], axis: -1)
                let cached = cache.update(keys: compressedKeys, values: latentValues)
                latentKeys = cached.0[.ellipsis, 0..<kvLoraRank]
                ropeKeys = cached.0[.ellipsis, kvLoraRank...]
                latentValues = cached.1
            }
            var out = absorbed.attend(
                qNope: qNope,
                qPe: qPe,
                latentKeys: latentKeys,
                ropeKeys: ropeKeys,
                latentValues: latentValues,
                scale: scale,
                mask: mask
            )
            out = out.asType(qStates.dtype)
            out = out.transposed(0, 2, 1, 3).reshaped(B, L, -1)
            return oProj(out)
        }

        let kv = kvBProj(kvNope).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let kNope = kv[.ellipsis, 0..<qkNopeHeadDim]
        var values = kv[.ellipsis, qkNopeHeadDim...]

        kPe = repeatAlongHeads(kPe, heads: numHeads)

        var keys = MLX.concatenated([kNope, kPe], axis: -1)

        if let cache {
            let cached = cache.update(keys: keys, values: values)
            keys = cached.0
            values = cached.1
        }

        let queries = MLX.concatenated([qNope, qPe], axis: -1)

        let qF32 = queries.asType(.float32)
        let kF32 = keys.asType(.float32)
        let vF32 = values.asType(.float32)
        var out = MLXFast.scaledDotProductAttention(
            queries: qF32,
            keys: kF32,
            values: vF32,
            scale: scale,
            mask: mask
        )
        out = out.asType(queries.dtype)
        out = out.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(out)
    }

    private func resolvedAbsorbedMLA() -> GLM47AbsorbedMLAWeights? {
        let source = ObjectIdentifier(kvBProj)
        if let cachedAbsorbedMLA, cachedAbsorbedMLA.source == source {
            return cachedAbsorbedMLA.weights
        }

        let denseWeight: MLXArray
        if type(of: kvBProj) == Linear.self {
            denseWeight = kvBProj.weight
        } else if (type(of: kvBProj) == QuantizedLinear.self
            || type(of: kvBProj) == PortableQuantizedLinear.self),
            let quantized = kvBProj as? QuantizedLinear,
            quantized.bias == nil {
            denseWeight = MLX.dequantized(
                quantized.weight,
                scales: quantized.scales,
                biases: quantized.biases,
                groupSize: quantized.groupSize,
                bits: quantized.bits,
                mode: quantized.mode,
                dtype: quantized.scales.dtype
            )
        } else {
            return nil
        }

        let expectedRows = numHeads * (qkNopeHeadDim + vHeadDim)
        guard denseWeight.ndim == 2,
              denseWeight.dim(0) == expectedRows,
              denseWeight.dim(1) == kvLoraRank else {
            return nil
        }
        let perHead = denseWeight.reshaped(
            numHeads,
            qkNopeHeadDim + vHeadDim,
            kvLoraRank
        )
        let weights = GLM47AbsorbedMLAWeights(
            key: perHead[0..., 0..<qkNopeHeadDim, 0...],
            value: perHead[0..., qkNopeHeadDim..., 0...]
        )
        MLX.eval(weights.key, weights.value)
        cachedAbsorbedMLA = CachedAbsorbedMLA(weights: weights, source: source)
        return weights
    }
}

private protocol GLM47MLP: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray
}

private final class GLM47FlashDenseMLP: Module, GLM47MLP {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: GLM47FlashConfig, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        let inDim = hiddenSize ?? config.hiddenSize
        let hiddenDim = intermediateSize ?? config.intermediateSize
        self._gateProj.wrappedValue = Linear(inDim, hiddenDim, bias: false)
        self._upProj.wrappedValue = Linear(inDim, hiddenDim, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDim, inDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(swiglu(gateProj(x), upProj(x)))
    }
}

private final class GLM47FlashMoE: Module, GLM47MLP {
    @ModuleInfo(key: "switch_mlp") var switchMLP: GLM47SwitchGLU
    @ModuleInfo(key: "gate") var gate: GLM47MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM47FlashDenseMLP?

    init(config: GLM47FlashConfig) {
        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 8
        let experts = config.nRoutedExperts ?? 0
        self._switchMLP.wrappedValue = GLM47SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: experts,
            groupSize: groupSize,
            bits: bits
        )
        self._gate.wrappedValue = GLM47MoEGate(config: config)
        if let shared = config.nSharedExperts, shared > 0 {
            self._sharedExperts.wrappedValue = GLM47FlashDenseMLP(
                config: config,
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize * shared
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, indices: inds)
        y = (y * scores.expandedDimensions(axis: scores.ndim)).sum(axis: -2)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

fileprivate final class GLM47FlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: GLM47FlashAttention
    let mlp: GLM47MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: GLM47FlashConfig, layerIndex: Int) {
        self._selfAttention.wrappedValue = GLM47FlashAttention(config: config)
        if let routed = config.nRoutedExperts,
           layerIndex >= config.firstKDenseReplace,
           layerIndex % 1 == 0,
           routed > 0 {
            self.mlp = GLM47FlashMoE(config: config)
        } else {
            self.mlp = GLM47FlashDenseMLP(config: config)
        }
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let r = selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let mlpOut = mlp(postAttentionLayerNorm(h))
        return h + mlpOut
    }
}

public final class GLM47FlashModel: Module {
    @ModuleInfo(key: "model") var model: GLM47FlashTransformer
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(config: GLM47FlashConfig) {
        self._model.wrappedValue = GLM47FlashTransformer(config: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    func callAsFunction(_ inputIds: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputIds, cache: cache)
        return lmHead(hidden)
    }

    /// Logits for the final position only. Prefill never reads the other
    /// positions' logits, and skipping them avoids a [prompt, vocab] lm_head
    /// matmul plus its materialization.
    func lastPositionLogits(_ inputIds: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = model(inputIds, cache: cache)
        let sequenceLength = hidden.dim(1)
        if sequenceLength > 1 {
            hidden = hidden[0..., (sequenceLength - 1)..., 0...]
        }
        return lmHead(hidden)
    }
}

public final class GLM47FlashTransformer: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") fileprivate var layers: [GLM47FlashDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: GLM47FlashConfig) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map {
            GLM47FlashDecoderLayer(config: config, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputIds: MLXArray, cache: [KVCache]?) -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }

        var h = embedTokens(tokenIds)
        let mask = createMask(h: h, cache: cache?.first)

        for (idx, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[idx])
        }
        return norm(h)
    }
}
