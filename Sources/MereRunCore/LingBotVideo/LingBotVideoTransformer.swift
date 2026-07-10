// Architecture derived from Robbyant LingBot-Video's Apache-2.0 reference implementation.
import Foundation
import MLX
import MLXFast
import MLXNN

private final class LingBotVideoFusedProjections {
    private let weight: MLXArray
    private let bias: MLXArray?
    private let splitIndices: [Int]
    private let sourceIDs: [ObjectIdentifier]

    private init(
        weight: MLXArray,
        bias: MLXArray?,
        splitIndices: [Int],
        sourceIDs: [ObjectIdentifier]
    ) {
        self.weight = weight
        self.bias = bias
        self.splitIndices = splitIndices
        self.sourceIDs = sourceIDs
    }

    static func fuse(_ projections: [Linear]) -> LingBotVideoFusedProjections? {
        guard projections.count >= 2, let first = projections.first else { return nil }
        guard projections.allSatisfy({
            type(of: $0) == Linear.self
                && $0.weight.ndim == 2
                && $0.weight.dim(1) == first.weight.dim(1)
                && $0.weight.dtype == first.weight.dtype
        }) else {
            return nil
        }

        let biasesPresent = projections.allSatisfy { $0.bias != nil }
        let biasesAbsent = projections.allSatisfy { $0.bias == nil }
        guard biasesPresent || biasesAbsent else { return nil }

        var splitIndices: [Int] = []
        var runningTotal = 0
        for projection in projections.dropLast() {
            runningTotal += projection.weight.dim(0)
            splitIndices.append(runningTotal)
        }
        let weight = MLX.concatenated(projections.map(\.weight), axis: 0)
        let bias = biasesPresent
            ? MLX.concatenated(projections.compactMap(\.bias), axis: 0)
            : nil
        var sourceIDs = projections.map(ObjectIdentifier.init)
        sourceIDs.append(contentsOf: projections.map { ObjectIdentifier($0.weight) })
        sourceIDs.append(contentsOf: projections.compactMap(\.bias).map(ObjectIdentifier.init))
        return LingBotVideoFusedProjections(
            weight: weight,
            bias: bias,
            splitIndices: splitIndices,
            sourceIDs: sourceIDs
        )
    }

    func matches(_ projections: [Linear]) -> Bool {
        var currentIDs = projections.map(ObjectIdentifier.init)
        currentIDs.append(contentsOf: projections.map { ObjectIdentifier($0.weight) })
        currentIDs.append(contentsOf: projections.compactMap(\.bias).map(ObjectIdentifier.init))
        return currentIDs == sourceIDs
    }

    func callSplit(_ x: MLXArray) -> [MLXArray] {
        var output = MLX.matmul(x, weight.T)
        if let bias {
            output = output + bias
        }
        return MLX.split(output, indices: splitIndices, axis: -1)
    }

    var arrays: [MLXArray] {
        if let bias {
            return [weight, bias]
        }
        return [weight]
    }
}

private enum LingBotVideoFusionPolicy {
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_LINGBOT_FUSED_PROJECTIONS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw != "0" && raw != "false" && raw != "off"
    }()
}

final class LingBotVideoRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    private let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLX.ones([dimensions], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight.asType(x.dtype), eps: eps)
    }
}

final class LingBotVideoTimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(frequencyDimension: Int, hiddenSize: Int, bias: Bool) {
        self._linear1.wrappedValue = Linear(frequencyDimension, hiddenSize, bias: bias)
        self._linear2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: bias)
        super.init()
    }

    func callAsFunction(_ timestep: MLXArray, frequencyDimension: Int) -> MLXArray {
        let projected = Self.project(timestep, dimension: frequencyDimension)
        return linear2(silu(linear1(projected)))
    }

    private static func project(_ timestep: MLXArray, dimension: Int) -> MLXArray {
        let half = dimension / 2
        let exponent = -Foundation.log(10_000.0) * MLXArray(0..<half).asType(.float32)
            / MLXArray(Float(half))
        let frequencies = MLX.exp(exponent)
        let args = timestep.asType(.float32).reshaped(-1, 1) * frequencies.reshaped(1, half)
        var embedding = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
        if dimension % 2 == 1 {
            embedding = padded(embedding, widths: [[0, 0], [0, 1]])
        }
        return embedding
    }
}

final class LingBotVideoModulation: Module {
    @ModuleInfo(key: "linear") var linear: Linear

    init(hiddenSize: Int, outputSize: Int) {
        self._linear.wrappedValue = Linear(hiddenSize, outputSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear(silu(x))
    }
}

final class LingBotVideoTextEmbedder: Module {
    @ModuleInfo(key: "norm") var norm: LingBotVideoRMSNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(textDimension: Int, hiddenSize: Int) {
        self._norm.wrappedValue = LingBotVideoRMSNorm(dimensions: textDimension, eps: 1e-6)
        self._linear1.wrappedValue = Linear(textDimension, hiddenSize, bias: true)
        self._linear2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(norm(x))))
    }
}

final class LingBotVideoAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "norm_q") var normQ: LingBotVideoRMSNorm
    @ModuleInfo(key: "norm_k") var normK: LingBotVideoRMSNorm

    private let numHeads: Int
    private let headDim: Int
    private let scale: Float
    private var fusedQKV: LingBotVideoFusedProjections?

    init(config: LingBotVideoTransformerConfig) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self._toQ.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
        self._toK.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
        self._toV.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
        self._toOut.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.outBias)
        self._normQ.wrappedValue = LingBotVideoRMSNorm(dimensions: config.headDim, eps: config.normEps)
        self._normK.wrappedValue = LingBotVideoRMSNorm(dimensions: config.headDim, eps: config.normEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        rotaryCos: MLXArray,
        rotarySin: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let rawQuery: MLXArray
        let rawKey: MLXArray
        let rawValue: MLXArray
        if let fused = resolvedFusedQKV() {
            let projections = fused.callSplit(x)
            rawQuery = projections[0]
            rawKey = projections[1]
            rawValue = projections[2]
        } else {
            rawQuery = toQ(x)
            rawKey = toK(x)
            rawValue = toV(x)
        }
        var query = normQ(rawQuery.reshaped(batch, length, numHeads, headDim))
        var key = normK(rawKey.reshaped(batch, length, numHeads, headDim))
        let value = rawValue.reshaped(batch, length, numHeads, headDim)

        query = Self.applyRotary(query, cos: rotaryCos, sin: rotarySin)
        key = Self.applyRotary(key, cos: rotaryCos, sin: rotarySin)

        let attended = MLXFast.scaledDotProductAttention(
            queries: query.transposed(0, 2, 1, 3),
            keys: key.transposed(0, 2, 1, 3),
            values: value.transposed(0, 2, 1, 3),
            scale: scale,
            mask: attentionMask
        )
        return toOut(attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }

    func prepareInferenceCaches() -> [MLXArray] {
        resolvedFusedQKV()?.arrays ?? []
    }

    private func resolvedFusedQKV() -> LingBotVideoFusedProjections? {
        guard LingBotVideoFusionPolicy.enabled else { return nil }
        let projections = [toQ, toK, toV]
        if let fusedQKV, fusedQKV.matches(projections) {
            return fusedQKV
        }
        fusedQKV = LingBotVideoFusedProjections.fuse(projections)
        return fusedQKV
    }

    private static func applyRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let shape = x.shape
        let pairs = x.reshaped(Array(shape.dropLast()) + [shape.last! / 2, 2]).asType(.float32)
        let real = pairs[0..., 0..., 0..., 0..., 0]
        let imaginary = pairs[0..., 0..., 0..., 0..., 1]
        let rotatedReal = real * cos - imaginary * sin
        let rotatedImaginary = real * sin + imaginary * cos
        return MLX.stacked([rotatedReal, rotatedImaginary], axis: -1).reshaped(shape).asType(x.dtype)
    }
}

final class LingBotVideoMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    private var fusedGateUp: LingBotVideoFusedProjections?

    init(config: LingBotVideoTransformerConfig, intermediateSize: Int? = nil) {
        let resolvedIntermediateSize = intermediateSize ?? config.intermediateSize
        self._gateProj.wrappedValue = Linear(config.hiddenSize, resolvedIntermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, resolvedIntermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(resolvedIntermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate: MLXArray
        let up: MLXArray
        if let fused = resolvedFusedGateUp() {
            let projections = fused.callSplit(x)
            gate = projections[0]
            up = projections[1]
        } else {
            gate = gateProj(x)
            up = upProj(x)
        }
        return downProj(silu(gate) * up)
    }

    func prepareInferenceCaches() -> [MLXArray] {
        resolvedFusedGateUp()?.arrays ?? []
    }

    private func resolvedFusedGateUp() -> LingBotVideoFusedProjections? {
        guard LingBotVideoFusionPolicy.enabled else { return nil }
        let projections = [gateProj, upProj]
        if let fusedGateUp, fusedGateUp.matches(projections) {
            return fusedGateUp
        }
        fusedGateUp = LingBotVideoFusedProjections.fuse(projections)
        return fusedGateUp
    }
}

final class LingBotVideoRouter: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    private let numExperts: Int
    private let topK: Int
    private let scoreFunction: String
    private let normalizeTopK: Bool
    private let groupCount: Int?
    private let selectedGroupCount: Int?
    private let routeScale: Float

    init(config: LingBotVideoTransformerConfig) {
        self.numExperts = config.numExperts
        self.topK = config.numExpertsPerTok
        self.scoreFunction = config.scoreFunc
        self.normalizeTopK = config.normTopKProb
        self.groupCount = config.nGroup
        self.selectedGroupCount = config.topKGroup
        self.routeScale = config.routedScalingFactor
        self._weight.wrappedValue = MLX.zeros([config.numExperts, config.hiddenSize], dtype: .float32)
        self._correctionBias.wrappedValue = MLX.zeros([config.numExperts], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        let logits = MLX.matmul(tokens.asType(.float32), weight.asType(.float32).T)
        let scores = scoreFunction == "softmax"
            ? softmax(logits, axis: -1, precise: true)
            : sigmoid(logits)
        let choiceScores = scores + correctionBias
        let indices = selectedExperts(choiceScores)
        var topScores = takeAlong(scores, indices, axis: -1)
        if normalizeTopK, topK > 1 {
            topScores = topScores / (topScores.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, (topScores * routeScale).asType(tokens.dtype))
    }

    private func selectedExperts(_ scores: MLXArray) -> MLXArray {
        guard let groupCount,
              let selectedGroupCount,
              groupCount > 1,
              selectedGroupCount < groupCount,
              numExperts % groupCount == 0
        else {
            return argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, 0..<topK]
        }

        let tokenCount = scores.size / numExperts
        let expertsPerGroup = numExperts / groupCount
        let grouped = scores.reshaped(tokenCount, groupCount, expertsPerGroup)
        let pairIndices = argPartition(-grouped, kth: 1, axis: -1)[.ellipsis, 0..<2]
        let groupScores = takeAlong(grouped, pairIndices, axis: -1).sum(axis: -1)
        let selectedGroups = argPartition(
            -groupScores,
            kth: selectedGroupCount - 1,
            axis: -1
        )[.ellipsis, 0..<selectedGroupCount]

        let expertGroups = MLXArray(0..<numExperts).asType(.int32)
            .floorDivide(expertsPerGroup)
            .reshaped(1, 1, numExperts)
        let allowed = (
            selectedGroups.expandedDimensions(axis: -1) .== expertGroups
        ).asType(.int32).sum(axis: 1) .> 0
        let flatScores = scores.reshaped(tokenCount, numExperts)
        let masked = MLX.where(allowed, flatScores, MLXArray(-Float.infinity))
        return argPartition(-masked, kth: topK - 1, axis: -1)[.ellipsis, 0..<topK]
            .reshaped(Array(scores.shape.dropLast()) + [topK])
    }
}

final class LingBotVideoGroupedExperts: Module {
    @ModuleInfo(key: "w1") var w1: Q35SwitchLinear
    @ModuleInfo(key: "w2") var w2: Q35SwitchLinear
    @ModuleInfo(key: "w3") var w3: Q35SwitchLinear

    init(
        config: LingBotVideoTransformerConfig,
        quantization: LingBotVideoQuantizationConfig?
    ) {
        let groupSize = quantization?.groupSize ?? 64
        let bits = quantization?.bits ?? 4
        let quantized = quantization != nil
        self._w1.wrappedValue = Q35SwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: config.moeIntermediateSize,
            numExperts: config.numExperts,
            groupSize: groupSize,
            bits: bits,
            quantized: quantized,
            bias: false
        )
        self._w2.wrappedValue = Q35SwitchLinear(
            inputDims: config.moeIntermediateSize,
            outputDims: config.hiddenSize,
            numExperts: config.numExperts,
            groupSize: groupSize,
            bits: bits,
            quantized: quantized,
            bias: false
        )
        self._w3.wrappedValue = Q35SwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: config.moeIntermediateSize,
            numExperts: config.numExperts,
            groupSize: groupSize,
            bits: bits,
            quantized: quantized,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let activated = silu(w1(x, indices: indices)) * w3(x, indices: indices)
        return w2(activated, indices: indices)
    }
}

final class LingBotVideoFeedForward: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear?
    @ModuleInfo(key: "up_proj") var upProj: Linear?
    @ModuleInfo(key: "down_proj") var downProj: Linear?
    @ModuleInfo(key: "router") var router: LingBotVideoRouter?
    @ModuleInfo(key: "experts") var experts: LingBotVideoGroupedExperts?
    @ModuleInfo(key: "shared_experts") var sharedExperts: LingBotVideoMLP?

    private let usesMoE: Bool
    private var fusedGateUp: LingBotVideoFusedProjections?

    init(
        config: LingBotVideoTransformerConfig,
        quantization: LingBotVideoQuantizationConfig?
    ) {
        self.usesMoE = config.numExperts > 0
        if usesMoE {
            self._gateProj.wrappedValue = nil
            self._upProj.wrappedValue = nil
            self._downProj.wrappedValue = nil
            self._router.wrappedValue = LingBotVideoRouter(config: config)
            self._experts.wrappedValue = LingBotVideoGroupedExperts(
                config: config,
                quantization: quantization
            )
            let sharedCount = config.nSharedExperts ?? 0
            self._sharedExperts.wrappedValue = sharedCount > 0
                ? LingBotVideoMLP(
                    config: config,
                    intermediateSize: config.moeIntermediateSize * sharedCount
                )
                : nil
        } else {
            self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
            self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
            self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
            self._router.wrappedValue = nil
            self._experts.wrappedValue = nil
            self._sharedExperts.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if usesMoE, let router, let experts {
            let routing = router(x)
            let expertOutput = experts(x, indices: routing.indices)
            let routed = (
                expertOutput.asType(.float32)
                    * routing.scores.expandedDimensions(axis: -1).asType(.float32)
            ).sum(axis: -2).asType(x.dtype)
            if let sharedExperts {
                return routed + sharedExperts(x)
            }
            return routed
        }

        guard let gateProj, let upProj, let downProj else { return x }
        let gate: MLXArray
        let up: MLXArray
        if let fused = resolvedFusedGateUp(gate: gateProj, up: upProj) {
            let projections = fused.callSplit(x)
            gate = projections[0]
            up = projections[1]
        } else {
            gate = gateProj(x)
            up = upProj(x)
        }
        return downProj(silu(gate) * up)
    }

    func prepareInferenceCaches() -> [MLXArray] {
        if usesMoE {
            return sharedExperts?.prepareInferenceCaches() ?? []
        }
        guard let gateProj, let upProj else { return [] }
        return resolvedFusedGateUp(gate: gateProj, up: upProj)?.arrays ?? []
    }

    private func resolvedFusedGateUp(
        gate: Linear,
        up: Linear
    ) -> LingBotVideoFusedProjections? {
        guard LingBotVideoFusionPolicy.enabled else { return nil }
        let projections = [gate, up]
        if let fusedGateUp, fusedGateUp.matches(projections) {
            return fusedGateUp
        }
        fusedGateUp = LingBotVideoFusedProjections.fuse(projections)
        return fusedGateUp
    }
}

final class LingBotVideoBlock: Module {
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ModuleInfo(key: "norm1") var norm1: LingBotVideoRMSNorm
    @ModuleInfo(key: "attn") var attention: LingBotVideoAttention
    @ModuleInfo(key: "norm_post_attn") var normPostAttention: LingBotVideoRMSNorm
    @ModuleInfo(key: "norm2") var norm2: LingBotVideoRMSNorm
    @ModuleInfo(key: "ffn") var feedForward: LingBotVideoFeedForward
    @ModuleInfo(key: "norm_post_ffn") var normPostFeedForward: LingBotVideoRMSNorm

    private let hiddenSize: Int

    init(
        config: LingBotVideoTransformerConfig,
        quantization: LingBotVideoQuantizationConfig?
    ) {
        self.hiddenSize = config.hiddenSize
        self._scaleShiftTable.wrappedValue = MLX.zeros([1, 6, config.hiddenSize], dtype: .float32)
        self._norm1.wrappedValue = LingBotVideoRMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        self._attention.wrappedValue = LingBotVideoAttention(config: config)
        self._normPostAttention.wrappedValue = LingBotVideoRMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        self._norm2.wrappedValue = LingBotVideoRMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        self._feedForward.wrappedValue = LingBotVideoFeedForward(
            config: config,
            quantization: quantization
        )
        self._normPostFeedForward.wrappedValue = LingBotVideoRMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        timestepModulation: MLXArray,
        rotaryCos: MLXArray,
        rotarySin: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let modulation = timestepModulation.asType(.float32) + scaleShiftTable
        let shiftAttention = modulation[0..., 0..., 0, 0...]
        let scaleAttention = modulation[0..., 0..., 1, 0...]
        let gateAttention = MLX.tanh(modulation[0..., 0..., 2, 0...])
        let shiftMLP = modulation[0..., 0..., 3, 0...]
        let scaleMLP = modulation[0..., 0..., 4, 0...]
        let gateMLP = MLX.tanh(modulation[0..., 0..., 5, 0...])

        let bulkType = x.dtype
        let attentionInput = (
            norm1(x).asType(.float32) * (1 + scaleAttention) + shiftAttention
        ).asType(bulkType)
        let attentionOutput = attention(
            attentionInput,
            rotaryCos: rotaryCos,
            rotarySin: rotarySin,
            attentionMask: attentionMask
        )
        var hidden = (
            x.asType(.float32) + gateAttention * normPostAttention(attentionOutput).asType(.float32)
        ).asType(bulkType)

        let mlpInput = (
            norm2(hidden).asType(.float32) * (1 + scaleMLP) + shiftMLP
        ).asType(bulkType)
        let mlpOutput = feedForward(mlpInput)
        hidden = (
            hidden.asType(.float32) + gateMLP * normPostFeedForward(mlpOutput).asType(.float32)
        ).asType(bulkType)
        return hidden
    }

    func prepareInferenceCaches() -> [MLXArray] {
        attention.prepareInferenceCaches() + feedForward.prepareInferenceCaches()
    }
}

public final class LingBotVideoTransformer: Module {
    public let config: LingBotVideoTransformerConfig

    @ModuleInfo(key: "patch_embedder") var patchEmbedder: Linear
    @ModuleInfo(key: "time_embedder") var timeEmbedder: LingBotVideoTimestepEmbedding
    @ModuleInfo(key: "time_modulation") var timeModulation: LingBotVideoModulation
    @ModuleInfo(key: "text_embedder") var textEmbedder: LingBotVideoTextEmbedder
    @ModuleInfo(key: "blocks") var blocks: [LingBotVideoBlock]
    @ModuleInfo(key: "norm_out_modulation") var normOutModulation: LingBotVideoModulation
    @ModuleInfo(key: "proj_out") var projectionOut: Linear

    private var rotaryCache: [String: (cos: MLXArray, sin: MLXArray)] = [:]

    public init(
        config: LingBotVideoTransformerConfig,
        quantization: LingBotVideoQuantizationConfig? = nil
    ) {
        self.config = config
        let patchVolume = config.patchSize.reduce(1, *)
        self._patchEmbedder.wrappedValue = Linear(
            config.inChannels * patchVolume,
            config.hiddenSize,
            bias: config.patchEmbedBias
        )
        self._timeEmbedder.wrappedValue = LingBotVideoTimestepEmbedding(
            frequencyDimension: config.freqDim,
            hiddenSize: config.hiddenSize,
            bias: config.timestepMLPBias
        )
        self._timeModulation.wrappedValue = LingBotVideoModulation(
            hiddenSize: config.hiddenSize,
            outputSize: config.hiddenSize * 6
        )
        self._textEmbedder.wrappedValue = LingBotVideoTextEmbedder(
            textDimension: config.textDim,
            hiddenSize: config.hiddenSize
        )
        self._blocks.wrappedValue = (0..<config.depth).map { _ in
            LingBotVideoBlock(config: config, quantization: quantization)
        }
        self._normOutModulation.wrappedValue = LingBotVideoModulation(
            hiddenSize: config.hiddenSize,
            outputSize: config.hiddenSize * 2
        )
        self._projectionOut.wrappedValue = Linear(
            config.hiddenSize,
            config.outChannels * patchVolume,
            bias: true
        )
        super.init()
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        timestep: MLXArray,
        encoderHiddenStates: MLXArray,
        encoderAttentionMask: MLXArray? = nil,
        encoderTextLengths: [Int]? = nil,
        blockProgressInterval: Int = 1,
        blockProgressHandler: ((Int, Int) -> Void)? = nil
    ) -> MLXArray {
        precondition(hiddenStates.ndim == 5, "LingBot video latents must be [B,C,T,H,W].")

        let batch = hiddenStates.dim(0)
        precondition(encoderHiddenStates.dim(0) == batch)
        precondition(timestep.dim(0) == batch)
        let channels = hiddenStates.dim(1)
        let frames = hiddenStates.dim(2)
        let height = hiddenStates.dim(3)
        let width = hiddenStates.dim(4)
        let patchFrames = config.patchSize[0]
        let patchHeight = config.patchSize[1]
        let patchWidth = config.patchSize[2]
        let gridFrames = frames / patchFrames
        let gridHeight = height / patchHeight
        let gridWidth = width / patchWidth
        let videoTokenCount = gridFrames * gridHeight * gridWidth
        let textLength = encoderHiddenStates.dim(1)

        var patches = hiddenStates.reshaped(
            batch, channels,
            gridFrames, patchFrames,
            gridHeight, patchHeight,
            gridWidth, patchWidth
        )
        patches = patches.transposed(0, 2, 4, 6, 3, 5, 7, 1)
            .reshaped(batch, videoTokenCount, patchFrames * patchHeight * patchWidth * channels)

        let video = patchEmbedder(patches.asType(.bfloat16))
        let text = textEmbedder(encoderHiddenStates.asType(.bfloat16))
        var joint = MLX.concatenated([video, text], axis: 1)
        let jointLength = joint.dim(1)
        let resolvedTextLengths = encoderTextLengths ?? [Int](repeating: textLength, count: batch)
        precondition(resolvedTextLengths.count == batch)
        precondition(resolvedTextLengths.allSatisfy { $0 > 0 && $0 <= textLength })
        let rotary = rotaryEmbeddings(
            textLengths: resolvedTextLengths,
            paddedTextLength: textLength,
            gridFrames: gridFrames,
            gridHeight: gridHeight,
            gridWidth: gridWidth
        )

        let embeddedTime = timeEmbedder(timestep.asType(.float32), frequencyDimension: config.freqDim)
        let timeInput = MLX.broadcast(
            embeddedTime.expandedDimensions(axis: 1),
            to: [batch, jointLength, config.hiddenSize]
        )
        let timestepModulation = timeModulation(timeInput.reshaped(batch * jointLength, config.hiddenSize))
            .reshaped(batch, jointLength, 6, config.hiddenSize)
        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let encoderAttentionMask {
            precondition(encoderAttentionMask.shape == [batch, textLength])
            let videoMask = MLX.ones([batch, videoTokenCount], dtype: .bool)
            let textMask = encoderAttentionMask.asType(.bool)
            let jointMask = MLX.concatenated([videoMask, textMask], axis: 1)
                .reshaped(batch, 1, 1, jointLength)
            attentionMask = .array(jointMask)
        } else {
            attentionMask = .none
        }

        for (index, block) in blocks.enumerated() {
            joint = block(
                joint,
                timestepModulation: timestepModulation,
                rotaryCos: rotary.cos,
                rotarySin: rotary.sin,
                attentionMask: attentionMask
            )
            let completedBlocks = index + 1
            let shouldReportBlock = completedBlocks == 1
                || completedBlocks == blocks.count
                || completedBlocks % max(1, blockProgressInterval) == 0
            if let blockProgressHandler, shouldReportBlock {
                MLX.eval(joint)
                blockProgressHandler(completedBlocks, blocks.count)
            }
        }

        let finalModulation = normOutModulation(timeInput.reshaped(batch * jointLength, config.hiddenSize))
            .reshaped(batch, jointLength, 2, config.hiddenSize)
        let shift = finalModulation[0..., 0..., 0, 0...]
        let scale = finalModulation[0..., 0..., 1, 0...]
        let normalized = Self.layerNorm(joint, eps: config.normEps)
        let projected = projectionOut((normalized * (1 + scale) + shift).asType(joint.dtype))
        let videoOutput = projected[0..., 0..<videoTokenCount, 0...]

        return videoOutput
            .reshaped(batch, gridFrames, gridHeight, gridWidth, patchFrames, patchHeight, patchWidth, config.outChannels)
            .transposed(0, 7, 1, 4, 2, 5, 3, 6)
            .reshaped(batch, config.outChannels, frames, height, width)
            .asType(.float32)
    }

    public func prepareInferenceCaches() {
        let arrays = blocks.flatMap { $0.prepareInferenceCaches() }
        if !arrays.isEmpty {
            MLX.eval(arrays)
        }
    }

    static func mapWeight(key: String, value: MLXArray, config: LingBotVideoTransformerConfig) -> [(String, MLXArray)] {
        let mappedKey = mapWeightKey(key)
        var mappedValue = value
        if mappedKey.hasSuffix("scale_shift_table"), value.ndim == 2 {
            mappedValue = value.reshaped(1, 6, config.hiddenSize)
        }
        return [(mappedKey, mappedValue)]
    }

    static func mapWeightKey(_ key: String) -> String {
        let mapped = key
            .replacingOccurrences(of: "time_modulation.1.", with: "time_modulation.linear.")
            .replacingOccurrences(of: "norm_out_modulation.1.", with: "norm_out_modulation.linear.")
        if mapped.hasSuffix(".ffn.experts.w1")
            || mapped.hasSuffix(".ffn.experts.w2")
            || mapped.hasSuffix(".ffn.experts.w3") {
            return mapped + ".weight"
        }
        return mapped
    }

    private static func layerNorm(_ x: MLXArray, eps: Float) -> MLXArray {
        MLXFast.layerNorm(x, eps: eps)
    }

    private func rotaryEmbeddings(
        textLengths: [Int],
        paddedTextLength: Int,
        gridFrames: Int,
        gridHeight: Int,
        gridWidth: Int
    ) -> (cos: MLXArray, sin: MLXArray) {
        let key = "\(textLengths.map(String.init).joined(separator: ",")):\(paddedTextLength):\(gridFrames):\(gridHeight):\(gridWidth)"
        if let cached = rotaryCache[key] {
            return cached
        }

        var positions: [(Int, Int, Int)] = []
        let positionsPerSample = gridFrames * gridHeight * gridWidth + paddedTextLength
        positions.reserveCapacity(positionsPerSample * textLengths.count)
        for textLength in textLengths {
            for frame in 0..<gridFrames {
                for row in 0..<gridHeight {
                    for column in 0..<gridWidth {
                        positions.append((textLength + 1 + frame, row, column))
                    }
                }
            }
            for token in 0..<paddedTextLength {
                positions.append(token < textLength ? (token + 1, 0, 0) : (0, 0, 0))
            }
        }

        let pairCount = config.headDim / 2
        var cosines: [Float] = []
        var sines: [Float] = []
        cosines.reserveCapacity(positions.count * pairCount)
        sines.reserveCapacity(positions.count * pairCount)
        for position in positions {
            let axisPositions = [position.0, position.1, position.2]
            for axis in 0..<3 {
                let dimension = config.axesDims[axis]
                for pair in 0..<(dimension / 2) {
                    let frequency = pow(Double(config.ropeTheta), -Double(pair * 2) / Double(dimension))
                    let angle = Double(axisPositions[axis]) * frequency
                    cosines.append(Float(Foundation.cos(angle)))
                    sines.append(Float(Foundation.sin(angle)))
                }
            }
        }

        let shape = [textLengths.count, positionsPerSample, 1, pairCount]
        let result = (
            cos: MLXArray(cosines, shape).asType(.float32),
            sin: MLXArray(sines, shape).asType(.float32)
        )
        rotaryCache[key] = result
        return result
    }
}
