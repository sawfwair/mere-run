import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

final class LagunaRoPE: Module, OffsetLayer {
    private let dimensions: Int
    private let traditional: Bool
    private let magnitudeScale: Float
    private let base: Float?
    private let frequencies: MLXArray?

    init(headDim: Int, parameters: LagunaRopeParameters) {
        let resolvedDimensions = max(2, Int(Float(headDim) * parameters.partialRotaryFactor))
        self.dimensions = resolvedDimensions
        self.traditional = false

        if parameters.ropeType == "yarn" {
            let factor = parameters.factor ?? 1
            let originalContext = parameters.originalMaxPositionEmbeddings ?? 4_096
            let betaFast = parameters.betaFast ?? 32
            let betaSlow = parameters.betaSlow ?? 1

            func correctionDimension(rotations: Float) -> Float {
                let numerator = Float(resolvedDimensions)
                    * log(Float(originalContext) / (rotations * 2 * Float.pi))
                return numerator / (2 * log(parameters.ropeTheta))
            }

            let low = max(Int(floor(correctionDimension(rotations: betaFast))), 0)
            let high = min(Int(ceil(correctionDimension(rotations: betaSlow))), resolvedDimensions - 1)
            let denominator: Float = low == high ? 0.001 : Float(high - low)
            let half = max(1, resolvedDimensions / 2)
            let halfIndices = MLXArray((0..<half).map(Float.init))
            let evenIndices = MLXArray(
                Array(stride(from: 0, to: resolvedDimensions, by: 2)).map(Float.init)
            )
            let frequencyExtra = MLX.pow(
                MLXArray(parameters.ropeTheta),
                evenIndices / Float(resolvedDimensions)
            )
            let frequencyInterpolated = MLXArray(factor) * frequencyExtra
            let ramp = MLX.clip(
                (halfIndices - Float(low)) / denominator,
                min: 0,
                max: 1
            )
            let frequencyMask = MLXArray(1) - ramp
            self.frequencies = (frequencyInterpolated * frequencyExtra)
                / (
                    frequencyInterpolated * frequencyMask
                        + frequencyExtra * (MLXArray(1) - frequencyMask)
                )
            self.magnitudeScale = parameters.attentionFactor
                ?? (factor <= 1 ? 1 : 0.1 * log(factor) + 1)
            self.base = nil
        } else {
            self.frequencies = nil
            self.magnitudeScale = 1
            self.base = parameters.ropeTheta
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        let input: MLXArray
        if magnitudeScale == 1 {
            input = x
        } else if dimensions < x.dim(-1) {
            input = concatenated(
                [
                    x[.ellipsis, ..<dimensions] * MLXArray(magnitudeScale).asType(x.dtype),
                    x[.ellipsis, dimensions...],
                ],
                axis: -1
            )
        } else {
            input = x * MLXArray(magnitudeScale).asType(x.dtype)
        }

        return MLXFast.RoPE(
            input,
            dimensions: dimensions,
            traditional: traditional,
            base: base,
            scale: 1,
            offset: offset,
            freqs: frequencies
        )
    }
}

class LagunaFeedForward: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("LagunaFeedForward subclasses must implement callAsFunction(_:).")
    }
}

final class LagunaDenseMLP: LagunaFeedForward {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(inputDimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(inputDimensions, hiddenDimensions, bias: false)
        self._upProj.wrappedValue = Linear(inputDimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, inputDimensions, bias: false)
        super.init()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(MLXNN.silu(gateProj(x)) * upProj(x))
    }
}

final class LagunaSwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray?
    @ModuleInfo(key: "biases") var biases: MLXArray?

    private let groupSize: Int
    private let bits: Int
    private let mode: QuantizationMode

    init(
        inputDimensions: Int,
        outputDimensions: Int,
        expertCount: Int,
        quantization: LagunaQuantizationConfig?
    ) {
        self.groupSize = quantization?.groupSize ?? 16
        self.bits = quantization?.bits ?? 4
        self.mode = quantization.flatMap { QuantizationMode(rawValue: $0.mode) } ?? .affine

        if quantization != nil {
            let packedInputDimensions = (inputDimensions * bits + 31) / 32
            self._weight.wrappedValue = MLXArray.zeros(
                [expertCount, outputDimensions, packedInputDimensions],
                dtype: .uint32
            )
            self._scales.wrappedValue = MLXArray.zeros(
                [expertCount, outputDimensions, max(1, inputDimensions / groupSize)]
            )
        } else {
            let scale = sqrt(1 / Float(max(1, inputDimensions)))
            self._weight.wrappedValue = MLXRandom.uniform(
                low: -scale,
                high: scale,
                [expertCount, outputDimensions, inputDimensions]
            )
            self._scales.wrappedValue = nil
        }
        self._biases.wrappedValue = nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let topK = indices.dim(2)
        let inputDimensions = x.dim(-1)
        let tokenCount = batch * sequenceLength

        let flatInput: MLXArray
        if x.ndim == 4 && x.dim(2) == topK {
            flatInput = x.reshaped([tokenCount * topK, 1, inputDimensions])
        } else {
            let expanded = MLX.repeated(
                MLX.expandedDimensions(x.reshaped([tokenCount, 1, inputDimensions]), axis: 1),
                count: topK,
                axis: 1
            )
            flatInput = expanded.reshaped([tokenCount * topK, 1, inputDimensions])
        }

        let flatIndices = indices.reshaped([tokenCount * topK])
        let output: MLXArray
        if let scales {
            output = portableGatherQuantizedMM(
                flatInput,
                weight,
                scales: scales,
                biases: biases,
                rhsIndices: flatIndices,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: mode,
                sortedIndices: false
            )
        } else {
            output = gatherMM(
                flatInput,
                weight.swappedAxes(-1, -2),
                rhsIndices: flatIndices,
                sortedIndices: false
            )
        }

        return output.reshaped([batch, sequenceLength, topK, output.dim(-1)])
    }
}

final class LagunaSwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: LagunaSwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: LagunaSwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: LagunaSwitchLinear

    init(config: LagunaConfig) {
        self._gateProj.wrappedValue = LagunaSwitchLinear(
            inputDimensions: config.hiddenSize,
            outputDimensions: config.moeIntermediateSize,
            expertCount: config.numExperts,
            quantization: config.quantization
        )
        self._upProj.wrappedValue = LagunaSwitchLinear(
            inputDimensions: config.hiddenSize,
            outputDimensions: config.moeIntermediateSize,
            expertCount: config.numExperts,
            quantization: config.quantization
        )
        self._downProj.wrappedValue = LagunaSwitchLinear(
            inputDimensions: config.moeIntermediateSize,
            outputDimensions: config.hiddenSize,
            expertCount: config.numExperts,
            quantization: config.quantization
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let gate = gateProj(x, indices: indices)
        let up = upProj(x, indices: indices)
        return downProj(MLXNN.silu(gate) * up, indices: indices)
    }
}

final class LagunaRouter: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    private let topK: Int
    private let normalize: Bool
    private let softcap: Float

    init(config: LagunaConfig) {
        self.topK = config.numExpertsPerToken
        self.normalize = config.normTopKProbability
        self.softcap = config.moeRouterLogitSoftcapping
        self._weight.wrappedValue = MLXArray.zeros([config.numExperts, config.hiddenSize])
        self._correctionBias.wrappedValue = MLXArray.zeros([config.numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        var logits = x.matmul(weight.T).asType(.float32)
        if softcap > 0 {
            logits = tanh(logits / softcap) * softcap
        }
        let scores = sigmoid(logits)
        let selectionScores = scores + correctionBias.asType(scores.dtype)
        let count = min(topK, selectionScores.dim(-1))
        let indices = argPartition(-selectionScores, kth: count - 1, axis: -1)[.ellipsis, ..<count]
        var weights = takeAlong(scores, indices, axis: -1)
        if normalize {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        return (indices, weights.asType(x.dtype))
    }
}

final class LagunaSparseMoE: LagunaFeedForward {
    @ModuleInfo(key: "gate") var gate: LagunaRouter
    @ModuleInfo(key: "switch_mlp") var switchMLP: LagunaSwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: LagunaDenseMLP

    private let scalingFactor: Float

    init(config: LagunaConfig) {
        self._gate.wrappedValue = LagunaRouter(config: config)
        self._switchMLP.wrappedValue = LagunaSwitchGLU(config: config)
        self._sharedExpert.wrappedValue = LagunaDenseMLP(
            inputDimensions: config.hiddenSize,
            hiddenDimensions: config.sharedExpertIntermediateSize
        )
        self.scalingFactor = config.moeRoutedScalingFactor
        super.init()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let routed = gate(x)
        var expertOutput = switchMLP(x, indices: routed.indices)
        expertOutput = (
            expertOutput * MLX.expandedDimensions(routed.weights, axis: routed.weights.ndim)
        ).sum(axis: -2)
        if scalingFactor != 1 {
            expertOutput = expertOutput * scalingFactor
        }
        return expertOutput + sharedExpert(x)
    }
}

final class LagunaAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    private let headCount: Int
    private let keyValueHeadCount: Int
    private let headDim: Int
    private let scale: Float
    private let slidingWindow: Int?
    private let gatePerHead: Bool
    private let rope: LagunaRoPE

    init(config: LagunaConfig, layerIndex: Int) {
        self.headCount = config.attentionHeads(layerIndex: layerIndex)
        self.keyValueHeadCount = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.slidingWindow = config.layerTypes[layerIndex] == "sliding_attention"
            ? config.slidingWindow
            : nil
        self.gatePerHead = config.gating == "per-head"
        self.rope = LagunaRoPE(
            headDim: config.headDim,
            parameters: config.ropeParameters(layerIndex: layerIndex)
        )

        self._qProj.wrappedValue = Linear(
            config.hiddenSize,
            headCount * config.headDim,
            bias: config.attentionBias
        )
        self._kProj.wrappedValue = Linear(
            config.hiddenSize,
            keyValueHeadCount * config.headDim,
            bias: config.attentionBias
        )
        self._vProj.wrappedValue = Linear(
            config.hiddenSize,
            keyValueHeadCount * config.headDim,
            bias: config.attentionBias
        )
        self._oProj.wrappedValue = Linear(headCount * config.headDim, config.hiddenSize, bias: false)
        if config.gating == "none" || config.gating == "false" {
            self._gProj.wrappedValue = nil
        } else {
            let gateDimensions = gatePerHead ? headCount : headCount * config.headDim
            self._gProj.wrappedValue = Linear(config.hiddenSize, gateDimensions, bias: false)
        }
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Gemma4AttentionCache?) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let offset = cache?.offset ?? 0

        var queries = qProj(x).reshaped(batch, sequenceLength, headCount, headDim)
        var keys = kProj(x).reshaped(batch, sequenceLength, keyValueHeadCount, headDim)
        var values = vProj(x).reshaped(batch, sequenceLength, keyValueHeadCount, headDim)
        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        if let cache {
            cache.append(keys: keys, values: values)
            let state = sequenceLength == 1 ? cache.decodeState() : cache.currentState()
            keys = state!.0
            values = state!.1
        }

        let mask = attentionMask(
            queryLength: sequenceLength,
            queryOffset: offset,
            keyLength: keys.dim(2),
            dtype: x.dtype
        )
        var output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        ).transposed(0, 2, 1, 3)

        if let gProj {
            let gate = MLXNN.softplus(gProj(x).asType(.float32)).asType(output.dtype)
            if gatePerHead {
                output = output * MLX.expandedDimensions(gate, axis: gate.ndim)
            } else {
                output = output.reshaped(batch, sequenceLength, -1) * gate
                return oProj(output)
            }
        }
        return oProj(output.reshaped(batch, sequenceLength, -1))
    }

    private func attentionMask(
        queryLength: Int,
        queryOffset: Int,
        keyLength: Int,
        dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard queryLength > 1 else {
            return .none
        }

        let keyStart = max(0, queryOffset + queryLength - keyLength)
        let queryPositions = MLXArray(
            Int32(queryOffset)..<Int32(queryOffset + queryLength)
        ).reshaped(queryLength, 1)
        let keyPositions = MLXArray(
            Int32(keyStart)..<Int32(keyStart + keyLength)
        ).reshaped(1, keyLength)
        var allowed = keyPositions .<= queryPositions
        if let slidingWindow {
            allowed = allowed .&& (keyPositions .> (queryPositions - Int32(slidingWindow)))
        }

        let typed = allowed.asType(dtype).reshaped(1, 1, queryLength, keyLength)
        let zeros = MLXArray.zeros([1, 1, queryLength, keyLength], dtype: dtype)
        let negative = zeros + MLXArray(-1e9).asType(dtype)
        return .array(MLX.where(typed .> MLXArray(0).asType(dtype), zeros, negative))
    }
}

final class LagunaDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: LagunaAttention
    @ModuleInfo(key: "mlp") var mlp: LagunaFeedForward
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: LagunaConfig, layerIndex: Int) {
        self._selfAttention.wrappedValue = LagunaAttention(config: config, layerIndex: layerIndex)
        self._mlp.wrappedValue = config.isSparse(layerIndex: layerIndex)
            ? LagunaSparseMoE(config: config)
            : LagunaDenseMLP(
                inputDimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize
            )
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Gemma4AttentionCache?) -> MLXArray {
        let attended = x + selfAttention(inputLayerNorm(x), cache: cache)
        return attended + mlp(postAttentionLayerNorm(attended))
    }
}

final class LagunaLanguageModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [LagunaDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    private let config: LagunaConfig

    init(config: LagunaConfig) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map {
            LagunaDecoderLayer(config: config, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputIDs: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> MLXArray {
        var hidden = embedTokens(inputIDs)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: cache?[index])
        }
        return norm(hidden)
    }

    func makeCache() -> [Gemma4AttentionCache] {
        config.layerTypes.map { layerType in
            if layerType == "sliding_attention" {
                return Gemma4SlidingKVCache(maxSize: config.slidingWindow)
            }
            return Gemma4FullKVCache()
        }
    }
}

final class LagunaCausalLM: Module {
    @ModuleInfo(key: "model") var model: LagunaLanguageModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let config: LagunaConfig

    init(config: LagunaConfig) {
        self.config = config
        self._model.wrappedValue = LagunaLanguageModel(config: config)
        self._lmHead.wrappedValue = config.tieWordEmbeddings
            ? nil
            : Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    func callAsFunction(_ inputIDs: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> MLXArray {
        let hidden = model(inputIDs, cache: cache)
        return lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    func lastPositionLogits(
        _ inputIDs: MLXArray,
        cache: [Gemma4AttentionCache]? = nil
    ) -> MLXArray {
        var hidden = model(inputIDs, cache: cache)
        if hidden.dim(1) > 1 {
            hidden = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        }
        return lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    func makeCache() -> [Gemma4AttentionCache] {
        model.makeCache()
    }
}
