import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

@inline(__always)
private func inklingSwiGLU(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
    MLXNN.silu(gate) * up
}

public final class InklingConvCache: @unchecked Sendable {
    fileprivate var states: [MLXArray?] = Array(repeating: nil, count: 4)

    public init() {}

    public func fork() -> InklingConvCache {
        let copy = InklingConvCache()
        copy.states = states.map { $0?.asType($0!.dtype) }
        return copy
    }
}

public final class InklingLayerCache: @unchecked Sendable {
    let attention: KVCache
    let convolution: InklingConvCache

    public init(attention: KVCache = KVCacheSimple(step: 256)) {
        self.attention = attention
        self.convolution = InklingConvCache()
    }

    public func fork() -> InklingLayerCache {
        let copy = InklingLayerCache(attention: attention.fork())
        copy.convolution.states = convolution.states.map { $0?.asType($0!.dtype) }
        return copy
    }
}

private enum InklingBandedMask {
    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    private static let kernel = MLXFast.metalKernel(
        name: "mere_inkling_banded_relative_mask",
        inputNames: ["rel", "proj"],
        outputNames: ["out"],
        source: """
            uint j = thread_position_in_grid.x;
            uint i = thread_position_in_grid.y;
            uint bh = thread_position_in_grid.z;
            if (i >= LQ || j >= S || bh >= B * H) return;
            uint b = bh / H;
            uint h = bh % H;
            int dist = (int(i) + int(Q_OFFSET)) - int(j);
            T value;
            if (dist < 0) {
                value = T(-1e30f);
            } else if (SLIDING > 0 && dist >= int(SLIDING)) {
                value = T(-1e30f);
            } else if (dist < int(REL_EXTENT)) {
                float total = 0.0f;
                uint rel_base = ((b * LQ + i) * H + h) * D_REL;
                for (uint d = 0; d < D_REL; ++d) {
                    total += float(rel[rel_base + d]) * float(proj[d * REL_EXTENT + uint(dist)]);
                }
                value = T(total);
            } else {
                value = T(0.0f);
            }
            out[((b * H + h) * LQ + i) * S + j] = value;
        """,
        ensureRowContiguous: true
    )
    #endif

    static func make(
        relative: MLXArray,
        projection: MLXArray,
        queryOffset: Int,
        keyLength: Int,
        slidingWindow: Int,
        relativeExtent: Int
    ) -> MLXArray {
        let batch = relative.dim(0)
        let queryLength = relative.dim(1)
        let heads = relative.dim(2)
        let relativeDimensions = relative.dim(3)

        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        if Device.defaultDevice().deviceType == .gpu {
            let roundUp: (Int, Int) -> Int = { value, multiple in
                ((value + multiple - 1) / multiple) * multiple
            }
            return kernel(
                [relative, projection],
                template: [
                    ("T", relative.dtype),
                    ("B", batch),
                    ("H", heads),
                    ("LQ", queryLength),
                    ("S", keyLength),
                    ("Q_OFFSET", queryOffset),
                    ("D_REL", relativeDimensions),
                    ("REL_EXTENT", relativeExtent),
                    ("SLIDING", slidingWindow),
                ],
                grid: (roundUp(keyLength, 8), roundUp(queryLength, 8), batch * heads),
                threadGroup: (8, 8, 1),
                outputShapes: [[batch, heads, queryLength, keyLength]],
                outputDTypes: [relative.dtype]
            )[0]
        }
        #endif

        let projected = relative.matmul(projection).transposed(0, 2, 1, 3)
        let queryPositions = MLXArray(
            Int32(queryOffset)..<Int32(queryOffset + queryLength)
        ).reshaped(queryLength, 1)
        let keyPositions = MLXArray(0..<Int32(keyLength)).reshaped(1, keyLength)
        let distance = queryPositions - keyPositions
        let gatherIndices = MLX.broadcast(
            MLX.clip(distance, min: 0, max: Float(relativeExtent - 1))
                .asType(.int32)
                .reshaped(1, 1, queryLength, keyLength),
            to: [batch, heads, queryLength, keyLength]
        )
        var mask = takeAlong(projected, gatherIndices, axis: -1)
        mask = MLX.where(
            (distance .>= relativeExtent).reshaped(1, 1, queryLength, keyLength),
            MLXArray(0).asType(relative.dtype),
            mask
        )
        var blocked = distance .< 0
        if slidingWindow > 0 {
            blocked = blocked .|| (distance .>= slidingWindow)
        }
        return MLX.where(
            blocked.reshaped(1, 1, queryLength, keyLength),
            MLXArray(-1e30).asType(relative.dtype),
            mask
        )
    }
}

final class InklingShortConvolution: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d

    private let kernelSize: Int
    private let cacheIndex: Int

    init(channels: Int, kernelSize: Int, cacheIndex: Int) {
        self.kernelSize = kernelSize
        self.cacheIndex = cacheIndex
        self._conv.wrappedValue = Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: kernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: channels,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: InklingConvCache?) -> MLXArray {
        let originalType = x.dtype
        let floatInput = x.asType(.float32)
        let keep = max(0, kernelSize - 1)
        let paddedInput: MLXArray
        if let cache {
            let state = cache.states[cacheIndex]
                ?? MLXArray.zeros([x.dim(0), keep, x.dim(2)], dtype: .float32)
            paddedInput = concatenated([state, floatInput], axis: 1)
            cache.states[cacheIndex] = keep > 0
                ? paddedInput[0..., (paddedInput.dim(1) - keep)..., 0...]
                : MLXArray.zeros([x.dim(0), 0, x.dim(2)], dtype: .float32)
        } else {
            paddedInput = padded(
                floatInput,
                widths: [[0, 0], [keep, 0], [0, 0]],
                value: MLXArray(0).asType(.float32)
            )
        }
        return (conv(paddedInput.asType(conv.weight.dtype)).asType(.float32) + floatInput)
            .asType(originalType)
    }
}

final class InklingAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "r_proj") var rProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "k_sconv") var keyConvolution: InklingShortConvolution
    @ModuleInfo(key: "v_sconv") var valueConvolution: InklingShortConvolution
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ParameterInfo(key: "rel_proj") var relativeProjection: MLXArray

    private let headDimension: Int
    private let headCount: Int
    private let keyValueHeadCount: Int
    private let relativeDimensions: Int
    private let relativeExtent: Int
    private let slidingWindow: Int
    private let logScalingFloor: Int?
    private let logScalingAlpha: Float
    private let scale: Float

    init(config: InklingTextConfig, layerIndex: Int) {
        let sliding = config.layerIsSliding(layerIndex)
        headDimension = sliding ? config.swaHeadDim : config.headDim
        headCount = sliding ? config.swaNumAttentionHeads : config.numAttentionHeads
        keyValueHeadCount = sliding ? config.swaNumKeyValueHeads : config.numKeyValueHeads
        relativeDimensions = config.dRel
        relativeExtent = sliding ? config.slidingWindowSize : config.relExtent
        slidingWindow = sliding ? config.slidingWindowSize : 0
        logScalingFloor = sliding ? nil : config.logScalingNFloor
        logScalingAlpha = config.logScalingAlpha
        scale = 1 / Float(max(1, headDimension))

        self._qProj.wrappedValue = Linear(config.hiddenSize, headCount * headDimension, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, keyValueHeadCount * headDimension, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, keyValueHeadCount * headDimension, bias: false)
        self._rProj.wrappedValue = Linear(config.hiddenSize, headCount * config.dRel, bias: false)
        self._oProj.wrappedValue = Linear(headCount * headDimension, config.hiddenSize, bias: false)
        self._keyConvolution.wrappedValue = InklingShortConvolution(
            channels: keyValueHeadCount * headDimension,
            kernelSize: config.sconvKernelSize,
            cacheIndex: 0
        )
        self._valueConvolution.wrappedValue = InklingShortConvolution(
            channels: keyValueHeadCount * headDimension,
            kernelSize: config.sconvKernelSize,
            cacheIndex: 1
        )
        self._queryNorm.wrappedValue = RMSNorm(dimensions: headDimension, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = RMSNorm(dimensions: headDimension, eps: config.rmsNormEps)
        self._relativeProjection.wrappedValue = MLXArray.zeros([config.dRel, relativeExtent])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: InklingLayerCache?) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let convolutionCache = cache?.convolution
        var queries = qProj(x).reshaped(batch, length, headCount, headDimension)
        var keys = keyConvolution(kProj(x), cache: convolutionCache)
            .reshaped(batch, length, keyValueHeadCount, headDimension)
        var values = valueConvolution(vProj(x), cache: convolutionCache)
            .reshaped(batch, length, keyValueHeadCount, headDimension)
        let relative = rProj(x).reshaped(batch, length, headCount, relativeDimensions)

        queries = queryNorm(queries).transposed(0, 2, 1, 3)
        keys = keyNorm(keys).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let offset = cache?.attention.offset ?? 0
        if let cache {
            (keys, values) = cache.attention.update(keys: keys, values: values)
        }
        var mask = InklingBandedMask.make(
            relative: relative,
            projection: relativeProjection.asType(x.dtype),
            queryOffset: offset,
            keyLength: keys.dim(2),
            slidingWindow: slidingWindow,
            relativeExtent: relativeExtent
        )

        if let logScalingFloor {
            let positions = MLXArray(
                Int32(offset + 1)..<Int32(offset + length + 1)
            ).asType(.float32)
            let ratio = positions / Float(logScalingFloor)
            let tau = (1 + logScalingAlpha * MLX.log(MLX.maximum(ratio, MLXArray(1))))
                .reshaped(1, 1, length, 1)
                .asType(x.dtype)
            queries = queries * tau
            mask = MLX.where(mask .> MLXArray(-1e29).asType(mask.dtype), mask * tau, mask)
        }

        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .array(mask)
        )
        return oProj(
            attended.transposed(0, 2, 1, 3)
                .reshaped(batch, length, headCount * headDimension)
        )
    }
}

class InklingFeedForward: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("InklingFeedForward is abstract")
    }
}

final class InklingDenseMLP: InklingFeedForward {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ParameterInfo(key: "global_scale") var globalScale: MLXArray

    init(config: InklingTextConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.denseIntermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.denseIntermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.denseIntermediateSize, config.hiddenSize, bias: false)
        self._globalScale.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(inklingSwiGLU(gateProj(x), upProj(x))) * globalScale
    }
}

final class InklingSwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Q35SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: Q35SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: Q35SwitchLinear

    init(config: InklingTextConfig, expertCount: Int, quantization: InklingQuantizationConfig?) {
        let groupSize = quantization?.groupSize ?? InklingResources.quantizationGroupSize
        let bits = quantization?.bits ?? InklingResources.quantizationBits
        let quantized = quantization != nil
        self._gateProj.wrappedValue = Q35SwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: config.moeIntermediateSize,
            numExperts: expertCount,
            groupSize: groupSize,
            bits: bits,
            quantized: quantized,
            bias: false
        )
        self._upProj.wrappedValue = Q35SwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: config.moeIntermediateSize,
            numExperts: expertCount,
            groupSize: groupSize,
            bits: bits,
            quantized: quantized,
            bias: false
        )
        self._downProj.wrappedValue = Q35SwitchLinear(
            inputDims: config.moeIntermediateSize,
            outputDims: config.hiddenSize,
            numExperts: expertCount,
            groupSize: groupSize,
            bits: bits,
            quantized: quantized,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        downProj(inklingSwiGLU(gateProj(x, indices: indices), upProj(x, indices: indices)), indices: indices)
    }
}

final class InklingSparseMoE: InklingFeedForward {
    @ParameterInfo(key: "gate_weight") var gateWeight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray
    @ParameterInfo(key: "global_scale") var globalScale: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: InklingSwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: InklingSwitchGLU

    private let routedExpertCount: Int
    private let sharedExpertCount: Int
    private let topK: Int
    private let routeScale: Float

    init(config: InklingTextConfig, quantization: InklingQuantizationConfig?) {
        routedExpertCount = config.routedExpertCount
        sharedExpertCount = config.sharedExpertCount
        topK = config.expertsPerToken
        routeScale = config.routeScale
        self._gateWeight.wrappedValue = MLXArray.zeros([
            config.routedExpertCount + config.sharedExpertCount,
            config.hiddenSize,
        ])
        self._correctionBias.wrappedValue = MLXArray.zeros([config.routedExpertCount])
        self._globalScale.wrappedValue = MLXArray.ones([1])
        self._switchMLP.wrappedValue = InklingSwitchGLU(
            config: config,
            expertCount: config.routedExpertCount,
            quantization: quantization
        )
        self._sharedExperts.wrappedValue = InklingSwitchGLU(
            config: config,
            expertCount: config.sharedExpertCount,
            quantization: nil
        )
        super.init()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let dimensions = x.dim(2)
        let flat = x.reshaped(1, batch * length, dimensions)
        let logits = flat.matmul(gateWeight.asType(x.dtype).T)
        let scores = MLX.sigmoid(logits.asType(.float32))
        let routedScores = scores[.ellipsis, ..<routedExpertCount]
            + correctionBias.asType(.float32)
        let indices = argPartition(-routedScores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]

        let routedLogits = logits[.ellipsis, ..<routedExpertCount]
        let sharedLogits = logits[.ellipsis, routedExpertCount...]
        let selected = concatenated(
            [takeAlong(routedLogits, indices, axis: -1), sharedLogits],
            axis: -1
        ).asType(.float32)
        let logProbabilities = -MLX.logAddExp(MLXArray.zeros(like: selected), -selected)
        let weights = MLX.exp(
            logProbabilities - logSumExp(logProbabilities, axis: -1, keepDims: true)
        ) * routeScale * globalScale.asType(.float32)

        let routedWeights = weights[.ellipsis, ..<topK].asType(x.dtype)
        let sharedWeights = weights[.ellipsis, topK...].asType(x.dtype)
        let routed = (
            switchMLP(flat, indices: indices)
                * routedWeights.expandedDimensions(axis: -1)
        ).sum(axis: -2)
        let sharedIndices = MLX.broadcast(
            MLXArray(0..<Int32(sharedExpertCount)).reshaped(1, 1, sharedExpertCount),
            to: [1, batch * length, sharedExpertCount]
        )
        let shared = (
            sharedExperts(flat, indices: sharedIndices)
                * sharedWeights.expandedDimensions(axis: -1)
        ).sum(axis: -2)
        return (routed + shared).reshaped(batch, length, dimensions).asType(x.dtype)
    }
}

final class InklingDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: InklingAttention
    @ModuleInfo(key: "mlp") var mlp: InklingFeedForward
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "attn_sconv") var attentionConvolution: InklingShortConvolution
    @ModuleInfo(key: "mlp_sconv") var mlpConvolution: InklingShortConvolution

    init(config: InklingConfig, layerIndex: Int) {
        let text = config.textConfig
        self._attention.wrappedValue = InklingAttention(config: text, layerIndex: layerIndex)
        self._mlp.wrappedValue = text.layerIsDense(layerIndex)
            ? InklingDenseMLP(config: text)
            : InklingSparseMoE(config: text, quantization: config.quantization)
        self._inputNorm.wrappedValue = RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        self._postAttentionNorm.wrappedValue = RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        self._attentionConvolution.wrappedValue = InklingShortConvolution(
            channels: text.hiddenSize,
            kernelSize: text.sconvKernelSize,
            cacheIndex: 2
        )
        self._mlpConvolution.wrappedValue = InklingShortConvolution(
            channels: text.hiddenSize,
            kernelSize: text.sconvKernelSize,
            cacheIndex: 3
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: InklingLayerCache?) -> MLXArray {
        let attentionOutput = attention(inputNorm(x), cache: cache)
        let hidden = x + attentionConvolution(attentionOutput, cache: cache?.convolution)
        return hidden + mlpConvolution(mlp(postAttentionNorm(hidden)), cache: cache?.convolution)
    }
}

final class InklingTransformer: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: RMSNorm?
    @ModuleInfo(key: "layers") var layers: [InklingDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: InklingConfig) {
        let text = config.textConfig
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: text.vocabSize,
            dimensions: text.hiddenSize
        )
        self._embedNorm.wrappedValue = text.useEmbedNorm
            ? RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
            : nil
        self._layers.wrappedValue = (0..<text.numHiddenLayers).map {
            InklingDecoderLayer(config: config, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputIDs: MLXArray, cache: [InklingLayerCache]?) -> MLXArray {
        var ids = inputIDs
        if ids.dtype != .int32 {
            ids = ids.asType(.int32)
        }
        var hidden = embedTokens(ids)
        if let embedNorm {
            hidden = embedNorm(hidden)
        }
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: cache?[index])
        }
        return norm(hidden)
    }
}

struct InklingForwardOutput {
    let hidden: MLXArray
    let logits: MLXArray
}

public final class InklingLanguageModel: Module, @unchecked Sendable {
    @ModuleInfo(key: "model") var model: InklingTransformer
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: InklingConfig

    public init(config: InklingConfig) {
        self.config = config
        self._model.wrappedValue = InklingTransformer(config: config)
        self._lmHead.wrappedValue = Linear(
            config.textConfig.hiddenSize,
            config.textConfig.vocabSize,
            bias: false
        )
        super.init()
    }

    func forward(_ inputIDs: MLXArray, cache: [InklingLayerCache]?) -> InklingForwardOutput {
        let hidden = model(inputIDs, cache: cache)
        return InklingForwardOutput(hidden: hidden, logits: logits(from: hidden))
    }

    func forwardPrefill(_ inputIDs: MLXArray, cache: [InklingLayerCache]?) -> InklingForwardOutput {
        var hidden = model(inputIDs, cache: cache)
        if hidden.dim(1) > 1 {
            hidden = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        }
        return InklingForwardOutput(hidden: hidden, logits: logits(from: hidden))
    }

    public func callAsFunction(_ inputIDs: MLXArray, cache: [InklingLayerCache]?) -> MLXArray {
        forward(inputIDs, cache: cache).logits
    }

    private func logits(from hidden: MLXArray) -> MLXArray {
        var logits = lmHead(hidden / config.textConfig.logitsMUPWidthMultiplier)
        if let unpadded = config.textConfig.unpaddedVocabSize,
           unpadded < logits.dim(-1) {
            logits = logits[.ellipsis, ..<unpadded]
        }
        return logits
    }
}
