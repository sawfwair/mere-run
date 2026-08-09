import Foundation
import MLX
import MLXNN

final class Ideogram4EmbedScalar: Module {
    let dim: Int
    let rangeMin: Float
    let rangeMax: Float

    @ModuleInfo(key: "mlp_in") var mlpIn: Linear
    @ModuleInfo(key: "mlp_out") var mlpOut: Linear

    init(dim: Int, inputRange: (Float, Float) = (0, 1)) {
        self.dim = dim
        self.rangeMin = inputRange.0
        self.rangeMax = inputRange.1
        self._mlpIn.wrappedValue = Linear(dim, dim, bias: true)
        self._mlpOut.wrappedValue = Linear(dim, dim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scaled = (x.asType(.float32) - MLXArray(rangeMin)) / MLXArray(rangeMax - rangeMin)
        let embedding = sinusoidalEmbedding(scaled * MLXArray(10_000.0), dim: dim)
        return mlpOut(MLXNN.silu(mlpIn(embedding.asType(.bfloat16))))
    }

    private func sinusoidalEmbedding(_ x: MLXArray, dim: Int) -> MLXArray {
        let half = dim / 2
        let denominator = Float(max(half - 1, 1))
        let exponent = MLXArray(0..<half).asType(.float32)
            * MLXArray(Float(-Foundation.log(10_000.0)) / denominator)
        let frequencies = MLX.exp(exponent)
        let args = x[.ellipsis, .newAxis] * frequencies
        let sinPart = MLX.sin(args)
        let cosPart = MLX.cos(args)
        var embedding = MLX.concatenated([sinPart, cosPart], axis: -1)
        if dim % 2 == 1 {
            var padShape = embedding.shape
            padShape[padShape.count - 1] = 1
            embedding = MLX.concatenated([embedding, MLX.zeros(padShape, dtype: embedding.dtype)], axis: -1)
        }
        return embedding
    }
}

final class Ideogram4Attention: Module {
    let hiddenSize: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float
    let fusedKernelsEnabled: Bool

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "o") var output: Linear

    init(hiddenSize: Int, numHeads: Int, headDim: Int, eps: Float, fusedKernelsEnabled: Bool) {
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.headDim = headDim
        self.scale = 1.0 / sqrt(Float(headDim))
        self.fusedKernelsEnabled = fusedKernelsEnabled
        self._qkv.wrappedValue = Linear(hiddenSize, hiddenSize * 3, bias: false)
        self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._output.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        segmentIds: MLXArray?,
        cos: MLXArray,
        sin: MLXArray,
        dynamicSparseRuntime: DynamicSparseAttentionRuntime?,
        layerIndex: Int,
        prefixTokenCount: Int
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let projected = qkv(x)
        var queries: MLXArray
        var keys: MLXArray
        let values: MLXArray
        if fusedKernelsEnabled,
           let fused = Ideogram4FusedKernels.qkvNorms(
               qkv: projected,
               qWeight: normQ.weight,
               kWeight: normK.weight,
               eps: normQ.eps,
               numHeads: numHeads,
               headDim: headDim
           ) {
            queries = fused.queries
            keys = fused.keys
            values = fused.values
        } else {
            let qkvParts = MLX.split(projected, parts: 3, axis: -1)
            queries = normQ(qkvParts[0].reshaped(batch, sequenceLength, numHeads, headDim))
                .transposed(0, 2, 1, 3)
            keys = normK(qkvParts[1].reshaped(batch, sequenceLength, numHeads, headDim))
                .transposed(0, 2, 1, 3)
            values = qkvParts[2].reshaped(batch, sequenceLength, numHeads, headDim)
                .transposed(0, 2, 1, 3)
        }

        (queries, keys) = applyRotaryPosEmb(queries, keys, cos: cos, sin: sin)

        let sparse = segmentIds == nil
            ? dynamicSparseRuntime?.call(
                queries: queries,
                keys: keys,
                values: values,
                layerIndex: layerIndex,
                prefixTokenCount: prefixTokenCount,
                scale: scale
            )
            : nil
        let attended = (sparse ?? denseAttention(
            queries: queries,
            keys: keys,
            values: values,
            segmentIds: segmentIds
        )).asType(x.dtype)
            .transposed(0, 2, 1, 3)
            .reshaped(batch, sequenceLength, hiddenSize)
        return output(attended)
    }

    private func denseAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        segmentIds: MLXArray?
    ) -> MLXArray {
        let scores = MLX.matmul(
            queries.asType(.float32),
            keys.asType(.float32).transposed(0, 1, 3, 2)
        ) * MLXArray(scale)
        let attention: MLXArray
        if let segmentIds {
            let mask = blockDiagonalAttentionMask(segmentIds: segmentIds)
            let maskedScores = MLX.where(
                mask,
                scores,
                MLX.zeros(scores.shape, dtype: scores.dtype) + MLXArray(-Float.greatestFiniteMagnitude)
            )
            attention = softmax(maskedScores, axis: -1)
        } else {
            attention = softmax(scores, axis: -1)
        }
        return MLX.matmul(attention, values.asType(.float32))
    }

    private func blockDiagonalAttentionMask(segmentIds: MLXArray) -> MLXArray {
        let batch = segmentIds.dim(0)
        let sequenceLength = segmentIds.dim(1)
        let rows = segmentIds.reshaped(batch, sequenceLength, 1)
        let columns = segmentIds.reshaped(batch, 1, sequenceLength)
        let sameSegment = rows .== columns
        let validRows = rows .>= MLXArray(Int32(0))
        let validColumns = columns .>= MLXArray(Int32(0))
        return (sameSegment .&& validRows .&& validColumns).reshaped(batch, 1, sequenceLength, sequenceLength)
    }
}

final class Ideogram4MLP: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        self._w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(MLXNN.silu(w1(x)) * w3(x))
    }
}

final class Ideogram4TransformerBlock: Module {
    let hiddenSize: Int
    let fusedKernelsEnabled: Bool

    @ModuleInfo(key: "attention") var attention: Ideogram4Attention
    @ModuleInfo(key: "feed_forward") var feedForward: Ideogram4MLP
    @ModuleInfo(key: "attention_norm1") var attentionNorm1: RMSNorm
    @ModuleInfo(key: "ffn_norm1") var ffnNorm1: RMSNorm
    @ModuleInfo(key: "attention_norm2") var attentionNorm2: RMSNorm
    @ModuleInfo(key: "ffn_norm2") var ffnNorm2: RMSNorm
    @ModuleInfo(key: "adaln_modulation") var adalnModulation: Linear

    init(configuration: Ideogram4TransformerConfiguration, fusedKernelsEnabled: Bool) {
        self.hiddenSize = configuration.embeddingDim
        self.fusedKernelsEnabled = fusedKernelsEnabled
        self._attention.wrappedValue = Ideogram4Attention(
            hiddenSize: configuration.embeddingDim,
            numHeads: configuration.numAttentionHeads,
            headDim: configuration.attentionHeadDim,
            eps: 1e-5,
            fusedKernelsEnabled: fusedKernelsEnabled
        )
        self._feedForward.wrappedValue = Ideogram4MLP(
            dim: configuration.embeddingDim,
            hiddenDim: configuration.intermediateSize
        )
        self._attentionNorm1.wrappedValue = RMSNorm(dimensions: configuration.embeddingDim, eps: configuration.normEps)
        self._ffnNorm1.wrappedValue = RMSNorm(dimensions: configuration.embeddingDim, eps: configuration.normEps)
        self._attentionNorm2.wrappedValue = RMSNorm(dimensions: configuration.embeddingDim, eps: configuration.normEps)
        self._ffnNorm2.wrappedValue = RMSNorm(dimensions: configuration.embeddingDim, eps: configuration.normEps)
        self._adalnModulation.wrappedValue = Linear(configuration.adalnDim, configuration.embeddingDim * 4, bias: true)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        segmentIds: MLXArray?,
        cos: MLXArray,
        sin: MLXArray,
        adalnInput: MLXArray,
        dynamicSparseRuntime: DynamicSparseAttentionRuntime?,
        layerIndex: Int,
        prefixTokenCount: Int
    ) -> MLXArray {
        let modulation = adalnModulation(adalnInput)
        let modulationParts = MLX.split(modulation, parts: 4, axis: -1)
        let scaleMSA = 1 + modulationParts[0]
        let gateMSA = MLX.tanh(modulationParts[1])
        let scaleMLP = 1 + modulationParts[2]
        let gateMLP = MLX.tanh(modulationParts[3])

        let attentionInput = fusedKernelsEnabled
            ? Ideogram4FusedKernels.scaledRMSNorm(
                x,
                weight: attentionNorm1.weight,
                modulation: modulation,
                modulationIndex: 0,
                eps: attentionNorm1.eps
            ) ?? (attentionNorm1(x) * scaleMSA)
            : attentionNorm1(x) * scaleMSA
        let attnOut = attention(
            attentionInput,
            segmentIds: segmentIds,
            cos: cos,
            sin: sin,
            dynamicSparseRuntime: dynamicSparseRuntime,
            layerIndex: layerIndex,
            prefixTokenCount: prefixTokenCount
        )
        var out = fusedKernelsEnabled
            ? Ideogram4FusedKernels.gatedResidualRMSNorm(
                attnOut,
                residual: x,
                weight: attentionNorm2.weight,
                modulation: modulation,
                modulationIndex: 1,
                eps: attentionNorm2.eps
            ) ?? (x + gateMSA * attentionNorm2(attnOut))
            : x + gateMSA * attentionNorm2(attnOut)
        let mlpInput = fusedKernelsEnabled
            ? Ideogram4FusedKernels.scaledRMSNorm(
                out,
                weight: ffnNorm1.weight,
                modulation: modulation,
                modulationIndex: 2,
                eps: ffnNorm1.eps
            ) ?? (ffnNorm1(out) * scaleMLP)
            : ffnNorm1(out) * scaleMLP
        let mlpOut = feedForward(mlpInput)
        out = fusedKernelsEnabled
            ? Ideogram4FusedKernels.gatedResidualRMSNorm(
                mlpOut,
                residual: out,
                weight: ffnNorm2.weight,
                modulation: modulation,
                modulationIndex: 3,
                eps: ffnNorm2.eps
            ) ?? (out + gateMLP * ffnNorm2(mlpOut))
            : out + gateMLP * ffnNorm2(mlpOut)
        return out
    }
}

final class Ideogram4FinalLayer: Module {
    @ModuleInfo(key: "norm_final") var normFinal: LayerNorm
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "adaln_modulation") var adalnModulation: Linear

    init(hiddenSize: Int, outChannels: Int, adalnDim: Int) {
        self._normFinal.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6, affine: false)
        self._linear.wrappedValue = Linear(hiddenSize, outChannels, bias: true)
        self._adalnModulation.wrappedValue = Linear(adalnDim, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
        let scale = 1 + adalnModulation(MLXNN.silu(conditioning))
        return linear(normFinal(x) * scale)
    }
}

public final class Ideogram4Transformer: Module {
    public let configuration: Ideogram4TransformerConfiguration
    let fusedKernelsEnabled: Bool
    private let dynamicSparseRuntime: DynamicSparseAttentionRuntime?

    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "llm_cond_norm") var llmCondNorm: RMSNorm
    @ModuleInfo(key: "llm_cond_proj") var llmCondProj: Linear
    @ModuleInfo(key: "t_embedding") var timeEmbedding: Ideogram4EmbedScalar
    @ModuleInfo(key: "adaln_proj") var adalnProj: Linear
    @ModuleInfo(key: "embed_image_indicator") var imageIndicatorEmbedding: Embedding
    @ModuleInfo(key: "layers") var layers: [Ideogram4TransformerBlock]
    @ModuleInfo(key: "final_layer") var finalLayer: Ideogram4FinalLayer

    let rotaryEmbedding: Qwen3VLRotaryEmbedding

    public convenience init(configuration: Ideogram4TransformerConfiguration) {
        self.init(
            configuration: configuration,
            fusedKernelsEnabled: Ideogram4FusedKernelPolicy.enabled
        )
    }

    init(configuration: Ideogram4TransformerConfiguration, fusedKernelsEnabled: Bool) {
        self.configuration = configuration
        self.fusedKernelsEnabled = fusedKernelsEnabled
        self.dynamicSparseRuntime = DynamicSparseAttentionRuntime.configured(model: .ideogram4)
        self._inputProj.wrappedValue = Linear(configuration.inChannels, configuration.embeddingDim, bias: true)
        self._llmCondNorm.wrappedValue = RMSNorm(dimensions: configuration.llmFeaturesDim, eps: 1e-6)
        self._llmCondProj.wrappedValue = Linear(configuration.llmFeaturesDim, configuration.embeddingDim, bias: true)
        self._timeEmbedding.wrappedValue = Ideogram4EmbedScalar(dim: configuration.embeddingDim)
        self._adalnProj.wrappedValue = Linear(configuration.embeddingDim, configuration.adalnDim, bias: true)
        self._imageIndicatorEmbedding.wrappedValue = Embedding(embeddingCount: 2, dimensions: configuration.embeddingDim)
        self._layers.wrappedValue = (0..<configuration.numLayers).map { _ in
            Ideogram4TransformerBlock(
                configuration: configuration,
                fusedKernelsEnabled: fusedKernelsEnabled
            )
        }
        self._finalLayer.wrappedValue = Ideogram4FinalLayer(
            hiddenSize: configuration.embeddingDim,
            outChannels: configuration.inChannels,
            adalnDim: configuration.adalnDim
        )
        self.rotaryEmbedding = Qwen3VLRotaryEmbedding(
            dim: configuration.attentionHeadDim,
            base: configuration.ropeTheta,
            mropeSection: configuration.mropeSection
        )
        super.init()
    }

    func beginDenoisingStep(index: Int, count: Int) {
        dynamicSparseRuntime?.beginStep(index: index, count: count)
    }

    public func callAsFunction(sample: Ideogram4PackedSample, timestep: MLXArray) -> MLXArray {
        callAsFunction(
            llmFeatures: sample.llmFeatures,
            x: sample.x,
            timestep: timestep,
            positionIds: sample.positionIds,
            segmentIds: sample.segmentIds,
            indicator: sample.indicator,
            segmentsAreUniform: true,
            dynamicSparsePrefixTokenCount: sample.textTokenCount
        )
    }

    public func callAsFunction(
        llmFeatures: MLXArray,
        x: MLXArray,
        timestep: MLXArray,
        positionIds: MLXArray,
        segmentIds: MLXArray,
        indicator: MLXArray
    ) -> MLXArray {
        callAsFunction(
            llmFeatures: llmFeatures,
            x: x,
            timestep: timestep,
            positionIds: positionIds,
            segmentIds: segmentIds,
            indicator: indicator,
            segmentsAreUniform: false,
            dynamicSparsePrefixTokenCount: 0
        )
    }

    func callAsFunction(
        llmFeatures: MLXArray,
        x: MLXArray,
        timestep: MLXArray,
        positionIds: MLXArray,
        segmentIds: MLXArray,
        indicator: MLXArray,
        segmentsAreUniform: Bool,
        dynamicSparsePrefixTokenCount: Int = 0
    ) -> MLXArray {
        let outputMask = roleMask(indicator, role: Ideogram4SampleBuilder.outputImageIndicator, dtype: x.dtype)
        let llmMask = roleMask(indicator, role: Ideogram4SampleBuilder.llmTokenIndicator, dtype: x.dtype)

        let projectedImage = inputProj(x * outputMask) * outputMask
        let normalizedFeatures = llmCondNorm(llmFeatures * llmMask)
        let projectedFeatures = llmCondProj(normalizedFeatures) * llmMask

        var hidden = projectedImage + projectedFeatures
        let imageIndicatorIds = (indicator .== MLXArray(Int32(Ideogram4SampleBuilder.outputImageIndicator))).asType(.int32)
        hidden = hidden + imageIndicatorEmbedding(imageIndicatorIds)

        var timeCondition = timeEmbedding(timestep)
        if timestep.ndim == 1 {
            timeCondition = timeCondition.expandedDimensions(axis: 1)
        }
        let adalnInput = MLXNN.silu(adalnProj(timeCondition))

        let normalizedPositionIds = normalizePositionIds(positionIds)
        let (cos, sin) = rotaryEmbedding(positionIds: normalizedPositionIds, dtype: hidden.dtype)
        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                segmentIds: segmentsAreUniform ? nil : segmentIds,
                cos: cos,
                sin: sin,
                adalnInput: adalnInput,
                dynamicSparseRuntime: dynamicSparseRuntime,
                layerIndex: index,
                prefixTokenCount: dynamicSparsePrefixTokenCount
            )
        }
        return finalLayer(hidden, conditioning: adalnInput).asType(.float32)
    }

    private func roleMask(_ indicator: MLXArray, role: Int, dtype: DType) -> MLXArray {
        (indicator .== MLXArray(Int32(role))).asType(dtype).expandedDimensions(axis: -1)
    }

    private func normalizePositionIds(_ positionIds: MLXArray) -> MLXArray {
        if positionIds.ndim == 3, positionIds.dim(0) == 3 {
            return positionIds
        }
        if positionIds.ndim == 3, positionIds.dim(2) == 3 {
            return positionIds.transposed(2, 0, 1)
        }
        return positionIds
    }
}
