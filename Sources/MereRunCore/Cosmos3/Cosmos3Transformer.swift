import Foundation
import MLX
import MLXFast
import MLXNN

public struct Cosmos3RotaryEmbedding {
    public let headDimension: Int
    public let theta: Float
    public let axesDimensions: [Int]
    private let inverseFrequencies: [Float]

    public init(headDimension: Int, theta: Float, axesDimensions: [Int]) {
        precondition(headDimension.isMultiple(of: 2))
        precondition(axesDimensions.count == 3)
        precondition(axesDimensions.reduce(0, +) == headDimension / 2)
        self.headDimension = headDimension
        self.theta = theta
        self.axesDimensions = axesDimensions
        self.inverseFrequencies = stride(from: 0, to: headDimension, by: 2).map {
            1 / pow(theta, Float($0) / Float(headDimension))
        }
    }

    public func callAsFunction(positionIDs: MLXArray, dtype: DType = .float32)
        -> (cosine: MLXArray, sine: MLXArray) {
        precondition(positionIDs.ndim == 2 && positionIDs.dim(0) == 3)
        let positions = positionIDs.asType(.float32)
        let interleavedEnd = (axesDimensions[1] + axesDimensions[2]) * 3 / 2
        let frequencyColumns = inverseFrequencies.enumerated().map { index, frequency in
            let axis: Int
            if index < interleavedEnd {
                axis = index % 3
            } else {
                axis = 0
            }
            return positions[axis] * frequency
        }
        let frequencies = MLX.stacked(frequencyColumns, axis: 1)
        let embedding = MLX.concatenated([frequencies, frequencies], axis: 1)
        return (MLX.cos(embedding).asType(dtype), MLX.sin(embedding).asType(dtype))
    }

    public static func apply(
        _ input: MLXArray,
        cosine: MLXArray,
        sine: MLXArray
    ) -> MLXArray {
        precondition(input.ndim == 3)
        let half = input.dim(-1) / 2
        let rotatedHalf = MLX.concatenated(
            [-input[0..., 0..., half...], input[0..., 0..., 0..<half]],
            axis: -1
        )
        let cos = cosine.expandedDimensions(axis: 1)
        let sin = sine.expandedDimensions(axis: 1)
        return input * cos + rotatedHalf * sin
    }
}

public final class Cosmos3DomainAwareLinear: Module {
    public let inputSize: Int
    public let outputSize: Int
    public let domainCount: Int

    @ModuleInfo(key: "fc") var weights: Embedding
    @ModuleInfo(key: "bias") var biases: Embedding

    public init(inputSize: Int, outputSize: Int, domainCount: Int) {
        precondition(inputSize > 0 && outputSize > 0 && domainCount > 0)
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.domainCount = domainCount
        self._weights.wrappedValue = Embedding(
            embeddingCount: domainCount,
            dimensions: outputSize * inputSize
        )
        self._biases.wrappedValue = Embedding(embeddingCount: domainCount, dimensions: outputSize)
    }

    public func callAsFunction(_ input: MLXArray, domainIDs: MLXArray) -> MLXArray {
        precondition(input.ndim == 2 || input.ndim == 3)
        precondition(input.dim(0) == domainIDs.size)
        let ids = domainIDs.reshaped(-1).asType(.int32)
        let domainWeights = weights(ids)
            // NVIDIA stores each embedding row flattened from
            // [input_size, output_size], then views it directly before bmm.
            .reshaped(ids.size, inputSize, outputSize)
        let domainBiases = biases(ids).reshaped(ids.size, outputSize)
        if input.ndim == 2 {
            return MLX.matmul(input.expandedDimensions(axis: 1), domainWeights)
                .squeezed(axis: 1) + domainBiases
        }
        return MLX.matmul(input, domainWeights) + domainBiases.expandedDimensions(axis: 1)
    }
}

final class Cosmos3MLP: Module {
    let activation: Cosmos3FeedForwardActivation
    @ModuleInfo(key: "gate_proj") var gateProjection: Linear?
    @ModuleInfo(key: "up_proj") var upProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear

    init(
        hiddenSize: Int,
        intermediateSize: Int,
        activation: Cosmos3FeedForwardActivation
    ) {
        self.activation = activation
        self._gateProjection.wrappedValue = activation == .siluGated
            ? Linear(hiddenSize, intermediateSize, bias: false)
            : nil
        self._upProjection.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProjection.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        switch activation {
        case .reluSquared:
            let activated = MLX.maximum(upProjection(input), MLXArray(0))
            return downProjection(activated * activated)
        case .siluGated:
            return downProjection(MLXNN.silu(gateProjection!(input)) * upProjection(input))
        }
    }
}

final class Cosmos3PackedAttention: Module {
    let attentionHeads: Int
    let keyValueHeads: Int
    let headDimension: Int
    let scale: Float

    @ModuleInfo(key: "to_q") var understandingQuery: Linear
    @ModuleInfo(key: "to_k") var understandingKey: Linear
    @ModuleInfo(key: "to_v") var understandingValue: Linear
    @ModuleInfo(key: "to_out") var understandingOutput: Linear
    @ModuleInfo(key: "norm_q") var understandingQueryNorm: RMSNorm?
    @ModuleInfo(key: "norm_k") var understandingKeyNorm: RMSNorm?
    @ModuleInfo(key: "add_q_proj") var generationQuery: Linear
    @ModuleInfo(key: "add_k_proj") var generationKey: Linear
    @ModuleInfo(key: "add_v_proj") var generationValue: Linear
    @ModuleInfo(key: "to_add_out") var generationOutput: Linear
    @ModuleInfo(key: "norm_added_q") var generationQueryNorm: RMSNorm
    @ModuleInfo(key: "norm_added_k") var generationKeyNorm: RMSNorm
    @ModuleInfo(key: "k_norm_und_for_gen") var understandingKeyNormForGeneration: RMSNorm?

    init(configuration: Cosmos3TransformerConfiguration) {
        self.attentionHeads = configuration.attentionHeadCount
        self.keyValueHeads = configuration.keyValueHeadCount
        self.headDimension = configuration.headDimension
        self.scale = 1 / sqrt(Float(configuration.headDimension))
        let hidden = configuration.hiddenSize
        let qSize = configuration.attentionHeadCount * configuration.headDimension
        let kvSize = configuration.keyValueHeadCount * configuration.headDimension
        let bias = configuration.attentionBias
        self._understandingQuery.wrappedValue = Linear(hidden, qSize, bias: bias)
        self._understandingKey.wrappedValue = Linear(hidden, kvSize, bias: bias)
        self._understandingValue.wrappedValue = Linear(hidden, kvSize, bias: bias)
        self._understandingOutput.wrappedValue = Linear(qSize, hidden, bias: bias)
        self._understandingQueryNorm.wrappedValue = configuration.normalizesUnderstandingQueriesAndKeys
            ? RMSNorm(dimensions: configuration.headDimension, eps: configuration.rmsNormEpsilon)
            : nil
        self._understandingKeyNorm.wrappedValue = configuration.normalizesUnderstandingQueriesAndKeys
            ? RMSNorm(dimensions: configuration.headDimension, eps: configuration.rmsNormEpsilon)
            : nil
        self._generationQuery.wrappedValue = Linear(hidden, qSize, bias: bias)
        self._generationKey.wrappedValue = Linear(hidden, kvSize, bias: bias)
        self._generationValue.wrappedValue = Linear(hidden, kvSize, bias: bias)
        self._generationOutput.wrappedValue = Linear(qSize, hidden, bias: bias)
        self._generationQueryNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDimension,
            eps: configuration.rmsNormEpsilon
        )
        self._generationKeyNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDimension,
            eps: configuration.rmsNormEpsilon
        )
        self._understandingKeyNormForGeneration.wrappedValue =
            configuration.normalizesUnderstandingKeysForGeneration
                ? RMSNorm(
                    dimensions: configuration.headDimension,
                    eps: configuration.rmsNormEpsilon
                )
                : nil
    }

    func callAsFunction(
        understanding: MLXArray,
        generation: MLXArray,
        rotary: (
            understandingCosine: MLXArray,
            understandingSine: MLXArray,
            generationCosine: MLXArray,
            generationSine: MLXArray
        )
    ) -> (understanding: MLXArray, generation: MLXArray) {
        var qUnderstanding = understandingQuery(understanding)
            .reshaped(-1, attentionHeads, headDimension)
        var kUnderstanding = understandingKey(understanding)
            .reshaped(-1, keyValueHeads, headDimension)
        let vUnderstanding = understandingValue(understanding)
            .reshaped(-1, keyValueHeads, headDimension)
        var qGeneration = generationQuery(generation)
            .reshaped(-1, attentionHeads, headDimension)
        var kGeneration = generationKey(generation)
            .reshaped(-1, keyValueHeads, headDimension)
        let vGeneration = generationValue(generation)
            .reshaped(-1, keyValueHeads, headDimension)

        if let understandingQueryNorm, let understandingKeyNorm {
            qUnderstanding = understandingQueryNorm(qUnderstanding)
            kUnderstanding = understandingKeyNorm(kUnderstanding)
        }
        let understandingKeyForGeneration = understandingKeyNormForGeneration?(kUnderstanding)
            ?? kUnderstanding
        qUnderstanding = Cosmos3RotaryEmbedding.apply(
            qUnderstanding,
            cosine: rotary.understandingCosine,
            sine: rotary.understandingSine
        )
        kUnderstanding = Cosmos3RotaryEmbedding.apply(
            kUnderstanding,
            cosine: rotary.understandingCosine,
            sine: rotary.understandingSine
        )
        let normalizedUnderstandingKey = Cosmos3RotaryEmbedding.apply(
            understandingKeyForGeneration,
            cosine: rotary.understandingCosine,
            sine: rotary.understandingSine
        )
        qGeneration = Cosmos3RotaryEmbedding.apply(
            generationQueryNorm(qGeneration),
            cosine: rotary.generationCosine,
            sine: rotary.generationSine
        )
        kGeneration = Cosmos3RotaryEmbedding.apply(
            generationKeyNorm(kGeneration),
            cosine: rotary.generationCosine,
            sine: rotary.generationSine
        )

        let causal = attend(
            queries: qUnderstanding,
            keys: kUnderstanding,
            values: vUnderstanding,
            mask: .causal
        )
        let full = attend(
            queries: qGeneration,
            keys: MLX.concatenated([normalizedUnderstandingKey, kGeneration], axis: 0),
            values: MLX.concatenated([vUnderstanding, vGeneration], axis: 0),
            mask: .none
        )
        return (
            understandingOutput(causal),
            generationOutput(full)
        )
    }

    func understandingOnly(
        _ understanding: MLXArray,
        cosine: MLXArray,
        sine: MLXArray
    ) -> MLXArray {
        var queries = understandingQuery(understanding)
            .reshaped(-1, attentionHeads, headDimension)
        var keys = understandingKey(understanding)
            .reshaped(-1, keyValueHeads, headDimension)
        let values = understandingValue(understanding)
            .reshaped(-1, keyValueHeads, headDimension)
        if let understandingQueryNorm, let understandingKeyNorm {
            queries = understandingQueryNorm(queries)
            keys = understandingKeyNorm(keys)
        }
        queries = Cosmos3RotaryEmbedding.apply(
            queries,
            cosine: cosine,
            sine: sine
        )
        keys = Cosmos3RotaryEmbedding.apply(
            keys,
            cosine: cosine,
            sine: sine
        )
        return understandingOutput(attend(
            queries: queries,
            keys: keys,
            values: values,
            mask: .causal
        ))
    }

    private func attend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let queryLength = queries.dim(0)
        var transposedKeys = keys.transposed(1, 0, 2).expandedDimensions(axis: 0)
        var transposedValues = values.transposed(1, 0, 2).expandedDimensions(axis: 0)
        if keyValueHeads != attentionHeads {
            let repeats = attentionHeads / keyValueHeads
            let keyLength = keys.dim(0)
            transposedKeys = MLX.broadcast(
                transposedKeys.reshaped(1, keyValueHeads, 1, keyLength, headDimension),
                to: [1, keyValueHeads, repeats, keyLength, headDimension]
            ).reshaped(1, attentionHeads, keyLength, headDimension)
            transposedValues = MLX.broadcast(
                transposedValues.reshaped(1, keyValueHeads, 1, keyLength, headDimension),
                to: [1, keyValueHeads, repeats, keyLength, headDimension]
            ).reshaped(1, attentionHeads, keyLength, headDimension)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries.transposed(1, 0, 2).expandedDimensions(axis: 0),
            keys: transposedKeys,
            values: transposedValues,
            scale: scale,
            mask: mask
        )
        return attended.squeezed(axis: 0).transposed(1, 0, 2)
            .reshaped(queryLength, attentionHeads * headDimension)
    }
}

final class Cosmos3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Cosmos3PackedAttention
    @ModuleInfo(key: "mlp") var understandingMLP: Cosmos3MLP
    @ModuleInfo(key: "mlp_moe_gen") var generationMLP: Cosmos3MLP
    @ModuleInfo(key: "input_layernorm") var understandingInputNorm: RMSNorm
    @ModuleInfo(key: "input_layernorm_moe_gen") var generationInputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var understandingPostAttentionNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm_moe_gen") var generationPostAttentionNorm: RMSNorm

    init(configuration: Cosmos3TransformerConfiguration) {
        self._attention.wrappedValue = Cosmos3PackedAttention(configuration: configuration)
        self._understandingMLP.wrappedValue = Cosmos3MLP(
            hiddenSize: configuration.hiddenSize,
            intermediateSize: configuration.intermediateSize,
            activation: configuration.feedForwardActivation
        )
        self._generationMLP.wrappedValue = Cosmos3MLP(
            hiddenSize: configuration.hiddenSize,
            intermediateSize: configuration.intermediateSize,
            activation: configuration.feedForwardActivation
        )
        self._understandingInputNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon
        )
        self._generationInputNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon
        )
        self._understandingPostAttentionNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon
        )
        self._generationPostAttentionNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon
        )
    }

    func callAsFunction(
        understanding: MLXArray,
        generation: MLXArray,
        rotary: (
            understandingCosine: MLXArray,
            understandingSine: MLXArray,
            generationCosine: MLXArray,
            generationSine: MLXArray
        )
    ) -> (understanding: MLXArray, generation: MLXArray) {
        let attended = attention(
            understanding: understandingInputNorm(understanding),
            generation: generationInputNorm(generation),
            rotary: rotary
        )
        let understandingResidual = understanding + attended.understanding
        let generationResidual = generation + attended.generation
        return (
            understandingResidual
                + understandingMLP(understandingPostAttentionNorm(understandingResidual)),
            generationResidual + generationMLP(generationPostAttentionNorm(generationResidual))
        )
    }

    func understandingOnly(
        _ understanding: MLXArray,
        cosine: MLXArray,
        sine: MLXArray
    ) -> MLXArray {
        let residual = understanding + attention.understandingOnly(
            understandingInputNorm(understanding),
            cosine: cosine,
            sine: sine
        )
        return residual + understandingMLP(understandingPostAttentionNorm(residual))
    }
}

final class Cosmos3TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var inputProjection: Linear
    @ModuleInfo(key: "linear_2") var outputProjection: Linear

    init(hiddenSize: Int) {
        self._inputProjection.wrappedValue = Linear(256, hiddenSize, bias: true)
        self._outputProjection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ timesteps: MLXArray) -> MLXArray {
        precondition(timesteps.ndim == 1)
        let half = 128
        let indices = MLXArray((0..<half).map(Float.init))
        let frequencies = MLX.exp(-Float(log(Double(10_000))) * indices / Float(half))
        let arguments = timesteps.asType(.float32).expandedDimensions(axis: 1)
            * frequencies.expandedDimensions(axis: 0)
        let embedding = MLX.concatenated([MLX.cos(arguments), MLX.sin(arguments)], axis: 1)
        return outputProjection(MLXNN.silu(inputProjection(embedding)))
    }
}

public final class Cosmos3OmniTransformerModel: Module {
    public let configuration: Cosmos3TransformerConfiguration
    public let rotaryEmbedding: Cosmos3RotaryEmbedding

    @ModuleInfo(key: "embed_tokens") var tokenEmbedding: Embedding
    @ModuleInfo(key: "layers") var layers: [Cosmos3DecoderLayer]
    @ModuleInfo(key: "norm") var understandingNorm: RMSNorm
    @ModuleInfo(key: "norm_moe_gen") var generationNorm: RMSNorm
    @ModuleInfo(key: "lm_head") var languageModelHead: Linear?
    @ModuleInfo(key: "proj_in") var visionInputProjection: Linear
    @ModuleInfo(key: "proj_out") var visionOutputProjection: Linear
    @ModuleInfo(key: "time_embedder") var timestepEmbedding: Cosmos3TimestepEmbedding
    @ModuleInfo(key: "action_proj_in") var actionInputProjection: Cosmos3DomainAwareLinear?
    @ModuleInfo(key: "action_proj_out") var actionOutputProjection: Cosmos3DomainAwareLinear?
    @ModuleInfo(key: "action_modality_embed") var actionModalityEmbedding: MLXArray?

    public init(configuration: Cosmos3TransformerConfiguration = Cosmos3TransformerConfiguration()) {
        precondition(configuration.validationIssues().isEmpty)
        self.configuration = configuration
        self.rotaryEmbedding = Cosmos3RotaryEmbedding(
            headDimension: configuration.headDimension,
            theta: configuration.ropeTheta,
            axesDimensions: configuration.ropeAxesDimensions
        )
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.hiddenSize
        )
        self._layers.wrappedValue = (0..<configuration.layerCount).map { _ in
            Cosmos3DecoderLayer(configuration: configuration)
        }
        self._understandingNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon
        )
        self._generationNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon
        )
        self._languageModelHead.wrappedValue = configuration.includesLanguageModelHead
            ? Linear(
                configuration.hiddenSize,
                configuration.vocabularySize,
                bias: false
            )
            : nil
        self._visionInputProjection.wrappedValue = Linear(
            configuration.patchLatentDimension,
            configuration.hiddenSize,
            bias: true
        )
        self._visionOutputProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.patchLatentDimension,
            bias: true
        )
        self._timestepEmbedding.wrappedValue = Cosmos3TimestepEmbedding(
            hiddenSize: configuration.hiddenSize
        )
        self._actionInputProjection.wrappedValue = configuration.generatesActions
            ? Cosmos3DomainAwareLinear(
                inputSize: configuration.actionDimension!,
                outputSize: configuration.hiddenSize,
                domainCount: configuration.embodimentDomainCount
            )
            : nil
        self._actionOutputProjection.wrappedValue = configuration.generatesActions
            ? Cosmos3DomainAwareLinear(
                inputSize: configuration.hiddenSize,
                outputSize: configuration.actionDimension!,
                domainCount: configuration.embodimentDomainCount
            )
            : nil
        self._actionModalityEmbedding.wrappedValue = configuration.generatesActions
            ? MLX.zeros([configuration.hiddenSize])
            : nil
    }

    public func embedText(tokenIDs: MLXArray) -> MLXArray {
        tokenEmbedding(tokenIDs.asType(.int32))
    }

    public func projectVision(_ tokens: MLXArray, timesteps: MLXArray) -> MLXArray {
        visionInputProjection(tokens) + timestepEmbedding(
            timesteps.asType(.float32) * configuration.timestepScale
        ).asType(tokens.dtype)
    }

    public func projectActions(
        _ actions: MLXArray,
        domainIDs: MLXArray,
        timesteps: MLXArray?
    ) -> MLXArray {
        precondition(configuration.generatesActions)
        var projected = actionInputProjection!(actions, domainIDs: domainIDs)
            + actionModalityEmbedding!
        if let timesteps {
            projected = projected + timestepEmbedding(
                timesteps.asType(.float32) * configuration.timestepScale
            ).asType(projected.dtype)
        }
        return projected
    }

    public func callAsFunction(
        understanding: MLXArray,
        generation: MLXArray,
        positionIDs: MLXArray
    ) -> (understanding: MLXArray, generation: MLXArray) {
        precondition(understanding.ndim == 2 && generation.ndim == 2)
        precondition(positionIDs.shape == [3, understanding.dim(0) + generation.dim(0)])
        let rotary = rotaryEmbedding(positionIDs: positionIDs, dtype: understanding.dtype)
        let understandingLength = understanding.dim(0)
        let layerRotary = (
            understandingCosine: rotary.cosine[0..<understandingLength],
            understandingSine: rotary.sine[0..<understandingLength],
            generationCosine: rotary.cosine[understandingLength...],
            generationSine: rotary.sine[understandingLength...]
        )
        var understandingHidden = understanding
        var generationHidden = generation
        for layer in layers {
            (understandingHidden, generationHidden) = layer(
                understanding: understandingHidden,
                generation: generationHidden,
                rotary: layerRotary
            )
        }
        return (
            understandingNorm(understandingHidden),
            generationNorm(generationHidden)
        )
    }

    public func visionPredictions(_ hidden: MLXArray) -> MLXArray {
        visionOutputProjection(hidden)
    }

    public func actionPredictions(_ hidden: MLXArray, domainIDs: MLXArray) -> MLXArray {
        precondition(configuration.generatesActions)
        return actionOutputProjection!(hidden, domainIDs: domainIDs)
    }

    public func reasonerHidden(
        inputEmbeddings: MLXArray,
        positionIDs: MLXArray
    ) -> MLXArray {
        precondition(inputEmbeddings.ndim == 2)
        precondition(positionIDs.shape == [3, inputEmbeddings.dim(0)])
        let rotary = rotaryEmbedding(
            positionIDs: positionIDs,
            dtype: inputEmbeddings.dtype
        )
        var hidden = inputEmbeddings
        for layer in layers {
            hidden = layer.understandingOnly(
                hidden,
                cosine: rotary.cosine,
                sine: rotary.sine
            )
        }
        return understandingNorm(hidden)
    }

    public func reasonerLogits(
        tokenIDs: MLXArray,
        positionIDs: MLXArray? = nil
    ) -> MLXArray {
        let tokens = tokenIDs.reshaped(-1).asType(.int32)
        let positions = positionIDs ?? MLX.stacked(
            Array(repeating: MLXArray(0..<tokens.size), count: 3),
            axis: 0
        )
        precondition(configuration.includesLanguageModelHead)
        return languageModelHead!(reasonerHidden(
            inputEmbeddings: embedText(tokenIDs: tokens),
            positionIDs: positions
        ))
    }
}
