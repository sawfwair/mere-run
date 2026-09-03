import Foundation
import MLX
import MLXFast
import MLXNN

private func flux1LayerNorm(_ value: MLXArray, epsilon: Float = 1e-6) -> MLXArray {
    let source = value.asType(.float32)
    let mean = source.mean(axis: -1, keepDims: true)
    let centered = source - mean
    let variance = (centered * centered).mean(axis: -1, keepDims: true)
    return (centered / MLX.sqrt(variance + epsilon)).asType(value.dtype)
}

final class Flux1TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inputSize: Int, hiddenSize: Int) {
        self._linear1.wrappedValue = Linear(inputSize, hiddenSize, bias: true)
        self._linear2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        linear2(silu(linear1(value)))
    }
}

final class Flux1TimeTextEmbedding: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: Flux1TimestepEmbedding
    @ModuleInfo(key: "guidance_embedder") var guidanceEmbedder: Flux1TimestepEmbedding?
    @ModuleInfo(key: "text_embedder") var textEmbedder: Flux1TimestepEmbedding

    init(hiddenSize: Int, pooledProjectionSize: Int, guidanceEmbeds: Bool) {
        self._timestepEmbedder.wrappedValue = Flux1TimestepEmbedding(inputSize: 256, hiddenSize: hiddenSize)
        self._guidanceEmbedder.wrappedValue = guidanceEmbeds
            ? Flux1TimestepEmbedding(inputSize: 256, hiddenSize: hiddenSize)
            : nil
        self._textEmbedder.wrappedValue = Flux1TimestepEmbedding(
            inputSize: pooledProjectionSize,
            hiddenSize: hiddenSize
        )
        super.init()
    }

    func callAsFunction(
        timestep: MLXArray,
        guidance: MLXArray?,
        pooledProjection: MLXArray
    ) -> MLXArray {
        var embedding = timestepEmbedder(Self.sinusoidal(timestep))
        if let guidance, let guidanceEmbedder {
            embedding = embedding + guidanceEmbedder(Self.sinusoidal(guidance))
        }
        return embedding + textEmbedder(pooledProjection)
    }

    private static func sinusoidal(_ values: MLXArray) -> MLXArray {
        let half = 128
        let indices = MLXArray(0..<half).asType(.float32)
        let frequencies = MLX.exp(-log(Float(10_000)) * indices / Float(half))
        let angles = values.asType(.float32).expandedDimensions(axis: -1) * frequencies
        return MLX.concatenated([MLX.cos(angles), MLX.sin(angles)], axis: -1)
    }
}

final class Flux1AdaLayerNormZero: Module {
    @ModuleInfo(key: "linear") var linear: Linear

    init(hiddenSize: Int) {
        self._linear.wrappedValue = Linear(hiddenSize, hiddenSize * 6, bias: true)
        super.init()
    }

    func callAsFunction(
        _ value: MLXArray,
        embedding: MLXArray
    ) -> (
        normalized: MLXArray,
        attentionGate: MLXArray,
        mlpShift: MLXArray,
        mlpScale: MLXArray,
        mlpGate: MLXArray
    ) {
        let chunks = MLX.split(linear(silu(embedding)), parts: 6, axis: -1)
        let shift = chunks[0].expandedDimensions(axis: 1)
        let scale = chunks[1].expandedDimensions(axis: 1)
        return (
            flux1LayerNorm(value) * (1 + scale) + shift,
            chunks[2].expandedDimensions(axis: 1),
            chunks[3].expandedDimensions(axis: 1),
            chunks[4].expandedDimensions(axis: 1),
            chunks[5].expandedDimensions(axis: 1)
        )
    }
}

final class Flux1AdaLayerNormZeroSingle: Module {
    @ModuleInfo(key: "linear") var linear: Linear

    init(hiddenSize: Int) {
        self._linear.wrappedValue = Linear(hiddenSize, hiddenSize * 3, bias: true)
        super.init()
    }

    func callAsFunction(_ value: MLXArray, embedding: MLXArray) -> (MLXArray, MLXArray) {
        let chunks = MLX.split(linear(silu(embedding)), parts: 3, axis: -1)
        let shift = chunks[0].expandedDimensions(axis: 1)
        let scale = chunks[1].expandedDimensions(axis: 1)
        let gate = chunks[2].expandedDimensions(axis: 1)
        return (flux1LayerNorm(value) * (1 + scale) + shift, gate)
    }
}

final class Flux1AdaLayerNormContinuous: Module {
    @ModuleInfo(key: "linear") var linear: Linear

    init(hiddenSize: Int) {
        self._linear.wrappedValue = Linear(hiddenSize, hiddenSize * 2, bias: true)
        super.init()
    }

    func callAsFunction(_ value: MLXArray, embedding: MLXArray) -> MLXArray {
        let chunks = MLX.split(linear(silu(embedding)), parts: 2, axis: -1)
        let scale = chunks[0].expandedDimensions(axis: 1)
        let shift = chunks[1].expandedDimensions(axis: 1)
        return flux1LayerNorm(value) * (1 + scale) + shift
    }
}

final class Flux1GELUProjection: Module {
    @ModuleInfo(key: "proj") var projection: Linear

    init(inputSize: Int, outputSize: Int) {
        self._projection.wrappedValue = Linear(inputSize, outputSize, bias: true)
        super.init()
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        MLXNN.geluApproximate(projection(value))
    }
}

final class Flux1FeedForward: Module {
    @ModuleInfo(key: "input") var input: Flux1GELUProjection
    @ModuleInfo(key: "output") var output: Linear

    init(hiddenSize: Int) {
        self._input.wrappedValue = Flux1GELUProjection(inputSize: hiddenSize, outputSize: hiddenSize * 4)
        self._output.wrappedValue = Linear(hiddenSize * 4, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        output(input(value))
    }
}

final class Flux1JointAttention: Module {
    let headCount: Int
    let headSize: Int

    @ModuleInfo(key: "to_q") var query: Linear
    @ModuleInfo(key: "to_k") var key: Linear
    @ModuleInfo(key: "to_v") var value: Linear
    @ModuleInfo(key: "norm_q") var queryNorm: RMSNorm
    @ModuleInfo(key: "norm_k") var keyNorm: RMSNorm
    @ModuleInfo(key: "to_out") var output: [Linear]

    @ModuleInfo(key: "add_q_proj") var contextQuery: Linear
    @ModuleInfo(key: "add_k_proj") var contextKey: Linear
    @ModuleInfo(key: "add_v_proj") var contextValue: Linear
    @ModuleInfo(key: "norm_added_q") var contextQueryNorm: RMSNorm
    @ModuleInfo(key: "norm_added_k") var contextKeyNorm: RMSNorm
    @ModuleInfo(key: "to_add_out") var contextOutput: Linear

    init(hiddenSize: Int, headCount: Int, headSize: Int) {
        self.headCount = headCount
        self.headSize = headSize
        self._query.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._key.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._value.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._queryNorm.wrappedValue = RMSNorm(dimensions: headSize, eps: 1e-6)
        self._keyNorm.wrappedValue = RMSNorm(dimensions: headSize, eps: 1e-6)
        self._output.wrappedValue = [Linear(hiddenSize, hiddenSize, bias: true)]
        self._contextQuery.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._contextKey.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._contextValue.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._contextQueryNorm.wrappedValue = RMSNorm(dimensions: headSize, eps: 1e-6)
        self._contextKeyNorm.wrappedValue = RMSNorm(dimensions: headSize, eps: 1e-6)
        self._contextOutput.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(
        image: MLXArray,
        context: MLXArray,
        rotary: (MLXArray, MLXArray)
    ) -> (image: MLXArray, context: MLXArray) {
        let batch = image.dim(0)
        let imageLength = image.dim(1)
        let contextLength = context.dim(1)

        func heads(_ value: MLXArray) -> MLXArray {
            value.reshaped(batch, -1, headCount, headSize).transposed(0, 2, 1, 3)
        }

        let imageQuery = queryNorm(heads(query(image)))
        let imageKey = keyNorm(heads(key(image)))
        let imageValue = heads(value(image))
        let textQuery = contextQueryNorm(heads(contextQuery(context)))
        let textKey = contextKeyNorm(heads(contextKey(context)))
        let textValue = heads(contextValue(context))

        let jointQuery = Flux2PosEmbed.applyRotaryEmb(
            MLX.concatenated([textQuery, imageQuery], axis: 2),
            freqs: rotary
        )
        let jointKey = Flux2PosEmbed.applyRotaryEmb(
            MLX.concatenated([textKey, imageKey], axis: 2),
            freqs: rotary
        )
        let jointValue = MLX.concatenated([textValue, imageValue], axis: 2)
        let attended = MLXFast.scaledDotProductAttention(
            queries: jointQuery,
            keys: jointKey,
            values: jointValue,
            scale: 1 / sqrt(Float(headSize)),
            mask: .none
        ).transposed(0, 2, 1, 3).reshaped(batch, contextLength + imageLength, headCount * headSize)
        let text = attended[0..., 0..<contextLength, 0...]
        let image = attended[0..., contextLength..., 0...]
        return (output[0](image), contextOutput(text))
    }
}

final class Flux1TransformerBlock: Module {
    @ModuleInfo(key: "norm1") var imageNorm: Flux1AdaLayerNormZero
    @ModuleInfo(key: "norm1_context") var contextNorm: Flux1AdaLayerNormZero
    @ModuleInfo(key: "attn") var attention: Flux1JointAttention
    @ModuleInfo(key: "ff") var imageFeedForward: Flux1FeedForward
    @ModuleInfo(key: "ff_context") var contextFeedForward: Flux1FeedForward

    init(configuration: Flux1TransformerConfiguration) {
        let hiddenSize = configuration.hiddenSize
        self._imageNorm.wrappedValue = Flux1AdaLayerNormZero(hiddenSize: hiddenSize)
        self._contextNorm.wrappedValue = Flux1AdaLayerNormZero(hiddenSize: hiddenSize)
        self._attention.wrappedValue = Flux1JointAttention(
            hiddenSize: hiddenSize,
            headCount: configuration.numAttentionHeads,
            headSize: configuration.attentionHeadDim
        )
        self._imageFeedForward.wrappedValue = Flux1FeedForward(hiddenSize: hiddenSize)
        self._contextFeedForward.wrappedValue = Flux1FeedForward(hiddenSize: hiddenSize)
        super.init()
    }

    func callAsFunction(
        image: MLXArray,
        context: MLXArray,
        embedding: MLXArray,
        rotary: (MLXArray, MLXArray)
    ) -> (context: MLXArray, image: MLXArray) {
        let imageModulation = imageNorm(image, embedding: embedding)
        let contextModulation = contextNorm(context, embedding: embedding)
        let attended = attention(
            image: imageModulation.normalized,
            context: contextModulation.normalized,
            rotary: rotary
        )

        var nextImage = image + imageModulation.attentionGate * attended.image
        var nextContext = context + contextModulation.attentionGate * attended.context
        let normalizedImage = flux1LayerNorm(nextImage)
            * (1 + imageModulation.mlpScale) + imageModulation.mlpShift
        let normalizedContext = flux1LayerNorm(nextContext)
            * (1 + contextModulation.mlpScale) + contextModulation.mlpShift
        nextImage = nextImage + imageModulation.mlpGate * imageFeedForward(normalizedImage)
        nextContext = nextContext + contextModulation.mlpGate * contextFeedForward(normalizedContext)
        return (nextContext, nextImage)
    }
}

final class Flux1SelfAttention: Module {
    let headCount: Int
    let headSize: Int

    @ModuleInfo(key: "to_q") var query: Linear
    @ModuleInfo(key: "to_k") var key: Linear
    @ModuleInfo(key: "to_v") var value: Linear
    @ModuleInfo(key: "norm_q") var queryNorm: RMSNorm
    @ModuleInfo(key: "norm_k") var keyNorm: RMSNorm

    init(hiddenSize: Int, headCount: Int, headSize: Int) {
        self.headCount = headCount
        self.headSize = headSize
        self._query.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._key.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._value.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._queryNorm.wrappedValue = RMSNorm(dimensions: headSize, eps: 1e-6)
        self._keyNorm.wrappedValue = RMSNorm(dimensions: headSize, eps: 1e-6)
        super.init()
    }

    func callAsFunction(_ input: MLXArray, rotary: (MLXArray, MLXArray)) -> MLXArray {
        let batch = input.dim(0)
        func heads(_ value: MLXArray) -> MLXArray {
            value.reshaped(batch, -1, headCount, headSize).transposed(0, 2, 1, 3)
        }
        let q = Flux2PosEmbed.applyRotaryEmb(queryNorm(heads(query(input))), freqs: rotary)
        let k = Flux2PosEmbed.applyRotaryEmb(keyNorm(heads(key(input))), freqs: rotary)
        let v = heads(value(input))
        return MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: 1 / sqrt(Float(headSize)),
            mask: .none
        ).transposed(0, 2, 1, 3).reshaped(batch, -1, headCount * headSize)
    }
}

final class Flux1SingleTransformerBlock: Module {
    @ModuleInfo(key: "norm") var norm: Flux1AdaLayerNormZeroSingle
    @ModuleInfo(key: "proj_mlp") var mlpProjection: Linear
    @ModuleInfo(key: "attn") var attention: Flux1SelfAttention
    @ModuleInfo(key: "proj_out") var outputProjection: Linear

    init(configuration: Flux1TransformerConfiguration) {
        let hiddenSize = configuration.hiddenSize
        self._norm.wrappedValue = Flux1AdaLayerNormZeroSingle(hiddenSize: hiddenSize)
        self._mlpProjection.wrappedValue = Linear(hiddenSize, hiddenSize * 4, bias: true)
        self._attention.wrappedValue = Flux1SelfAttention(
            hiddenSize: hiddenSize,
            headCount: configuration.numAttentionHeads,
            headSize: configuration.attentionHeadDim
        )
        self._outputProjection.wrappedValue = Linear(hiddenSize * 5, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        embedding: MLXArray,
        rotary: (MLXArray, MLXArray)
    ) -> MLXArray {
        let (normalized, gate) = norm(input, embedding: embedding)
        let attended = attention(normalized, rotary: rotary)
        let mlp = MLXNN.geluApproximate(mlpProjection(normalized))
        return input + gate * outputProjection(MLX.concatenated([attended, mlp], axis: -1))
    }
}

public final class Flux1Transformer: Module {
    public let configuration: Flux1TransformerConfiguration

    @ModuleInfo(key: "x_embedder") var imageEmbedder: Linear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "time_text_embed") var timeTextEmbedder: Flux1TimeTextEmbedding
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [Flux1TransformerBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleTransformerBlocks: [Flux1SingleTransformerBlock]
    @ModuleInfo(key: "norm_out") var outputNorm: Flux1AdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var outputProjection: Linear
    private let positionEmbedder: Flux2PosEmbed

    public init(configuration: Flux1TransformerConfiguration) {
        self.configuration = configuration
        self._imageEmbedder.wrappedValue = Linear(
            configuration.inChannels,
            configuration.hiddenSize,
            bias: true
        )
        self._contextEmbedder.wrappedValue = Linear(
            configuration.jointAttentionDim,
            configuration.hiddenSize,
            bias: true
        )
        self._timeTextEmbedder.wrappedValue = Flux1TimeTextEmbedding(
            hiddenSize: configuration.hiddenSize,
            pooledProjectionSize: configuration.pooledProjectionDim,
            guidanceEmbeds: configuration.guidanceEmbeds
        )
        self._transformerBlocks.wrappedValue = (0..<configuration.numLayers).map { _ in
            Flux1TransformerBlock(configuration: configuration)
        }
        self._singleTransformerBlocks.wrappedValue = (0..<configuration.numSingleLayers).map { _ in
            Flux1SingleTransformerBlock(configuration: configuration)
        }
        self._outputNorm.wrappedValue = Flux1AdaLayerNormContinuous(hiddenSize: configuration.hiddenSize)
        self._outputProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.patchSize * configuration.patchSize * configuration.outChannels,
            bias: true
        )
        self.positionEmbedder = Flux2PosEmbed(theta: 10_000, axesDim: configuration.axesDimsRope)
        super.init()
    }

    public func callAsFunction(
        image: MLXArray,
        context: MLXArray,
        pooledProjection: MLXArray,
        timestep: MLXArray,
        imageIDs: MLXArray,
        textIDs: MLXArray,
        guidance: MLXArray?
    ) -> MLXArray {
        var imageHidden = imageEmbedder(image)
        var contextHidden = contextEmbedder(context)
        let embedding = timeTextEmbedder(
            timestep: timestep.asType(imageHidden.dtype) * 1_000,
            guidance: guidance.map { $0.asType(imageHidden.dtype) * 1_000 },
            pooledProjection: pooledProjection
        )
        let ids = MLX.concatenated([textIDs, imageIDs], axis: 0)
        let rotary = positionEmbedder(ids)
        for block in transformerBlocks {
            (contextHidden, imageHidden) = block(
                image: imageHidden,
                context: contextHidden,
                embedding: embedding,
                rotary: rotary
            )
        }
        let textLength = contextHidden.dim(1)
        var combined = MLX.concatenated([contextHidden, imageHidden], axis: 1)
        for block in singleTransformerBlocks {
            combined = block(combined, embedding: embedding, rotary: rotary)
        }
        imageHidden = combined[0..., textLength..., 0...]
        return outputProjection(outputNorm(imageHidden, embedding: embedding))
    }
}

enum Flux1TransformerWeightMapper {
    static func map(key: String, value: MLXArray) -> [(String, MLXArray)] {
        var target = key
        target = target.replacingOccurrences(of: ".ff.net.0.proj", with: ".ff.input.proj")
        target = target.replacingOccurrences(of: ".ff.net.2", with: ".ff.output")
        target = target.replacingOccurrences(of: ".ff_context.net.0.proj", with: ".ff_context.input.proj")
        target = target.replacingOccurrences(of: ".ff_context.net.2", with: ".ff_context.output")
        return [(target, value)]
    }
}
