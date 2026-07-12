import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

public struct TripoSRSceneCode {
    /// `[batch, 3, planeHeight, planeWidth, channels]`.
    public let planes: MLXArray

    public init(planes: MLXArray) {
        precondition(planes.ndim == 5 && planes.dim(1) == 3)
        self.planes = planes
    }
}

struct TripoSRForwardDiagnostics {
    /// `[batch, imageToken, imageChannel]`.
    let imageTokens: MLXArray
    /// `[batch, triplaneChannel, triplaneToken]`.
    let initialTriplaneTokens: MLXArray
    let finalTriplaneTokens: MLXArray
    let sceneCode: TripoSRSceneCode
}

private final class TripoSRViTPatchEmbeddings: Module {
    @ModuleInfo(key: "projection") var projection: Conv2d

    init(configuration: TripoSRConfiguration) {
        self._projection.wrappedValue = Conv2d(
            inputChannels: 3,
            outputChannels: configuration.imageHiddenSize,
            kernelSize: IntOrPair(configuration.imagePatchSize),
            stride: IntOrPair(configuration.imagePatchSize),
            padding: IntOrPair(0),
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let patches = projection(input)
        return patches.reshaped(patches.dim(0), patches.dim(1) * patches.dim(2), patches.dim(3))
    }
}

private final class TripoSRViTEmbeddings: Module {
    let configuration: TripoSRConfiguration

    @ParameterInfo(key: "cls_token") var classToken: MLXArray
    @ParameterInfo(key: "position_embeddings") var positionEmbeddings: MLXArray
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: TripoSRViTPatchEmbeddings

    init(configuration: TripoSRConfiguration) {
        self.configuration = configuration
        self._classToken.wrappedValue = MLX.zeros([1, 1, configuration.imageHiddenSize], dtype: .float32)
        self._positionEmbeddings.wrappedValue = MLX.zeros(
            [1, 1 + configuration.imagePositionGridSize * configuration.imagePositionGridSize,
             configuration.imageHiddenSize],
            dtype: .float32
        )
        self._patchEmbeddings.wrappedValue = TripoSRViTPatchEmbeddings(configuration: configuration)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let gridHeight = input.dim(1) / configuration.imagePatchSize
        let gridWidth = input.dim(2) / configuration.imagePatchSize
        let patches = patchEmbeddings(input)
        let classTokens = MLX.broadcast(
            classToken.asType(input.dtype),
            to: [batch, 1, configuration.imageHiddenSize]
        )
        var hidden = MLX.concatenated([classTokens, patches], axis: 1)
        hidden = hidden + interpolatedPositions(
            gridHeight: gridHeight,
            gridWidth: gridWidth,
            dtype: hidden.dtype
        )
        return hidden
    }

    private func interpolatedPositions(gridHeight: Int, gridWidth: Int, dtype: DType) -> MLXArray {
        let sourceSize = configuration.imagePositionGridSize
        let classPosition = positionEmbeddings[0..., 0..<1, 0...].asType(dtype)
        let patchPositions = positionEmbeddings[0..., 1..., 0...]
            .reshaped(1, sourceSize, sourceSize, configuration.imageHiddenSize)
        let resized: MLXArray
        if gridHeight == sourceSize && gridWidth == sourceSize {
            resized = patchPositions.asType(dtype)
        } else {
            // transformers==4.35.0 follows DINO's scale-factor interpolation
            // with the historical +0.1 output-size offset.
            resized = dinoV2PyTorchBicubicResize(
                patchPositions.asType(.float32),
                outputHeight: gridHeight,
                outputWidth: gridWidth,
                offset: configuration.imagePositionInterpolationOffset
            ).asType(dtype)
        }
        return MLX.concatenated(
            [classPosition, resized.reshaped(1, gridHeight * gridWidth, configuration.imageHiddenSize)],
            axis: 1
        )
    }
}

private final class TripoSRViTSelfAttention: Module {
    let headCount: Int
    let headDimension: Int
    let scale: Float

    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear

    init(configuration: TripoSRConfiguration) {
        self.headCount = configuration.imageHeadCount
        self.headDimension = configuration.imageHiddenSize / configuration.imageHeadCount
        self.scale = 1 / sqrt(Float(headDimension))
        self._query.wrappedValue = Linear(configuration.imageHiddenSize, configuration.imageHiddenSize, bias: true)
        self._key.wrappedValue = Linear(configuration.imageHiddenSize, configuration.imageHiddenSize, bias: true)
        self._value.wrappedValue = Linear(configuration.imageHiddenSize, configuration.imageHiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let queries = query(input)
            .reshaped(batch, sequence, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        let keys = key(input)
            .reshaped(batch, sequence, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        let values = value(input)
            .reshaped(batch, sequence, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .none
        ).transposed(0, 2, 1, 3).reshaped(batch, sequence, headCount * headDimension)
    }
}

private final class TripoSRViTSelfOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear

    init(configuration: TripoSRConfiguration) {
        self._dense.wrappedValue = Linear(configuration.imageHiddenSize, configuration.imageHiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { dense(input) }
}

private final class TripoSRViTAttention: Module {
    @ModuleInfo(key: "attention") var attention: TripoSRViTSelfAttention
    @ModuleInfo(key: "output") var output: TripoSRViTSelfOutput

    init(configuration: TripoSRConfiguration) {
        self._attention.wrappedValue = TripoSRViTSelfAttention(configuration: configuration)
        self._output.wrappedValue = TripoSRViTSelfOutput(configuration: configuration)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { output(attention(input)) }
}

private final class TripoSRViTIntermediate: Module {
    @ModuleInfo(key: "dense") var dense: Linear

    init(configuration: TripoSRConfiguration) {
        self._dense.wrappedValue = Linear(
            configuration.imageHiddenSize,
            configuration.imageIntermediateSize,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { gelu(dense(input)) }
}

private final class TripoSRViTOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear

    init(configuration: TripoSRConfiguration) {
        self._dense.wrappedValue = Linear(
            configuration.imageIntermediateSize,
            configuration.imageHiddenSize,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { dense(input) }
}

private final class TripoSRViTLayer: Module {
    @ModuleInfo(key: "attention") var attention: TripoSRViTAttention
    @ModuleInfo(key: "intermediate") var intermediate: TripoSRViTIntermediate
    @ModuleInfo(key: "output") var output: TripoSRViTOutput
    @ModuleInfo(key: "layernorm_before") var layerNormBefore: LayerNorm
    @ModuleInfo(key: "layernorm_after") var layerNormAfter: LayerNorm

    init(configuration: TripoSRConfiguration) {
        self._attention.wrappedValue = TripoSRViTAttention(configuration: configuration)
        self._intermediate.wrappedValue = TripoSRViTIntermediate(configuration: configuration)
        self._output.wrappedValue = TripoSRViTOutput(configuration: configuration)
        self._layerNormBefore.wrappedValue = LayerNorm(
            dimensions: configuration.imageHiddenSize,
            eps: configuration.imageLayerNormEpsilon
        )
        self._layerNormAfter.wrappedValue = LayerNorm(
            dimensions: configuration.imageHiddenSize,
            eps: configuration.imageLayerNormEpsilon
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let afterAttention = input + attention(layerNormBefore(input))
        return afterAttention + output(intermediate(layerNormAfter(afterAttention)))
    }
}

private final class TripoSRViTEncoder: Module {
    @ModuleInfo(key: "layer") var layers: [TripoSRViTLayer]

    init(configuration: TripoSRConfiguration) {
        self._layers.wrappedValue = (0..<configuration.imageLayerCount).map { _ in
            TripoSRViTLayer(configuration: configuration)
        }
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden = input
        for layer in layers {
            hidden = layer(hidden)
            // TripoSR is inference-only. Bound the lazy Metal command graph at
            // encoder block boundaries to avoid macOS watchdog timeouts and
            // release intermediate activations before the 3072-token decoder.
            MLX.eval(hidden)
        }
        return hidden
    }
}

private final class TripoSRViTPooler: Module {
    @ModuleInfo(key: "dense") var dense: Linear

    init(configuration: TripoSRConfiguration) {
        self._dense.wrappedValue = Linear(configuration.imageHiddenSize, configuration.imageHiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        MLX.tanh(dense(input[0..., 0, 0...]))
    }
}

private final class TripoSRViTModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: TripoSRViTEmbeddings
    @ModuleInfo(key: "encoder") var encoder: TripoSRViTEncoder
    @ModuleInfo(key: "layernorm") var layerNorm: LayerNorm
    @ModuleInfo(key: "pooler") var pooler: TripoSRViTPooler

    init(configuration: TripoSRConfiguration) {
        self._embeddings.wrappedValue = TripoSRViTEmbeddings(configuration: configuration)
        self._encoder.wrappedValue = TripoSRViTEncoder(configuration: configuration)
        self._layerNorm.wrappedValue = LayerNorm(
            dimensions: configuration.imageHiddenSize,
            eps: configuration.imageLayerNormEpsilon
        )
        self._pooler.wrappedValue = TripoSRViTPooler(configuration: configuration)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        layerNorm(encoder(embeddings(input)))
    }
}

private final class TripoSRImageTokenizer: Module {
    @ModuleInfo(key: "model") var model: TripoSRViTModel

    init(configuration: TripoSRConfiguration) {
        self._model.wrappedValue = TripoSRViTModel(configuration: configuration)
        super.init()
    }

    /// Input is NHWC RGB in `[0, 1]`, matching upstream `TSR.forward`.
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        precondition(input.ndim == 4 && input.dim(3) == 3)
        let mean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped(1, 1, 1, 3)
        let standardDeviation = MLXArray([Float(0.229), 0.224, 0.225]).reshaped(1, 1, 1, 3)
        return model((input - mean.asType(input.dtype)) / standardDeviation.asType(input.dtype))
    }
}

private final class TripoSRTriplaneTokenizer: Module {
    let configuration: TripoSRConfiguration

    /// Kept in the exact upstream `[plane, channel, height, width]` layout.
    @ParameterInfo(key: "embeddings") var embeddings: MLXArray

    init(configuration: TripoSRConfiguration) {
        self.configuration = configuration
        self._embeddings.wrappedValue = MLX.zeros(
            [3, configuration.tokenChannels, configuration.planeSize, configuration.planeSize],
            dtype: .float32
        )
        super.init()
    }

    func callAsFunction(batchSize: Int, dtype: DType) -> MLXArray {
        MLX.broadcast(
            embeddings.asType(dtype).expandedDimensions(axis: 0),
            to: [batchSize, 3, configuration.tokenChannels, configuration.planeSize, configuration.planeSize]
        )
        .transposed(0, 2, 1, 3, 4)
        .reshaped(batchSize, configuration.tokenChannels, configuration.triplaneTokenCount)
    }

    func detokenize(_ tokens: MLXArray) -> MLXArray {
        precondition(tokens.ndim == 3)
        precondition(tokens.dim(1) == configuration.tokenChannels)
        precondition(tokens.dim(2) == configuration.triplaneTokenCount)
        return tokens
            .reshaped(tokens.dim(0), configuration.tokenChannels, 3, configuration.planeSize, configuration.planeSize)
            .transposed(0, 2, 1, 3, 4)
    }
}

private final class TripoSRAttention: Module {
    let headCount: Int
    let headDimension: Int
    let scale: Float
    let queryChunkSize: Int

    @ModuleInfo(key: "to_q") var query: Linear
    @ModuleInfo(key: "to_k") var key: Linear
    @ModuleInfo(key: "to_v") var value: Linear
    @ModuleInfo(key: "to_out") var output: [UnaryLayer]

    init(
        queryChannels: Int,
        crossAttentionChannels: Int,
        configuration: TripoSRConfiguration,
        queryChunkSize: Int
    ) {
        self.headCount = configuration.transformerHeadCount
        self.headDimension = configuration.transformerHeadDimension
        self.scale = 1 / sqrt(Float(headDimension))
        self.queryChunkSize = queryChunkSize
        let innerChannels = headCount * headDimension
        precondition(queryChannels == innerChannels)
        self._query.wrappedValue = Linear(queryChannels, innerChannels, bias: false)
        self._key.wrappedValue = Linear(crossAttentionChannels, innerChannels, bias: false)
        self._value.wrappedValue = Linear(crossAttentionChannels, innerChannels, bias: false)
        self._output.wrappedValue = [
            Linear(innerChannels, queryChannels, bias: true),
            Identity(),
        ]
        super.init()
    }

    func callAsFunction(_ input: MLXArray, encoderHiddenStates: MLXArray? = nil) -> MLXArray {
        let context = encoderHiddenStates ?? input
        let batch = input.dim(0)
        let queryCount = input.dim(1)
        let contextCount = context.dim(1)
        let queries = query(input)
            .reshaped(batch, queryCount, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        let keys = key(context)
            .reshaped(batch, contextCount, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        let values = value(context)
            .reshaped(batch, contextCount, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        let attended: MLXArray
        if queryCount <= queryChunkSize {
            attended = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .none
            )
        } else {
            var chunks: [MLXArray] = []
            chunks.reserveCapacity((queryCount + queryChunkSize - 1) / queryChunkSize)
            for start in stride(from: 0, to: queryCount, by: queryChunkSize) {
                let end = min(start + queryChunkSize, queryCount)
                let chunk = MLXFast.scaledDotProductAttention(
                    queries: queries[0..., 0..., start..<end, 0...],
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: .none
                )
                MLX.eval(chunk)
                chunks.append(chunk)
            }
            attended = MLX.concatenated(chunks, axis: 2)
        }
        let merged = attended.transposed(0, 2, 1, 3)
            .reshaped(batch, queryCount, headCount * headDimension)
        return output.reduce(merged) { hidden, layer in layer(hidden) }
    }
}

private final class TripoSRGEGLU: Module, UnaryLayer {
    let innerChannels: Int
    @ModuleInfo(key: "proj") var projection: Linear

    init(channels: Int, multiplier: Int) {
        self.innerChannels = channels * multiplier
        self._projection.wrappedValue = Linear(channels, innerChannels * 2, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let projected = projection(input)
        let hidden = projected[0..., 0..., 0..<innerChannels]
        let gate = projected[0..., 0..., innerChannels...]
        return hidden * gelu(gate)
    }
}

private final class TripoSRFeedForward: Module {
    let tokenChunkSize: Int
    @ModuleInfo(key: "net") var network: [UnaryLayer]

    init(configuration: TripoSRConfiguration, tokenChunkSize: Int) {
        self.tokenChunkSize = tokenChunkSize
        self._network.wrappedValue = [
            TripoSRGEGLU(
                channels: configuration.tokenChannels,
                multiplier: configuration.transformerFeedForwardMultiplier
            ),
            Identity(),
            Linear(
                configuration.tokenChannels * configuration.transformerFeedForwardMultiplier,
                configuration.tokenChannels,
                bias: true
            ),
        ]
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let tokenCount = input.dim(1)
        if tokenCount <= tokenChunkSize {
            return network.reduce(input) { hidden, layer in layer(hidden) }
        }
        var chunks: [MLXArray] = []
        chunks.reserveCapacity((tokenCount + tokenChunkSize - 1) / tokenChunkSize)
        for start in stride(from: 0, to: tokenCount, by: tokenChunkSize) {
            let end = min(start + tokenChunkSize, tokenCount)
            let chunk = network.reduce(input[0..., start..<end, 0...]) { hidden, layer in
                layer(hidden)
            }
            MLX.eval(chunk)
            chunks.append(chunk)
        }
        return MLX.concatenated(chunks, axis: 1)
    }
}

private final class TripoSRTransformerBlock: Module {
    @ModuleInfo(key: "norm1") var firstNorm: LayerNorm
    @ModuleInfo(key: "attn1") var selfAttention: TripoSRAttention
    @ModuleInfo(key: "norm2") var secondNorm: LayerNorm
    @ModuleInfo(key: "attn2") var crossAttention: TripoSRAttention
    @ModuleInfo(key: "norm3") var thirdNorm: LayerNorm
    @ModuleInfo(key: "ff") var feedForward: TripoSRFeedForward

    init(configuration: TripoSRConfiguration, memory: TripoSRMemoryConfiguration) {
        self._firstNorm.wrappedValue = LayerNorm(
            dimensions: configuration.tokenChannels,
            eps: configuration.transformerLayerNormEpsilon
        )
        self._selfAttention.wrappedValue = TripoSRAttention(
            queryChannels: configuration.tokenChannels,
            crossAttentionChannels: configuration.tokenChannels,
            configuration: configuration,
            queryChunkSize: memory.attentionQueryChunkSize
        )
        self._secondNorm.wrappedValue = LayerNorm(
            dimensions: configuration.tokenChannels,
            eps: configuration.transformerLayerNormEpsilon
        )
        self._crossAttention.wrappedValue = TripoSRAttention(
            queryChannels: configuration.tokenChannels,
            crossAttentionChannels: configuration.imageHiddenSize,
            configuration: configuration,
            queryChunkSize: memory.attentionQueryChunkSize
        )
        self._thirdNorm.wrappedValue = LayerNorm(
            dimensions: configuration.tokenChannels,
            eps: configuration.transformerLayerNormEpsilon
        )
        self._feedForward.wrappedValue = TripoSRFeedForward(
            configuration: configuration,
            tokenChunkSize: memory.feedForwardTokenChunkSize
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, encoderHiddenStates: MLXArray) -> MLXArray {
        var hidden = input + selfAttention(firstNorm(input))
        hidden = hidden + crossAttention(secondNorm(hidden), encoderHiddenStates: encoderHiddenStates)
        return hidden + feedForward(thirdNorm(hidden))
    }
}

private final class TripoSRTransformer1D: Module {
    let configuration: TripoSRConfiguration

    @ModuleInfo(key: "norm") var norm: GroupNorm
    @ModuleInfo(key: "proj_in") var inputProjection: Linear
    @ModuleInfo(key: "transformer_blocks") var blocks: [TripoSRTransformerBlock]
    @ModuleInfo(key: "proj_out") var outputProjection: Linear

    init(configuration: TripoSRConfiguration, memory: TripoSRMemoryConfiguration) {
        self.configuration = configuration
        self._norm.wrappedValue = GroupNorm(
            groupCount: configuration.transformerGroupCount,
            dimensions: configuration.tokenChannels,
            eps: configuration.transformerGroupNormEpsilon,
            affine: true,
            pytorchCompatible: true
        )
        self._inputProjection.wrappedValue = Linear(
            configuration.tokenChannels,
            configuration.tokenChannels,
            bias: true
        )
        self._blocks.wrappedValue = (0..<configuration.transformerLayerCount).map { _ in
            TripoSRTransformerBlock(configuration: configuration, memory: memory)
        }
        self._outputProjection.wrappedValue = Linear(
            configuration.tokenChannels,
            configuration.tokenChannels,
            bias: true
        )
        super.init()
    }

    /// PyTorch layout in/out is `[batch, channel, token]`.
    func callAsFunction(_ input: MLXArray, encoderHiddenStates: MLXArray) -> MLXArray {
        let residual = input
        var hidden = input.transposed(0, 2, 1)
        hidden = inputProjection(norm(hidden))
        for block in blocks {
            hidden = block(hidden, encoderHiddenStates: encoderHiddenStates)
            // A full 3072-token self/cross-attention block is intentionally a
            // separate command-buffer boundary. Enqueuing all sixteen blocks
            // as one lazy graph exceeds the macOS Metal watchdog in debug and
            // creates unnecessary peak activation pressure in production.
            MLX.eval(hidden)
        }
        hidden = outputProjection(hidden).transposed(0, 2, 1)
        return hidden + residual
    }
}

private final class TripoSRTriplaneUpsampleNetwork: Module {
    let configuration: TripoSRConfiguration
    @ModuleInfo(key: "upsample") var upsample: ConvTransposed2d

    init(configuration: TripoSRConfiguration) {
        self.configuration = configuration
        self._upsample.wrappedValue = ConvTransposed2d(
            inputChannels: configuration.tokenChannels,
            outputChannels: configuration.scenePlaneChannels,
            kernelSize: 2,
            stride: 2,
            padding: 0,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        precondition(input.ndim == 5 && input.dim(1) == 3)
        let batch = input.dim(0)
        let planar = input
            .transposed(0, 1, 3, 4, 2)
            .reshaped(batch * 3, configuration.planeSize, configuration.planeSize, configuration.tokenChannels)
        let output = upsample(planar)
        return output.reshaped(
            batch,
            3,
            configuration.scenePlaneSize,
            configuration.scenePlaneSize,
            configuration.scenePlaneChannels
        )
    }
}

public final class TripoSRNeRFDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [UnaryLayer]

    public init(configuration: TripoSRConfiguration = .production) {
        var network: [UnaryLayer] = []
        network.reserveCapacity(configuration.decoderHiddenLayerCount * 2 + 1)
        network.append(Linear(configuration.decoderInputSize, configuration.decoderHiddenSize, bias: true))
        network.append(SiLU())
        if configuration.decoderHiddenLayerCount > 1 {
            for _ in 0..<(configuration.decoderHiddenLayerCount - 1) {
                network.append(Linear(configuration.decoderHiddenSize, configuration.decoderHiddenSize, bias: true))
                network.append(SiLU())
            }
        }
        network.append(Linear(configuration.decoderHiddenSize, 4, bias: true))
        self._layers.wrappedValue = network
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        layers.reduce(input) { hidden, layer in layer(hidden) }
    }
}

/// Native MLX implementation of the complete TripoSR inference graph.
public final class TripoSRModel: Module {
    public let configuration: TripoSRConfiguration
    public let memoryConfiguration: TripoSRMemoryConfiguration

    @ModuleInfo(key: "image_tokenizer") fileprivate var imageTokenizer: TripoSRImageTokenizer
    @ModuleInfo(key: "tokenizer") fileprivate var tokenizer: TripoSRTriplaneTokenizer
    @ModuleInfo(key: "backbone") fileprivate var backbone: TripoSRTransformer1D
    @ModuleInfo(key: "post_processor") fileprivate var postProcessor: TripoSRTriplaneUpsampleNetwork
    @ModuleInfo(key: "decoder") public var decoder: TripoSRNeRFDecoder

    public init(
        configuration: TripoSRConfiguration = .production,
        memoryConfiguration: TripoSRMemoryConfiguration = .appleSilicon
    ) {
        self.configuration = configuration
        self.memoryConfiguration = memoryConfiguration
        self._imageTokenizer.wrappedValue = TripoSRImageTokenizer(configuration: configuration)
        self._tokenizer.wrappedValue = TripoSRTriplaneTokenizer(configuration: configuration)
        self._backbone.wrappedValue = TripoSRTransformer1D(
            configuration: configuration,
            memory: memoryConfiguration
        )
        self._postProcessor.wrappedValue = TripoSRTriplaneUpsampleNetwork(configuration: configuration)
        self._decoder.wrappedValue = TripoSRNeRFDecoder(configuration: configuration)
        super.init()
    }

    /// Input is `[batch, height, width, 3]` RGB in `[0, 1]`.
    public func callAsFunction(_ input: MLXArray) -> TripoSRSceneCode {
        forwardDiagnostics(input).sceneCode
    }

    func forwardDiagnostics(_ input: MLXArray) -> TripoSRForwardDiagnostics {
        precondition(input.ndim == 4 && input.dim(3) == 3)
        precondition(input.dim(1) == configuration.conditioningImageSize)
        precondition(input.dim(2) == configuration.conditioningImageSize)
        let imageTokens = imageTokenizer(input)
        let initialTokens = tokenizer(batchSize: input.dim(0), dtype: input.dtype)
        let finalTokens = backbone(initialTokens, encoderHiddenStates: imageTokens)
        let sceneCode = TripoSRSceneCode(
            planes: postProcessor(tokenizer.detokenize(finalTokens))
        )
        return TripoSRForwardDiagnostics(
            imageTokens: imageTokens,
            initialTriplaneTokens: initialTokens,
            finalTriplaneTokens: finalTokens,
            sceneCode: sceneCode
        )
    }
}
