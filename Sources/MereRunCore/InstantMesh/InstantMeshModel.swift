import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

public struct InstantMeshSceneCode {
    /// `[batch, 3, height, width, channels]`.
    public let planes: MLXArray

    public init(planes: MLXArray) {
        precondition(planes.ndim == 5 && planes.dim(1) == 3)
        self.planes = planes
    }
}

struct InstantMeshForwardDiagnostics {
    /// `[batch, view * imageToken, imageChannel]`.
    let imageTokens: MLXArray
    /// `[batch, triplaneToken, transformerChannel]`.
    let initialTriplaneTokens: MLXArray
    let finalTriplaneTokens: MLXArray
    let sceneCode: InstantMeshSceneCode
}

private final class InstantMeshViTPatchEmbeddings: Module {
    @ModuleInfo(key: "projection") var projection: Conv2d

    init(configuration: InstantMeshConfiguration) {
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

private final class InstantMeshViTEmbeddings: Module {
    let configuration: InstantMeshConfiguration

    @ParameterInfo(key: "cls_token") var classToken: MLXArray
    @ParameterInfo(key: "position_embeddings") var positionEmbeddings: MLXArray
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: InstantMeshViTPatchEmbeddings

    init(configuration: InstantMeshConfiguration) {
        self.configuration = configuration
        self._classToken.wrappedValue = MLX.zeros([1, 1, configuration.imageHiddenSize], dtype: .float32)
        self._positionEmbeddings.wrappedValue = MLX.zeros(
            [1, 1 + configuration.imagePositionGridSize * configuration.imagePositionGridSize,
             configuration.imageHiddenSize],
            dtype: .float32
        )
        self._patchEmbeddings.wrappedValue = InstantMeshViTPatchEmbeddings(configuration: configuration)
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
        let tokens = MLX.concatenated([classTokens, patches], axis: 1)
        return tokens + interpolatedPositions(
            gridHeight: gridHeight,
            gridWidth: gridWidth,
            dtype: tokens.dtype
        )
    }

    private func interpolatedPositions(gridHeight: Int, gridWidth: Int, dtype: DType) -> MLXArray {
        let sourceSize = configuration.imagePositionGridSize
        let classPosition = positionEmbeddings[0..., 0..<1, 0...].asType(dtype)
        let patchPositions = positionEmbeddings[0..., 1..., 0...]
            .reshaped(1, sourceSize, sourceSize, configuration.imageHiddenSize)
        let resized: MLXArray
        if gridHeight == sourceSize, gridWidth == sourceSize {
            resized = patchPositions.asType(dtype)
        } else {
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

private final class InstantMeshViTSelfAttention: Module {
    let headCount: Int
    let headDimension: Int
    let scale: Float

    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear

    init(configuration: InstantMeshConfiguration) {
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
        let count = input.dim(1)
        let queries = query(input).reshaped(batch, count, headCount, headDimension).transposed(0, 2, 1, 3)
        let keys = key(input).reshaped(batch, count, headCount, headDimension).transposed(0, 2, 1, 3)
        let values = value(input).reshaped(batch, count, headCount, headDimension).transposed(0, 2, 1, 3)
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .none
        ).transposed(0, 2, 1, 3).reshaped(batch, count, headCount * headDimension)
    }
}

private final class InstantMeshViTSelfOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear

    init(configuration: InstantMeshConfiguration) {
        self._dense.wrappedValue = Linear(configuration.imageHiddenSize, configuration.imageHiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { dense(input) }
}

private final class InstantMeshViTAttention: Module {
    @ModuleInfo(key: "attention") var attention: InstantMeshViTSelfAttention
    @ModuleInfo(key: "output") var output: InstantMeshViTSelfOutput

    init(configuration: InstantMeshConfiguration) {
        self._attention.wrappedValue = InstantMeshViTSelfAttention(configuration: configuration)
        self._output.wrappedValue = InstantMeshViTSelfOutput(configuration: configuration)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { output(attention(input)) }
}

private final class InstantMeshViTIntermediate: Module {
    @ModuleInfo(key: "dense") var dense: Linear

    init(configuration: InstantMeshConfiguration) {
        self._dense.wrappedValue = Linear(
            configuration.imageHiddenSize,
            configuration.imageIntermediateSize,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { gelu(dense(input)) }
}

private final class InstantMeshViTOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear

    init(configuration: InstantMeshConfiguration) {
        self._dense.wrappedValue = Linear(
            configuration.imageIntermediateSize,
            configuration.imageHiddenSize,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { dense(input) }
}

private final class InstantMeshViTLayer: Module {
    let hiddenSize: Int

    @ModuleInfo(key: "attention") var attention: InstantMeshViTAttention
    @ModuleInfo(key: "intermediate") var intermediate: InstantMeshViTIntermediate
    @ModuleInfo(key: "output") var output: InstantMeshViTOutput
    @ModuleInfo(key: "layernorm_before") var layerNormBefore: LayerNorm
    @ModuleInfo(key: "layernorm_after") var layerNormAfter: LayerNorm
    @ModuleInfo(key: "adaLN_modulation") var adaptiveLayerNorm: [UnaryLayer]

    init(configuration: InstantMeshConfiguration) {
        self.hiddenSize = configuration.imageHiddenSize
        self._attention.wrappedValue = InstantMeshViTAttention(configuration: configuration)
        self._intermediate.wrappedValue = InstantMeshViTIntermediate(configuration: configuration)
        self._output.wrappedValue = InstantMeshViTOutput(configuration: configuration)
        self._layerNormBefore.wrappedValue = LayerNorm(
            dimensions: configuration.imageHiddenSize,
            eps: configuration.imageLayerNormEpsilon
        )
        self._layerNormAfter.wrappedValue = LayerNorm(
            dimensions: configuration.imageHiddenSize,
            eps: configuration.imageLayerNormEpsilon
        )
        self._adaptiveLayerNorm.wrappedValue = [
            SiLU(),
            Linear(configuration.imageHiddenSize, 4 * configuration.imageHiddenSize, bias: true),
        ]
        super.init()
    }

    func callAsFunction(_ input: MLXArray, conditioning: MLXArray) -> MLXArray {
        let modulation = adaptiveLayerNorm.reduce(conditioning) { value, layer in layer(value) }
        let shiftAttention = modulation[0..., 0..<hiddenSize]
        let scaleAttention = modulation[0..., hiddenSize..<(2 * hiddenSize)]
        let shiftMLP = modulation[0..., (2 * hiddenSize)..<(3 * hiddenSize)]
        let scaleMLP = modulation[0..., (3 * hiddenSize)...]

        let normalizedAttention = modulate(
            layerNormBefore(input),
            shift: shiftAttention,
            scale: scaleAttention
        )
        let afterAttention = input + attention(normalizedAttention)
        let normalizedMLP = modulate(
            layerNormAfter(afterAttention),
            shift: shiftMLP,
            scale: scaleMLP
        )
        return afterAttention + output(intermediate(normalizedMLP))
    }

    private func modulate(_ input: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
        input * (1 + scale.expandedDimensions(axis: 1)) + shift.expandedDimensions(axis: 1)
    }
}

private final class InstantMeshViTEncoder: Module {
    @ModuleInfo(key: "layer") var layers: [InstantMeshViTLayer]

    init(configuration: InstantMeshConfiguration) {
        self._layers.wrappedValue = (0..<configuration.imageLayerCount).map { _ in
            InstantMeshViTLayer(configuration: configuration)
        }
        super.init()
    }

    func callAsFunction(_ input: MLXArray, conditioning: MLXArray) -> MLXArray {
        var hidden = input
        for layer in layers {
            hidden = layer(hidden, conditioning: conditioning)
            MLX.eval(hidden)
        }
        return hidden
    }
}

private final class InstantMeshViTModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: InstantMeshViTEmbeddings
    @ModuleInfo(key: "encoder") var encoder: InstantMeshViTEncoder
    @ModuleInfo(key: "layernorm") var layerNorm: LayerNorm

    init(configuration: InstantMeshConfiguration) {
        self._embeddings.wrappedValue = InstantMeshViTEmbeddings(configuration: configuration)
        self._encoder.wrappedValue = InstantMeshViTEncoder(configuration: configuration)
        self._layerNorm.wrappedValue = LayerNorm(
            dimensions: configuration.imageHiddenSize,
            eps: configuration.imageLayerNormEpsilon
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, conditioning: MLXArray) -> MLXArray {
        layerNorm(encoder(embeddings(input), conditioning: conditioning))
    }
}

private final class InstantMeshImageEncoder: Module {
    let configuration: InstantMeshConfiguration
    let viewBatchSize: Int

    @ModuleInfo(key: "model") var model: InstantMeshViTModel
    @ModuleInfo(key: "camera_embedder") var cameraEmbedder: [UnaryLayer]

    init(configuration: InstantMeshConfiguration, viewBatchSize: Int) {
        self.configuration = configuration
        self.viewBatchSize = viewBatchSize
        self._model.wrappedValue = InstantMeshViTModel(configuration: configuration)
        self._cameraEmbedder.wrappedValue = [
            Linear(configuration.cameraDimension, configuration.imageHiddenSize, bias: true),
            SiLU(),
            Linear(configuration.imageHiddenSize, configuration.imageHiddenSize, bias: true),
        ]
        super.init()
    }

    /// Images are flattened-view NHWC RGB in `[0, 1]`; cameras are `[view, 16]`.
    func callAsFunction(_ input: MLXArray, cameras: MLXArray) -> MLXArray {
        precondition(input.ndim == 4 && input.dim(3) == 3)
        precondition(cameras.shape == [input.dim(0), configuration.cameraDimension])
        let mean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped(1, 1, 1, 3)
        let deviation = MLXArray([Float(0.229), 0.224, 0.225]).reshaped(1, 1, 1, 3)
        let normalized = (input - mean.asType(input.dtype)) / deviation.asType(input.dtype)
        var chunks: [MLXArray] = []
        chunks.reserveCapacity((input.dim(0) + viewBatchSize - 1) / viewBatchSize)
        for start in stride(from: 0, to: input.dim(0), by: viewBatchSize) {
            let end = min(start + viewBatchSize, input.dim(0))
            let camera = cameraEmbedder.reduce(cameras[start..<end, 0...]) { value, layer in layer(value) }
            let output = model(normalized[start..<end, 0..., 0..., 0...], conditioning: camera)
            MLX.eval(output)
            chunks.append(output)
        }
        return MLX.concatenated(chunks, axis: 0)
    }
}

private final class InstantMeshCrossAttention: Module {
    let headCount: Int
    let headDimension: Int
    let scale: Float
    let queryChunkSize: Int

    @ParameterInfo(key: "q_proj_weight") var queryWeight: MLXArray
    @ParameterInfo(key: "k_proj_weight") var keyWeight: MLXArray
    @ParameterInfo(key: "v_proj_weight") var valueWeight: MLXArray
    @ModuleInfo(key: "out_proj") var output: Linear

    init(configuration: InstantMeshConfiguration, queryChunkSize: Int) {
        self.headCount = configuration.transformerHeadCount
        self.headDimension = configuration.transformerDimension / configuration.transformerHeadCount
        self.scale = 1 / sqrt(Float(headDimension))
        self.queryChunkSize = queryChunkSize
        self._queryWeight.wrappedValue = MLX.zeros(
            [configuration.transformerDimension, configuration.transformerDimension], dtype: .float32
        )
        self._keyWeight.wrappedValue = MLX.zeros(
            [configuration.transformerDimension, configuration.imageHiddenSize], dtype: .float32
        )
        self._valueWeight.wrappedValue = MLX.zeros(
            [configuration.transformerDimension, configuration.imageHiddenSize], dtype: .float32
        )
        self._output.wrappedValue = Linear(
            configuration.transformerDimension,
            configuration.transformerDimension,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, context: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let queryCount = input.dim(1)
        let contextCount = context.dim(1)
        let queries = MLX.matmul(input, queryWeight.T)
            .reshaped(batch, queryCount, headCount, headDimension).transposed(0, 2, 1, 3)
        let keys = MLX.matmul(context, keyWeight.T)
            .reshaped(batch, contextCount, headCount, headDimension).transposed(0, 2, 1, 3)
        let values = MLX.matmul(context, valueWeight.T)
            .reshaped(batch, contextCount, headCount, headDimension).transposed(0, 2, 1, 3)
        return output(attend(queries: queries, keys: keys, values: values))
    }

    private func attend(queries: MLXArray, keys: MLXArray, values: MLXArray) -> MLXArray {
        let batch = queries.dim(0)
        let count = queries.dim(2)
        var chunks: [MLXArray] = []
        chunks.reserveCapacity((count + queryChunkSize - 1) / queryChunkSize)
        for start in stride(from: 0, to: count, by: queryChunkSize) {
            let end = min(start + queryChunkSize, count)
            let value = MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., start..<end, 0...],
                keys: keys,
                values: values,
                scale: scale,
                mask: .none
            )
            MLX.eval(value)
            chunks.append(value)
        }
        return MLX.concatenated(chunks, axis: 2).transposed(0, 2, 1, 3)
            .reshaped(batch, count, headCount * headDimension)
    }
}

private final class InstantMeshSelfAttention: Module {
    let headCount: Int
    let headDimension: Int
    let scale: Float
    let modelDimension: Int
    let queryChunkSize: Int

    @ParameterInfo(key: "in_proj_weight") var inputProjectionWeight: MLXArray
    @ModuleInfo(key: "out_proj") var output: Linear

    init(configuration: InstantMeshConfiguration, queryChunkSize: Int) {
        self.headCount = configuration.transformerHeadCount
        self.headDimension = configuration.transformerDimension / configuration.transformerHeadCount
        self.scale = 1 / sqrt(Float(headDimension))
        self.modelDimension = configuration.transformerDimension
        self.queryChunkSize = queryChunkSize
        self._inputProjectionWeight.wrappedValue = MLX.zeros(
            [3 * configuration.transformerDimension, configuration.transformerDimension], dtype: .float32
        )
        self._output.wrappedValue = Linear(
            configuration.transformerDimension,
            configuration.transformerDimension,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let count = input.dim(1)
        let projected = MLX.matmul(input, inputProjectionWeight.T)
        let queries = projected[0..., 0..., 0..<modelDimension]
            .reshaped(batch, count, headCount, headDimension).transposed(0, 2, 1, 3)
        let keys = projected[0..., 0..., modelDimension..<(2 * modelDimension)]
            .reshaped(batch, count, headCount, headDimension).transposed(0, 2, 1, 3)
        let values = projected[0..., 0..., (2 * modelDimension)...]
            .reshaped(batch, count, headCount, headDimension).transposed(0, 2, 1, 3)
        var chunks: [MLXArray] = []
        chunks.reserveCapacity((count + queryChunkSize - 1) / queryChunkSize)
        for start in stride(from: 0, to: count, by: queryChunkSize) {
            let end = min(start + queryChunkSize, count)
            let value = MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., start..<end, 0...],
                keys: keys,
                values: values,
                scale: scale,
                mask: .none
            )
            MLX.eval(value)
            chunks.append(value)
        }
        let attended = MLX.concatenated(chunks, axis: 2).transposed(0, 2, 1, 3)
            .reshaped(batch, count, modelDimension)
        return output(attended)
    }
}

private final class InstantMeshGELU: Module, UnaryLayer {
    func callAsFunction(_ input: MLXArray) -> MLXArray { gelu(input) }
}

private final class InstantMeshTransformerBlock: Module {
    let feedForwardChunkSize: Int

    @ModuleInfo(key: "norm1") var firstNorm: LayerNorm
    @ModuleInfo(key: "cross_attn") var crossAttention: InstantMeshCrossAttention
    @ModuleInfo(key: "norm2") var secondNorm: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttention: InstantMeshSelfAttention
    @ModuleInfo(key: "norm3") var thirdNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: [UnaryLayer]

    init(configuration: InstantMeshConfiguration, memory: InstantMeshMemoryConfiguration) {
        self.feedForwardChunkSize = memory.feedForwardTokenChunkSize
        self._firstNorm.wrappedValue = LayerNorm(
            dimensions: configuration.transformerDimension,
            eps: configuration.transformerBlockLayerNormEpsilon
        )
        self._crossAttention.wrappedValue = InstantMeshCrossAttention(
            configuration: configuration,
            queryChunkSize: memory.attentionQueryChunkSize
        )
        self._secondNorm.wrappedValue = LayerNorm(
            dimensions: configuration.transformerDimension,
            eps: configuration.transformerBlockLayerNormEpsilon
        )
        self._selfAttention.wrappedValue = InstantMeshSelfAttention(
            configuration: configuration,
            queryChunkSize: memory.attentionQueryChunkSize
        )
        self._thirdNorm.wrappedValue = LayerNorm(
            dimensions: configuration.transformerDimension,
            eps: configuration.transformerBlockLayerNormEpsilon
        )
        self._mlp.wrappedValue = [
            Linear(
                configuration.transformerDimension,
                configuration.transformerDimension * configuration.transformerMLPMultiplier,
                bias: true
            ),
            InstantMeshGELU(),
            Identity(),
            Linear(
                configuration.transformerDimension * configuration.transformerMLPMultiplier,
                configuration.transformerDimension,
                bias: true
            ),
            Identity(),
        ]
        super.init()
    }

    func callAsFunction(_ input: MLXArray, context: MLXArray) -> MLXArray {
        var hidden = input + crossAttention(firstNorm(input), context: context)
        hidden = hidden + selfAttention(secondNorm(hidden))
        return hidden + feedForward(thirdNorm(hidden))
    }

    private func feedForward(_ input: MLXArray) -> MLXArray {
        if input.dim(1) <= feedForwardChunkSize {
            return mlp.reduce(input) { value, layer in layer(value) }
        }
        var chunks: [MLXArray] = []
        chunks.reserveCapacity((input.dim(1) + feedForwardChunkSize - 1) / feedForwardChunkSize)
        for start in stride(from: 0, to: input.dim(1), by: feedForwardChunkSize) {
            let end = min(start + feedForwardChunkSize, input.dim(1))
            let value = mlp.reduce(input[0..., start..<end, 0...]) { current, layer in layer(current) }
            MLX.eval(value)
            chunks.append(value)
        }
        return MLX.concatenated(chunks, axis: 1)
    }
}

private final class InstantMeshTriplaneTransformer: Module {
    let configuration: InstantMeshConfiguration

    @ParameterInfo(key: "pos_embed") var positionalEmbedding: MLXArray
    @ModuleInfo(key: "layers") var layers: [InstantMeshTransformerBlock]
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "deconv") var deconvolution: ConvTransposed2d

    init(configuration: InstantMeshConfiguration, memory: InstantMeshMemoryConfiguration) {
        self.configuration = configuration
        self._positionalEmbedding.wrappedValue = MLX.zeros(
            [1, configuration.triplaneTokenCount, configuration.transformerDimension], dtype: .float32
        )
        self._layers.wrappedValue = (0..<configuration.transformerLayerCount).map { _ in
            InstantMeshTransformerBlock(configuration: configuration, memory: memory)
        }
        self._norm.wrappedValue = LayerNorm(
            dimensions: configuration.transformerDimension,
            eps: configuration.transformerFinalLayerNormEpsilon
        )
        self._deconvolution.wrappedValue = ConvTransposed2d(
            inputChannels: configuration.transformerDimension,
            outputChannels: configuration.triplaneChannels,
            kernelSize: 2,
            stride: 2,
            padding: 0,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ imageFeatures: MLXArray) -> (initial: MLXArray, final: MLXArray, planes: MLXArray) {
        let batch = imageFeatures.dim(0)
        let initial = MLX.broadcast(
            positionalEmbedding.asType(imageFeatures.dtype),
            to: [batch, configuration.triplaneTokenCount, configuration.transformerDimension]
        )
        var hidden = initial
        for layer in layers {
            hidden = layer(hidden, context: imageFeatures)
            MLX.eval(hidden)
        }
        let final = norm(hidden)
        let planar = final
            .reshaped(
                batch, 3, configuration.triplaneLowResolution,
                configuration.triplaneLowResolution, configuration.transformerDimension
            )
            .reshaped(
                batch * 3, configuration.triplaneLowResolution,
                configuration.triplaneLowResolution, configuration.transformerDimension
            )
        let upsampled = deconvolution(planar).reshaped(
            batch, 3, configuration.triplaneHighResolution,
            configuration.triplaneHighResolution, configuration.triplaneChannels
        )
        return (initial, final, upsampled)
    }
}

public final class InstantMeshDecoder: Module {
    @ModuleInfo(key: "net_sdf") var signedDistanceNetwork: [UnaryLayer]
    @ModuleInfo(key: "net_rgb") var colorNetwork: [UnaryLayer]
    @ModuleInfo(key: "net_deformation") var deformationNetwork: [UnaryLayer]
    @ModuleInfo(key: "net_weight") var cellWeightNetwork: [UnaryLayer]

    public init(configuration: InstantMeshConfiguration = .production) {
        self._signedDistanceNetwork.wrappedValue = Self.network(
            input: configuration.decoderInputSize,
            hidden: configuration.decoderHiddenSize,
            hiddenLayerCount: configuration.decoderHiddenLayerCount,
            output: 1
        )
        self._colorNetwork.wrappedValue = Self.network(
            input: configuration.decoderInputSize,
            hidden: configuration.decoderHiddenSize,
            hiddenLayerCount: configuration.decoderHiddenLayerCount,
            output: 3
        )
        self._deformationNetwork.wrappedValue = Self.network(
            input: configuration.decoderInputSize,
            hidden: configuration.decoderHiddenSize,
            hiddenLayerCount: configuration.decoderHiddenLayerCount,
            output: 3
        )
        self._cellWeightNetwork.wrappedValue = Self.network(
            input: 8 * configuration.decoderInputSize,
            hidden: configuration.decoderHiddenSize,
            hiddenLayerCount: configuration.decoderHiddenLayerCount,
            output: 21
        )
        super.init()
    }

    public func signedDistance(_ features: MLXArray) -> MLXArray {
        signedDistanceNetwork.reduce(features) { value, layer in layer(value) }
    }

    public func color(_ features: MLXArray) -> MLXArray {
        let raw = colorNetwork.reduce(features) { value, layer in layer(value) }
        return MLX.sigmoid(raw) * 1.002 - 0.001
    }

    public func deformation(_ features: MLXArray) -> MLXArray {
        deformationNetwork.reduce(features) { value, layer in layer(value) }
    }

    public func cellWeights(_ cornerFeatures: MLXArray) -> MLXArray {
        precondition(cornerFeatures.ndim == 3 && cornerFeatures.dim(1) == 8)
        let flattened = cornerFeatures.reshaped(cornerFeatures.dim(0), -1)
        return cellWeightNetwork.reduce(flattened) { value, layer in layer(value) } * 0.1
    }

    private static func network(
        input: Int,
        hidden: Int,
        hiddenLayerCount: Int,
        output: Int
    ) -> [UnaryLayer] {
        var result: [UnaryLayer] = []
        result.reserveCapacity(hiddenLayerCount * 2 + 1)
        result.append(Linear(input, hidden, bias: true))
        result.append(ReLU())
        if hiddenLayerCount > 1 {
            for _ in 1..<hiddenLayerCount {
                result.append(Linear(hidden, hidden, bias: true))
                result.append(ReLU())
            }
        }
        result.append(Linear(hidden, output, bias: true))
        return result
    }
}

private final class InstantMeshSynthesizer: Module {
    @ModuleInfo(key: "decoder") var decoder: InstantMeshDecoder

    init(configuration: InstantMeshConfiguration) {
        self._decoder.wrappedValue = InstantMeshDecoder(configuration: configuration)
        super.init()
    }
}

/// Native MLX implementation of the reconstruction-only InstantMesh Base graph.
/// It accepts user-supplied 4/6-view images and never invokes a view generator.
public final class InstantMeshModel: Module {
    public let configuration: InstantMeshConfiguration
    public let memoryConfiguration: InstantMeshMemoryConfiguration

    @ModuleInfo(key: "encoder") fileprivate var encoder: InstantMeshImageEncoder
    @ModuleInfo(key: "transformer") fileprivate var transformer: InstantMeshTriplaneTransformer
    @ModuleInfo(key: "synthesizer") fileprivate var synthesizer: InstantMeshSynthesizer

    public var decoder: InstantMeshDecoder { synthesizer.decoder }

    public init(
        configuration: InstantMeshConfiguration = .production,
        memoryConfiguration: InstantMeshMemoryConfiguration = .appleSilicon
    ) {
        self.configuration = configuration
        self.memoryConfiguration = memoryConfiguration
        self._encoder.wrappedValue = InstantMeshImageEncoder(
            configuration: configuration,
            viewBatchSize: memoryConfiguration.imageViewBatchSize
        )
        self._transformer.wrappedValue = InstantMeshTriplaneTransformer(
            configuration: configuration,
            memory: memoryConfiguration
        )
        self._synthesizer.wrappedValue = InstantMeshSynthesizer(configuration: configuration)
        super.init()
    }

    /// Images are `[batch, view, height, width, 3]` RGB in `[0, 1]`.
    /// Cameras are `[batch, view, 16]` in the official C2W+intrinsics layout.
    public func callAsFunction(images: MLXArray, cameras: MLXArray) -> InstantMeshSceneCode {
        forwardDiagnostics(images: images, cameras: cameras).sceneCode
    }

    func forwardDiagnostics(images: MLXArray, cameras: MLXArray) -> InstantMeshForwardDiagnostics {
        precondition(images.ndim == 5 && images.dim(4) == 3)
        precondition(images.dim(1) == 4 || images.dim(1) == 6)
        precondition(images.dim(2) == configuration.conditioningImageSize)
        precondition(images.dim(3) == configuration.conditioningImageSize)
        precondition(cameras.shape == [images.dim(0), images.dim(1), configuration.cameraDimension])
        let batch = images.dim(0)
        let views = images.dim(1)
        let flattenedImages = images.reshaped(
            batch * views,
            configuration.conditioningImageSize,
            configuration.conditioningImageSize,
            3
        )
        let flattenedCameras = cameras.reshaped(batch * views, configuration.cameraDimension)
        let perViewTokens = encoder(flattenedImages, cameras: flattenedCameras)
        let imageTokens = perViewTokens.reshaped(
            batch,
            views * perViewTokens.dim(1),
            configuration.imageHiddenSize
        )
        let result = transformer(imageTokens)
        return InstantMeshForwardDiagnostics(
            imageTokens: imageTokens,
            initialTriplaneTokens: result.initial,
            finalTriplaneTokens: result.final,
            sceneCode: InstantMeshSceneCode(planes: result.planes)
        )
    }
}
