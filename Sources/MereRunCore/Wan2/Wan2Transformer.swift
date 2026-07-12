import Foundation
import MLX
import MLXFast
import MLXNN

public struct Wan2TransformerConfiguration: Hashable, Sendable {
    public let patchSize: [Int]
    public let textLength: Int
    public let inputChannels: Int
    public let hiddenSize: Int
    public let feedForwardSize: Int
    public let timestepFrequencySize: Int
    public let textEmbeddingSize: Int
    public let outputChannels: Int
    public let headCount: Int
    public let layerCount: Int
    public let epsilon: Float
    public let projectiveCameraConditioning: Bool
    public let projectiveCameraAttentionCompression: Int

    public init(
        patchSize: [Int] = [1, 2, 2],
        textLength: Int = 512,
        inputChannels: Int = 48,
        hiddenSize: Int = 3_072,
        feedForwardSize: Int = 14_336,
        timestepFrequencySize: Int = 256,
        textEmbeddingSize: Int = 4_096,
        outputChannels: Int = 48,
        headCount: Int = 24,
        layerCount: Int = 30,
        epsilon: Float = 1e-6,
        projectiveCameraConditioning: Bool = false,
        projectiveCameraAttentionCompression: Int = 1
    ) {
        precondition(patchSize.count == 3)
        precondition(hiddenSize % headCount == 0)
        precondition((hiddenSize / headCount) % 2 == 0)
        precondition(projectiveCameraAttentionCompression > 0)
        precondition(hiddenSize.isMultiple(of: projectiveCameraAttentionCompression))
        precondition(headCount.isMultiple(of: projectiveCameraAttentionCompression))
        self.patchSize = patchSize
        self.textLength = textLength
        self.inputChannels = inputChannels
        self.hiddenSize = hiddenSize
        self.feedForwardSize = feedForwardSize
        self.timestepFrequencySize = timestepFrequencySize
        self.textEmbeddingSize = textEmbeddingSize
        self.outputChannels = outputChannels
        self.headCount = headCount
        self.layerCount = layerCount
        self.epsilon = epsilon
        self.projectiveCameraConditioning = projectiveCameraConditioning
        self.projectiveCameraAttentionCompression = projectiveCameraAttentionCompression
    }
}

public struct Wan2GridSize: Hashable, Sendable {
    public let frames: Int
    public let height: Int
    public let width: Int

    public init(frames: Int, height: Int, width: Int) {
        self.frames = frames
        self.height = height
        self.width = width
    }

    public var sequenceLength: Int { frames * height * width }
}

enum Wan2PatchLayout {
    static func flatten(_ latent: MLXArray, patchSize: [Int]) -> (value: MLXArray, grid: Wan2GridSize) {
        precondition(latent.ndim == 4)
        precondition(patchSize.count == 3)
        let channels = latent.dim(0)
        let temporalPatch = patchSize[0]
        let heightPatch = patchSize[1]
        let widthPatch = patchSize[2]
        let grid = Wan2GridSize(
            frames: latent.dim(1) / temporalPatch,
            height: latent.dim(2) / heightPatch,
            width: latent.dim(3) / widthPatch
        )
        let flattened = latent
            .reshaped(
                channels,
                grid.frames, temporalPatch,
                grid.height, heightPatch,
                grid.width, widthPatch
            )
            .transposed(1, 3, 5, 0, 2, 4, 6)
            .reshaped(grid.sequenceLength, -1)
        return (flattened, grid)
    }
}

public struct Wan2AttentionKVCache {
    public let key: MLXArray
    public let value: MLXArray
}

final class Wan2RMSNorm: Module {
    let epsilon: Float
    @ModuleInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, epsilon: Float) {
        self.epsilon = epsilon
        self._weight.wrappedValue = MLX.ones([dimensions])
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(input, weight: weight, eps: epsilon)
    }
}

final class Wan2LayerNorm: Module {
    let epsilon: Float
    let weight: MLXArray?
    let bias: MLXArray?

    init(dimensions: Int, epsilon: Float, affine: Bool = false) {
        self.epsilon = epsilon
        self.weight = affine ? MLX.ones([dimensions]) : nil
        self.bias = affine ? MLX.zeros([dimensions]) : nil
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        MLXFast.layerNorm(
            input,
            weight: weight,
            bias: bias,
            eps: epsilon
        )
    }
}

enum Wan2RoPE {
    struct Cache {
        let cosine: MLXArray
        let sine: MLXArray
    }

    static func frequencies(maxSequence: Int, dimensions: [Int]) -> MLXArray {
        var tables: [MLXArray] = []
        tables.reserveCapacity(dimensions.count)
        for dimension in dimensions {
            precondition(dimension % 2 == 0)
            var values: [Float] = []
            values.reserveCapacity(maxSequence * dimension)
            for position in 0..<maxSequence {
                for index in stride(from: 0, to: dimension, by: 2) {
                    let angle = Float(position) / pow(10_000, Float(index) / Float(dimension))
                    values.append(cos(angle))
                    values.append(sin(angle))
                }
            }
            tables.append(MLXArray(values).reshaped(maxSequence, dimension / 2, 2))
        }
        return MLX.concatenated(tables, axis: 1)
    }

    static func prepare(
        grid: Wan2GridSize,
        frequencies: MLXArray,
        dtype: DType,
        temporalFrameIndices: [Int]? = nil
    ) -> Cache {
        let halfDimension = frequencies.dim(1)
        let temporalDimension = halfDimension - 2 * (halfDimension / 3)
        let heightDimension = halfDimension / 3
        let widthDimension = halfDimension / 3
        let typed = frequencies.asType(.float32)
        let temporalRows: MLXArray
        if let temporalFrameIndices {
            precondition(temporalFrameIndices.count == grid.frames)
            temporalRows = MLX.take(
                typed,
                MLXArray(temporalFrameIndices.map(Int32.init)),
                axis: 0
            )[0..., 0..<temporalDimension]
        } else {
            temporalRows = typed[0..<grid.frames, 0..<temporalDimension]
        }
        let temporal = MLX.broadcast(
            temporalRows
                .reshaped(grid.frames, 1, 1, temporalDimension, 2),
            to: [grid.frames, grid.height, grid.width, temporalDimension, 2]
        )
        let vertical = MLX.broadcast(
            typed[0..<grid.height, temporalDimension..<(temporalDimension + heightDimension)]
                .reshaped(1, grid.height, 1, heightDimension, 2),
            to: [grid.frames, grid.height, grid.width, heightDimension, 2]
        )
        let horizontalSource = typed[
            0..<grid.width,
            (temporalDimension + heightDimension)..<(temporalDimension + heightDimension + widthDimension)
        ]
        .reshaped(1, 1, grid.width, widthDimension, 2)
        let horizontal = MLX.broadcast(
            horizontalSource,
            to: [grid.frames, grid.height, grid.width, widthDimension, 2]
        )
        let combined = MLX.concatenated([temporal, vertical, horizontal], axis: 3)
            .reshaped(grid.sequenceLength, 1, halfDimension, 2)
        return Cache(
            cosine: combined[0..., 0..., 0..., 0],
            sine: combined[0..., 0..., 0..., 1]
        )
    }

    static func apply(_ input: MLXArray, grid: Wan2GridSize, cache: Cache) -> MLXArray {
        let batch = input.dim(0)
        let sequence = grid.sequenceLength
        let heads = input.dim(2)
        let dimension = input.dim(3)
        let paired = input[0..., 0..<sequence].reshaped(batch, sequence, heads, dimension / 2, 2)
        let real = paired[0..., 0..., 0..., 0..., 0]
        let imaginary = paired[0..., 0..., 0..., 0..., 1]
        let cosine = cache.cosine.expandedDimensions(axis: 0)
        let sine = cache.sine.expandedDimensions(axis: 0)
        let rotated = MLX.stacked(
            [real * cosine - imaginary * sine, real * sine + imaginary * cosine],
            axis: -1
        ).reshaped(batch, sequence, heads, dimension)
        guard sequence < input.dim(1) else { return rotated }
        return MLX.concatenated([rotated, input[0..., sequence...]], axis: 1)
    }
}

final class Wan2SelfAttention: Module {
    let heads: Int
    let headDimension: Int
    let scale: Float

    @ModuleInfo(key: "q") var query: Linear
    @ModuleInfo(key: "k") var key: Linear
    @ModuleInfo(key: "v") var value: Linear
    @ModuleInfo(key: "o") var output: Linear
    @ModuleInfo(key: "norm_q") var queryNorm: Wan2RMSNorm
    @ModuleInfo(key: "norm_k") var keyNorm: Wan2RMSNorm

    init(dimensions: Int, heads: Int, epsilon: Float) {
        self.heads = heads
        self.headDimension = dimensions / heads
        self.scale = 1 / Float(headDimension).squareRoot()
        self._query.wrappedValue = Linear(dimensions, dimensions, bias: true)
        self._key.wrappedValue = Linear(dimensions, dimensions, bias: true)
        self._value.wrappedValue = Linear(dimensions, dimensions, bias: true)
        self._output.wrappedValue = Linear(dimensions, dimensions, bias: true)
        self._queryNorm.wrappedValue = Wan2RMSNorm(dimensions: dimensions, epsilon: epsilon)
        self._keyNorm.wrappedValue = Wan2RMSNorm(dimensions: dimensions, epsilon: epsilon)
    }

    func callAsFunction(
        _ input: MLXArray,
        grid: Wan2GridSize,
        rope: Wan2RoPE.Cache,
        mask: MLXArray?
    ) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let dtype = query.weight.dtype
        let typed = input.asType(dtype)
        var q = queryNorm(query(typed)).reshaped(batch, sequence, heads, headDimension)
        var k = keyNorm(key(typed)).reshaped(batch, sequence, heads, headDimension)
        let v = value(typed).reshaped(batch, sequence, heads, headDimension).transposed(0, 2, 1, 3)
        q = Wan2RoPE.apply(q.asType(.float32), grid: grid, cache: rope).transposed(0, 2, 1, 3)
        k = Wan2RoPE.apply(k.asType(.float32), grid: grid, cache: rope).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask.map { .array($0) } ?? .none
        )
        let merged = attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, heads * headDimension)
        return output(merged)
    }
}

final class Wan2CrossAttention: Module {
    let heads: Int
    let headDimension: Int
    let scale: Float

    @ModuleInfo(key: "q") var query: Linear
    @ModuleInfo(key: "k") var key: Linear
    @ModuleInfo(key: "v") var value: Linear
    @ModuleInfo(key: "o") var output: Linear
    @ModuleInfo(key: "norm_q") var queryNorm: Wan2RMSNorm
    @ModuleInfo(key: "norm_k") var keyNorm: Wan2RMSNorm

    init(dimensions: Int, heads: Int, epsilon: Float) {
        self.heads = heads
        self.headDimension = dimensions / heads
        self.scale = 1 / Float(headDimension).squareRoot()
        self._query.wrappedValue = Linear(dimensions, dimensions, bias: true)
        self._key.wrappedValue = Linear(dimensions, dimensions, bias: true)
        self._value.wrappedValue = Linear(dimensions, dimensions, bias: true)
        self._output.wrappedValue = Linear(dimensions, dimensions, bias: true)
        self._queryNorm.wrappedValue = Wan2RMSNorm(dimensions: dimensions, epsilon: epsilon)
        self._keyNorm.wrappedValue = Wan2RMSNorm(dimensions: dimensions, epsilon: epsilon)
    }

    func prepareCache(context: MLXArray) -> Wan2AttentionKVCache {
        let batch = context.dim(0)
        let typed = context.asType(key.weight.dtype)
        let k = keyNorm(key(typed)).reshaped(batch, -1, heads, headDimension).transposed(0, 2, 1, 3)
        let v = value(typed).reshaped(batch, -1, heads, headDimension).transposed(0, 2, 1, 3)
        return Wan2AttentionKVCache(key: k, value: v)
    }

    func callAsFunction(_ input: MLXArray, context: MLXArray, cache: Wan2AttentionKVCache?) -> MLXArray {
        let batch = input.dim(0)
        let dtype = query.weight.dtype
        let q = queryNorm(query(input.asType(dtype)))
            .reshaped(batch, -1, heads, headDimension)
            .transposed(0, 2, 1, 3)
        let prepared = cache ?? prepareCache(context: context)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: prepared.key,
            values: prepared.value,
            scale: scale,
            mask: .none
        )
        let merged = attended.transposed(0, 2, 1, 3).reshaped(batch, -1, heads * headDimension)
        return output(merged)
    }
}

final class Wan2FeedForward: Module {
    @ModuleInfo(key: "fc1") var input: Linear
    @ModuleInfo(key: "fc2") var output: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._input.wrappedValue = Linear(dimensions, hiddenDimensions, bias: true)
        self._output.wrappedValue = Linear(hiddenDimensions, dimensions, bias: true)
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        output(MLXNN.geluApproximate(input(value.asType(input.weight.dtype))))
    }
}

final class Wan2TransformerBlock: Module {
    @ModuleInfo(key: "norm1") var selfNorm: Wan2LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttention: Wan2SelfAttention
    @ModuleInfo(key: "norm3") var crossNorm: Wan2LayerNorm
    @ModuleInfo(key: "cross_attn") var crossAttention: Wan2CrossAttention
    @ModuleInfo(key: "norm2") var feedForwardNorm: Wan2LayerNorm
    @ModuleInfo(key: "ffn") var feedForward: Wan2FeedForward
    @ModuleInfo(key: "cam_self_attn") var cameraAttention: Wan2ProjectiveSelfAttention?
    @ModuleInfo(key: "modulation") var modulation: MLXArray

    init(configuration: Wan2TransformerConfiguration) {
        let dimensions = configuration.hiddenSize
        self._selfNorm.wrappedValue = Wan2LayerNorm(dimensions: dimensions, epsilon: configuration.epsilon)
        self._selfAttention.wrappedValue = Wan2SelfAttention(
            dimensions: dimensions,
            heads: configuration.headCount,
            epsilon: configuration.epsilon
        )
        self._crossNorm.wrappedValue = Wan2LayerNorm(
            dimensions: dimensions,
            epsilon: configuration.epsilon,
            affine: true
        )
        self._crossAttention.wrappedValue = Wan2CrossAttention(
            dimensions: dimensions,
            heads: configuration.headCount,
            epsilon: configuration.epsilon
        )
        self._feedForwardNorm.wrappedValue = Wan2LayerNorm(dimensions: dimensions, epsilon: configuration.epsilon)
        self._feedForward.wrappedValue = Wan2FeedForward(
            dimensions: dimensions,
            hiddenDimensions: configuration.feedForwardSize
        )
        self._cameraAttention.wrappedValue = configuration.projectiveCameraConditioning
            ? Wan2ProjectiveSelfAttention(
                dimensions: dimensions,
                attentionDimensions: dimensions / configuration.projectiveCameraAttentionCompression,
                heads: configuration.headCount,
                epsilon: configuration.epsilon
            )
            : nil
        self._modulation.wrappedValue = MLX.zeros([1, 6, dimensions], dtype: .float32)
    }

    func callAsFunction(
        _ input: MLXArray,
        modulationInput: MLXArray,
        context: MLXArray,
        grid: Wan2GridSize,
        rope: Wan2RoPE.Cache,
        selfMask: MLXArray?,
        crossCache: Wan2AttentionKVCache?,
        cameraConditioning: Wan2ProjectiveCameraConditioning?
    ) -> MLXArray {
        let parts = MLX.split(modulation.expandedDimensions(axis: 1) + modulationInput, parts: 6, axis: 2)
            .map { $0.squeezed(axis: 2) }
        var hidden = input
        let hiddenType = hidden.dtype
        let selfInput = selfNorm(hidden.asType(.float32)) * (1 + parts[1]) + parts[0]
        var selfOutput = selfAttention(
            selfInput.asType(hiddenType),
            grid: grid,
            rope: rope,
            mask: selfMask
        )
        if let cameraAttention, let cameraConditioning {
            precondition(cameraConditioning.frameCount == grid.frames)
            selfOutput = selfOutput + cameraAttention(
                selfInput.asType(hiddenType),
                conditioning: cameraConditioning
            )
        }
        hidden = (hidden.asType(.float32) + selfOutput * parts[2]).asType(hiddenType)

        let crossInput = crossNorm(hidden.asType(.float32)).asType(hiddenType)
        hidden = hidden + crossAttention(crossInput, context: context, cache: crossCache)

        let feedForwardInput = feedForwardNorm(hidden.asType(.float32)) * (1 + parts[4]) + parts[3]
        let feedForwardOutput = feedForward(feedForwardInput.asType(hiddenType))
        hidden = (hidden.asType(.float32) + feedForwardOutput.asType(.float32) * parts[5])
            .asType(hiddenType)
        return hidden
    }
}

final class Wan2OutputHead: Module {
    let outputChannels: Int
    let patchSize: [Int]
    @ModuleInfo(key: "norm") var norm: Wan2LayerNorm
    @ModuleInfo(key: "head") var projection: Linear
    @ModuleInfo(key: "modulation") var modulation: MLXArray

    init(configuration: Wan2TransformerConfiguration) {
        self.outputChannels = configuration.outputChannels
        self.patchSize = configuration.patchSize
        let projectionSize = configuration.patchSize.reduce(1, *) * configuration.outputChannels
        self._norm.wrappedValue = Wan2LayerNorm(dimensions: configuration.hiddenSize, epsilon: configuration.epsilon)
        self._projection.wrappedValue = Linear(configuration.hiddenSize, projectionSize, bias: true)
        self._modulation.wrappedValue = MLX.zeros([1, 2, configuration.hiddenSize], dtype: .float32)
    }

    func callAsFunction(_ input: MLXArray, timestepEmbedding: MLXArray) -> MLXArray {
        let embedding = timestepEmbedding.ndim == 2
            ? timestepEmbedding.expandedDimensions(axis: 1)
            : timestepEmbedding
        let parts = MLX.split(
            modulation.expandedDimensions(axis: 1) + embedding.expandedDimensions(axis: 2),
            parts: 2,
            axis: 2
        ).map { $0.squeezed(axis: 2) }
        let normalized = norm(input.asType(.float32)) * (1 + parts[1]) + parts[0]
        return projection(normalized.asType(timestepEmbedding.dtype))
    }
}

public final class Wan2TransformerModel: Module {
    public let configuration: Wan2TransformerConfiguration
    let inverseTimestepFrequencies: MLXArray
    let ropeFrequencies: MLXArray

    @ModuleInfo(key: "patch_embedding_proj") var patchProjection: Linear
    @ModuleInfo(key: "text_embedding_0") var textProjectionIn: Linear
    @ModuleInfo(key: "text_embedding_1") var textProjectionOut: Linear
    @ModuleInfo(key: "time_embedding_0") var timestepProjectionIn: Linear
    @ModuleInfo(key: "time_embedding_1") var timestepProjectionOut: Linear
    @ModuleInfo(key: "time_projection") var modulationProjection: Linear
    @ModuleInfo(key: "blocks") var blocks: [Wan2TransformerBlock]
    @ModuleInfo(key: "head") var outputHead: Wan2OutputHead

    public init(configuration: Wan2TransformerConfiguration = Wan2TransformerConfiguration()) {
        self.configuration = configuration
        let patchVolume = configuration.patchSize.reduce(1, *)
        self._patchProjection.wrappedValue = Linear(
            configuration.inputChannels * patchVolume,
            configuration.hiddenSize,
            bias: true
        )
        self._textProjectionIn.wrappedValue = Linear(
            configuration.textEmbeddingSize,
            configuration.hiddenSize,
            bias: true
        )
        self._textProjectionOut.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.hiddenSize,
            bias: true
        )
        self._timestepProjectionIn.wrappedValue = Linear(
            configuration.timestepFrequencySize,
            configuration.hiddenSize,
            bias: true
        )
        self._timestepProjectionOut.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.hiddenSize,
            bias: true
        )
        self._modulationProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.hiddenSize * 6,
            bias: true
        )
        self._blocks.wrappedValue = (0..<configuration.layerCount).map { _ in
            Wan2TransformerBlock(configuration: configuration)
        }
        self._outputHead.wrappedValue = Wan2OutputHead(configuration: configuration)

        let half = configuration.timestepFrequencySize / 2
        self.inverseTimestepFrequencies = MLXArray((0..<half).map {
            pow(10_000, -Float($0) / Float(half))
        })
        let headDimension = configuration.hiddenSize / configuration.headCount
        let sixth = headDimension / 6
        self.ropeFrequencies = Wan2RoPE.frequencies(
            maxSequence: 1_024,
            dimensions: [headDimension - 4 * sixth, 2 * sixth, 2 * sixth]
        )
    }

    public func embedText(_ embeddings: MLXArray) -> MLXArray {
        precondition(embeddings.ndim == 3)
        let context: MLXArray
        if embeddings.dim(1) < configuration.textLength {
            let padding = MLX.zeros(
                [embeddings.dim(0), configuration.textLength - embeddings.dim(1), embeddings.dim(2)],
                dtype: embeddings.dtype
            )
            context = MLX.concatenated([embeddings, padding], axis: 1)
        } else {
            context = embeddings[0..., 0..<configuration.textLength]
        }
        return textProjectionOut(MLXNN.geluApproximate(textProjectionIn(context)))
            .asType(patchProjection.weight.dtype)
    }

    public func prepareCrossAttentionCaches(context: MLXArray) -> [Wan2AttentionKVCache] {
        blocks.map { $0.crossAttention.prepareCache(context: context) }
    }

    public func callAsFunction(
        latents: [MLXArray],
        timesteps: MLXArray,
        embeddedContext: MLXArray,
        crossCaches: [Wan2AttentionKVCache]? = nil,
        cameraConditioning: Wan2ProjectiveCameraConditioning? = nil,
        causalState: Wan2CausalTransformerState? = nil,
        currentStartToken: Int = 0
    ) -> [MLXArray] {
        precondition(!latents.isEmpty)
        let patchified = latents.map(patchify)
        let grid = patchified[0].grid
        precondition(patchified.allSatisfy { $0.grid == grid })
        var hidden = MLX.concatenated(patchified.map(\.value), axis: 0)

        let sinusoid = timesteps.asType(.float32).expandedDimensions(axis: -1) * inverseTimestepFrequencies
        let frequencyEmbedding = MLX.concatenated([MLX.cos(sinusoid), MLX.sin(sinusoid)], axis: -1)
            .asType(timestepProjectionIn.weight.dtype)
        let timestepEmbedding = timestepProjectionOut(MLXNN.silu(timestepProjectionIn(frequencyEmbedding)))
            .asType(embeddedContext.dtype)
        let modulation = modulationProjection(MLXNN.silu(timestepEmbedding)).asType(.float32)
            .reshaped(timesteps.dim(0), timesteps.ndim == 1 ? 1 : timesteps.dim(1), 6, configuration.hiddenSize)

        let context = embeddedContext.dim(0) == hidden.dim(0)
            ? embeddedContext
            : MLX.broadcast(
                embeddedContext,
                to: [hidden.dim(0), embeddedContext.dim(1), embeddedContext.dim(2)]
            )
        let rope = Wan2RoPE.prepare(grid: grid, frequencies: ropeFrequencies, dtype: patchProjection.weight.dtype)
        if let causalState {
            precondition(causalState.blocks.count == blocks.count)
        }
        for (index, block) in blocks.enumerated() {
            if let causalState {
                hidden = block.callCausal(
                    hidden,
                    modulationInput: modulation,
                    context: context,
                    grid: grid,
                    frequencies: ropeFrequencies,
                    crossCache: crossCaches?[index],
                    cameraConditioning: cameraConditioning,
                    state: causalState.blocks[index],
                    currentStartToken: currentStartToken
                )
            } else {
                hidden = block(
                    hidden,
                    modulationInput: modulation,
                    context: context,
                    grid: grid,
                    rope: rope,
                    selfMask: nil,
                    crossCache: crossCaches?[index],
                    cameraConditioning: cameraConditioning
                )
            }
        }
        let projected = outputHead(hidden, timestepEmbedding: timestepEmbedding)
        return unpatchify(projected, grid: grid)
    }

    private func patchify(_ latent: MLXArray) -> (value: MLXArray, grid: Wan2GridSize) {
        let patch = Wan2PatchLayout.flatten(latent, patchSize: configuration.patchSize)
        let projected = patchProjection(patch.value).asType(patchProjection.weight.dtype)
        return (projected.expandedDimensions(axis: 0), patch.grid)
    }

    private func unpatchify(_ patches: MLXArray, grid: Wan2GridSize) -> [MLXArray] {
        let temporalPatch = configuration.patchSize[0]
        let heightPatch = configuration.patchSize[1]
        let widthPatch = configuration.patchSize[2]
        return (0..<patches.dim(0)).map { index in
            patches[index, 0..<grid.sequenceLength]
                .reshaped(
                    grid.frames,
                    grid.height,
                    grid.width,
                    temporalPatch,
                    heightPatch,
                    widthPatch,
                    configuration.outputChannels
                )
                .transposed(6, 0, 3, 1, 4, 2, 5)
                .reshaped(
                    configuration.outputChannels,
                    grid.frames * temporalPatch,
                    grid.height * heightPatch,
                    grid.width * widthPatch
                )
                .asType(.float32)
        }
    }
}
